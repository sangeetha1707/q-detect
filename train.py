import os
import argparse
import random
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pennylane as qml
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Torch version:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}")
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_qubits = 6  
num_layers = 6  

# Try GPU-backed PennyLane lightning device; fallback to default CPU simulator
try:
    dev = qml.device("lightning.gpu", wires=num_qubits)
    print("Using PennyLane lightning.gpu backend")
except Exception:
    dev = qml.device("default.qubit", wires=num_qubits)
    print("Using PennyLane default.qubit backend")

def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

weight_shapes = {"weights": (num_layers, num_qubits, 3)} 
qnode = qml.QNode(quantum_circuit, dev, interface="torch")

class EnhancedQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.scale = nn.Parameter(torch.tensor(0.1)) 

    def forward(self, x):
        
        scaled_input = x * self.scale
        return self.q_layer(scaled_input)

 

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Extra augmentation applied conditionally to adenocarcinoma samples (if enabled)
extra_aden_aug = transforms.RandomApply([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
], p=0.5)

# Use project-local Data directory by default
data_dir = "Data"  
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid") 
test_dir = os.path.join(data_dir, "test")

if not (os.path.isdir(train_dir) and os.path.isdir(valid_dir) and os.path.isdir(test_dir)):
    raise RuntimeError(
        f"Data folders not found. Expected at '{data_dir}' with subfolders 'train', 'valid', 'test'."
    )

# --- Dataset with unified class mapping across splits ---
class LungCancerDataset(Dataset):
    def __init__(self, base_path, transform=None, class_to_idx=None, class_names=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = [] if class_names is None else list(class_names)
        self.class_to_idx = None if class_to_idx is None else dict(class_to_idx)
        # targeted augmentation controls
        self.enable_aug_adenocarcinoma = False
        self.aden_idx = None

        if self.class_to_idx is None:
            # Build from directories under this split
            cancer_types = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            self.class_names = sorted(cancer_types)
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        else:
            # Use provided mapping for consistent labels across splits
            if not self.class_names:
                # Derive class_names ordered by index
                inv = sorted(self.class_to_idx.items(), key=lambda kv: kv[1])
                self.class_names = [k for k, _ in inv]

        # Build simple alias mapping to handle folder name differences (e.g., 'adenocarcinoma' -> 'adenocarcinoma_*')
        aliases = {}
        train_base_tokens = {cn.split('_')[0]: cn for cn in self.class_names}

        for d in os.listdir(base_path):
            full = os.path.join(base_path, d)
            if not os.path.isdir(full):
                continue
            if d in self.class_to_idx:
                aliases[d] = d
            else:
                base = d.split('_')[0]
                if base in train_base_tokens:
                    aliases[d] = train_base_tokens[base]

        skipped_dirs = []
        for d in os.listdir(base_path):
            class_dir = os.path.join(base_path, d)
            if not os.path.isdir(class_dir):
                continue
            mapped = aliases.get(d)
            if mapped is None or mapped not in self.class_to_idx:
                skipped_dirs.append(d)
                continue
            class_idx = self.class_to_idx[mapped]
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith((
                    '.png', '.jpg', '.jpeg', '.tif', '.tiff'  # exclude .dcm by default
                )):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

        if len(self.image_paths) == 0:
            raise ValueError(f"❌ No images found in {base_path}! Checked dirs: {os.listdir(base_path)}")

        if skipped_dirs:
            print(f"[Dataset] Skipped unmatched folders in '{base_path}': {sorted(set(skipped_dirs))}")

        print(f"Found {len(self.image_paths)} images in {len(set(self.labels))} mapped classes. Base: {base_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        if self.enable_aug_adenocarcinoma and (self.aden_idx is not None) and (label == self.aden_idx):
            image = extra_aden_aug(image)
        return image, label

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--finetune', action='store_true', help='Fine-tune from best checkpoint')
    p.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    p.add_argument('--use-focal', action='store_true', help='Use Focal Loss instead of CrossEntropy')
    p.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    p.add_argument('--amp', action='store_true', help='Enable mixed precision training on CUDA')
    p.add_argument('--batch-size', type=int, default=0, help='Override batch size (0=auto)')
    p.add_argument('--balance', type=str, choices=['sampler','weights','none'], default='sampler', help='Class balancing strategy')
    p.add_argument('--no-quantum', action='store_true', help='Disable quantum layer (ablation)')
    p.add_argument('--temp-scale', action='store_true', help='Apply temperature scaling using validation set')
    p.add_argument('--augment-adenocarcinoma', action='store_true', help='Enable targeted extra augmentation for adenocarcinoma')
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument('--scheduler', type=str, choices=['plateau','cosine'], default='plateau', help='LR scheduler type')
    p.add_argument('--patience', type=int, default=8, help='Early stopping patience')
    p.add_argument('--clip-grad', type=float, default=1.0, help='Gradient clipping max norm')
    p.add_argument('--warmup-epochs', type=int, default=0, help='Freeze quantum layer for N warmup epochs')
    p.add_argument('--bottleneck-dropout', type=float, default=0.3, help='Dropout rate in bottleneck block')
    p.add_argument('--ema', action='store_true', help='Enable Exponential Moving Average of model params')
    p.add_argument('--ema-decay', type=float, default=0.999, help='EMA decay')
    p.add_argument('--swa', action='store_true', help='Enable Stochastic Weight Averaging (final N epochs)')
    p.add_argument('--swa-start-epoch', type=int, default=10, help='Epoch to start SWA averaging')
    p.add_argument('--mc-dropout', action='store_true', help='Keep dropout active during validation for uncertainty')
    return p

args = build_argparser().parse_args() if __name__ == '__main__' else argparse.Namespace(
    finetune=False, epochs=None, use_focal=False, gamma=2.0, amp=False, batch_size=0, balance='sampler', no_quantum=False, temp_scale=False, augment_adenocarcinoma=False,
    seed=42, scheduler='plateau', patience=8, clip_grad=1.0, warmup_epochs=0, bottleneck_dropout=0.3,
    ema=False, ema_decay=0.999, swa=False, swa_start_epoch=10, mc_dropout=False
)

set_seed(args.seed)

# Build datasets using training class mapping for all splits
train_dataset = LungCancerDataset(train_dir, transform=transform_train)
fixed_class_to_idx = train_dataset.class_to_idx
fixed_class_names = train_dataset.class_names

valid_dataset = LungCancerDataset(
    valid_dir, transform=transform_val, class_to_idx=fixed_class_to_idx, class_names=fixed_class_names
)
test_dataset = LungCancerDataset(
    test_dir, transform=transform_val, class_to_idx=fixed_class_to_idx, class_names=fixed_class_names
)

class_counts = [train_dataset.labels.count(i) for i in range(len(train_dataset.class_names))]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for label in train_dataset.labels]
weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# Targeted augmentation enabling
if args.augment_adenocarcinoma:
    aden_name = None
    for name in fixed_class_names:
        if name.startswith('adenocarcinoma'):
            aden_name = name
            break
    if aden_name is not None:
        train_dataset.enable_aug_adenocarcinoma = True
        train_dataset.aden_idx = fixed_class_to_idx[aden_name]
        print(f"Targeted augmentation enabled for class: {aden_name}")

# Batch size with AMP scaling
batch_size = (8 if device.type == 'cpu' else 16)
if args.batch_size and args.batch_size > 0:
    batch_size = args.batch_size
elif args.amp and device.type == 'cuda':
    batch_size = max(batch_size, 32)

pin_memory_flag = device.type == 'cuda'

if args.balance == 'sampler':
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=0, pin_memory=pin_memory_flag)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory_flag)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory_flag)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory_flag)

class QuantumHybridModel(nn.Module):
    def __init__(self, num_classes, use_quantum: bool = True, bottleneck_dropout: float = 0.3):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet50(weights=None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(bottleneck_dropout)
        )

        self.to_q = nn.Linear(256, num_qubits)
        self.quantum_layer = EnhancedQuantumLayer()

        self.classifier = nn.Sequential(
            nn.Linear(num_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        self.classifier_noq = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.use_quantum = use_quantum

    def forward(self, x):
        features = self.backbone(x)
        z = self.bottleneck(features)
        if self.use_quantum:
            q_in = self.to_q(z)
            q_out = self.quantum_layer(q_in)
            logits = self.classifier(q_out)
        else:
            logits = self.classifier_noq(z)
        return logits

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

model = QuantumHybridModel(len(train_dataset.class_names), use_quantum=(not args.no_quantum), bottleneck_dropout=args.bottleneck_dropout).to(device)

# choose loss weights only when balance == 'weights'
loss_weight = class_weights.to(device) if args.balance == 'weights' else None
criterion = (FocalLoss(weight=loss_weight, gamma=args.gamma)
             if args.use_focal else nn.CrossEntropyLoss(weight=loss_weight))

# Adjust LRs for fine-tuning
base_lr = 1e-3
backbone_lr = 5e-5
quantum_lr = 5e-3
if args.finetune:
    base_lr = 5e-4
    backbone_lr = 1e-5
    quantum_lr = 1e-3

optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': backbone_lr}, 
    {'params': model.bottleneck.parameters(), 'lr': base_lr},
    {'params': model.to_q.parameters(), 'lr': base_lr},
    {'params': model.quantum_layer.parameters(), 'lr': quantum_lr}, 
    {'params': model.classifier.parameters(), 'lr': base_lr},
    {'params': model.classifier_noq.parameters(), 'lr': base_lr}
], lr=base_lr, weight_decay=1e-5)

if args.scheduler == 'plateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
else:  # cosine
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# SWA setup (wrapper scheduler if enabled)
swa_model = None
swa_scheduler = None
if args.swa:
    from torch.optim.swa_utils import AveragedModel, SWALR
    swa_model = AveragedModel(model)
    # SWALR used only after start epoch; keep base scheduler until then
    swa_scheduler = SWALR(optimizer, anneal_strategy='cos', anneal_epochs=5, swa_lr=base_lr * 0.5)

# EMA state
ema_shadow = None
if args.ema:
    ema_shadow = {name: param.detach().clone() for name, param in model.state_dict().items() if param.dtype.is_floating_point}

def update_ema(model, ema_shadow, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in msd.items():
            if k in ema_shadow and v.dtype.is_floating_point:
                ema_shadow[k].mul_(decay).add_(v, alpha=1 - decay)

def apply_ema(model, ema_shadow):
    msd = model.state_dict()
    for k, v in msd.items():
        if k in ema_shadow and v.dtype.is_floating_point:
            v.copy_(ema_shadow[k])

def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False, clip_grad: float = 1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if (use_amp and device.type == 'cuda') else nullcontext()
        with amp_ctx:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1

def validate(model, dataloader, criterion, device, temperature: float = 1.0):
    # mc_dropout allows dropout layers to stay active for stochastic predictions
    if args.mc_dropout:
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.eval()
    else:
        model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            outputs = outputs / temperature
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return running_loss / len(dataloader), accuracy, precision, recall, f1, cm, all_preds, all_labels

def per_class_metrics(labels, preds, class_names):
    metrics = []
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    for idx, cname in enumerate(class_names):
        tp = np.sum((labels_arr == idx) & (preds_arr == idx))
        fp = np.sum((labels_arr != idx) & (preds_arr == idx))
        fn = np.sum((labels_arr == idx) & (preds_arr != idx))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics.append((cname, precision, recall, f1))
    return metrics

num_epochs = 20 if device.type == 'cpu' else 30  
if args.finetune and args.epochs is None:
    num_epochs = 10
elif args.epochs is not None:
    num_epochs = args.epochs
early_stop_patience = args.patience
early_stop_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
           'train_f1': [], 'val_f1': [], 'lr': []}

print(f"Starting training for {num_epochs} epochs...")
if args.finetune:
    try:
        ckpt = torch.load('best_quantum_lung_cancer_model.pth', map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded checkpoint for fine-tuning (epoch {ckpt.get('epoch','?')})")
    except Exception as e:
        print(f"Fine-tune: no checkpoint found or failed to load: {e}")

# Warmup: freeze quantum layer if requested
if args.warmup_epochs > 0 and model.use_quantum:
    for p in model.quantum_layer.parameters():
        p.requires_grad = False
    print(f"Quantum layer frozen for first {args.warmup_epochs} warmup epochs.")

scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == 'cuda'))

try:
    best_val_f1 = float("-inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=args.amp and device.type=='cuda', clip_grad=args.clip_grad
        )

        val_loss, val_acc, val_prec, val_rec, val_f1, conf_matrix, val_preds, val_labels = validate(
            model, valid_loader, criterion, device
        )

        # EMA update after training batch epoch
        if args.ema:
            update_ema(model, ema_shadow, args.ema_decay)

        # SWA update if enabled and reached start epoch
        if args.swa and epoch + 1 >= args.swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if args.scheduler == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.4f} | Valid F1: {val_f1:.4f}")
        print(f"Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | LR: {current_lr:.6f}")

        # Per-class metrics
        pcm = per_class_metrics(val_labels, val_preds, train_dataset.class_names)
        for cname, p, r, f in pcm:
            print(f"  [Class] {cname[:25]:25s} | P {p:.3f} R {r:.3f} F1 {f:.3f}")

        # Unfreeze quantum after warmup
        if args.warmup_epochs > 0 and epoch + 1 == args.warmup_epochs:
            if model.use_quantum:
                for p in model.quantum_layer.parameters():
                    p.requires_grad = True
                print("Quantum layer unfrozen; starting joint training.")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0
            # If EMA enabled, store EMA weights for best ckpt
            save_state = model.state_dict()
            if args.ema:
                # Temporarily apply EMA to save
                original = {k: v.detach().clone() for k, v in save_state.items()}
                apply_ema(model, ema_shadow)
                save_state = model.state_dict()
                # Revert
                for k, v in model.state_dict().items():
                    if k in original:
                        v.copy_(original[k])
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': save_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'class_names': train_dataset.class_names,
                'ema': args.ema,
                'swa': args.swa
            }, 'best_quantum_lung_cancer_model.pth')
            print("✅ Model improved and saved!")
        else:
            early_stop_counter += 1
            print(f"⚠️ No improvement for {early_stop_counter} epochs")
            
            if early_stop_counter >= early_stop_patience:
                print("⛔ Early stopping triggered. Training halted.")
                break
        
        print("-" * 60)
except KeyboardInterrupt:
    print("Training interrupted by user.")

print("\nEvaluating on test set...")
try:

    checkpoint = torch.load('best_quantum_lung_cancer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded best model from epoch {checkpoint['epoch']} with F1 score: {checkpoint['val_f1']:.4f}")
except Exception as e:
    print(f"Warning: Could not load best model. Using current model instead. {e}")

# If SWA enabled, swap model params to SWA averaged version before test if beyond start epoch
if args.swa and swa_model is not None and num_epochs >= args.swa_start_epoch:
    print("Applying SWA averaged parameters for final evaluation.")
    model.load_state_dict(swa_model.state_dict(), strict=False)

temperature = 1.0
def calibrate_temperature(model, valid_loader):
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            logits_list.append(logits.detach())
            labels_list.append(y.detach())
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    log_T = torch.zeros(1, device=device, requires_grad=True)
    opt = optim.Adam([log_T], lr=0.01)
    for _ in range(200):
        opt.zero_grad()
        T = torch.nn.functional.softplus(log_T) + 1e-6
        loss = nn.functional.cross_entropy(logits_all / T, labels_all)
        loss.backward()
        opt.step()
    return (torch.nn.functional.softplus(log_T) + 1e-6).item()

if args.temp_scale:
    try:
        temperature = calibrate_temperature(model, valid_loader)
        print(f"Calibrated temperature: {temperature:.3f}")
        import json
        with open('calibration.json', 'w') as f:
            json.dump({'temperature': temperature}, f)
    except Exception as e:
        print(f"Temperature scaling failed: {e}")

test_loss, test_acc, test_prec, test_rec, test_f1, test_cm, test_preds, test_labels = validate(
    model, test_loader, criterion, device, temperature=temperature
)

print(f"Test Results:")
print(f"Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1 Score: {test_f1:.4f}")


try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=train_dataset.class_names,
                yticklabels=train_dataset.class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
except Exception as e:
    print(f"Could not save confusion matrix: {e}")


try:
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['train_f1'], label='Train')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('F1 Score')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
except Exception as e:
    print(f"Could not save training history plot: {e}")

print("🎉 Training and evaluation complete!")
print(f"Model saved as best_quantum_lung_cancer_model.pth")