import os
import argparse
from contextlib import nullcontext
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import pennylane as qml
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

num_qubits = 6
num_layers = 6

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
        return self.q_layer(x * self.scale)

transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = "Data"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")

if not (os.path.isdir(train_dir) and os.path.isdir(valid_dir) and os.path.isdir(test_dir)):
    raise RuntimeError("Data folders missing")

class LungCancerDataset(Dataset):
    def __init__(self, base_path, transform=None, class_to_idx=None, class_names=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = [] if class_names is None else list(class_names)
        self.class_to_idx = None if class_to_idx is None else dict(class_to_idx)

        if self.class_to_idx is None:
            cancer_types = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            self.class_names = sorted(cancer_types)
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        else:
            if not self.class_names:
                inv = sorted(self.class_to_idx.items(), key=lambda kv: kv[1])
                self.class_names = [k for k, _ in inv]

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

        skipped = []
        for d in os.listdir(base_path):
            class_dir = os.path.join(base_path, d)
            if not os.path.isdir(class_dir):
                continue
            mapped = aliases.get(d)
            if mapped is None or mapped not in self.class_to_idx:
                skipped.append(d)
                continue
            class_idx = self.class_to_idx[mapped]
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('png','jpg','jpeg','tif','tiff')):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        if skipped:
            print(f"[Dataset] Skipped: {sorted(set(skipped))}")
        print(f"Found {len(self.image_paths)} images in {len(set(self.labels))} classes. Base: {base_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = LungCancerDataset(train_dir, transform=transform_train)
fixed_class_to_idx = train_dataset.class_to_idx
fixed_class_names = train_dataset.class_names
valid_dataset = LungCancerDataset(valid_dir, transform=transform_val, class_to_idx=fixed_class_to_idx, class_names=fixed_class_names)
test_dataset = LungCancerDataset(test_dir, transform=transform_val, class_to_idx=fixed_class_to_idx, class_names=fixed_class_names)

class_counts = [train_dataset.labels.count(i) for i in range(len(train_dataset.class_names))]
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[l] for l in train_dataset.labels]
weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=0)
parser.add_argument('--no-sampler', action='store_true', help='Disable sampler and use class weights only')
args = parser.parse_args() if __name__ == '__main__' else parser.parse_args([])

batch_size = 8 if device.type=='cpu' else 16
if args.amp and device.type=='cuda':
    batch_size = 32
if args.batch_size > 0:
    batch_size = args.batch_size

pin_memory = device.type=='cuda'
if args.no_sampler:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory)
    loss_weight = class_weights.to(device)
else:
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=0, pin_memory=pin_memory)
    loss_weight = None  # sampler handles balancing

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

class QuantumHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet50(weights=None)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dim_reduction = nn.Sequential(
            nn.Linear(feat, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_qubits)
        )
        self.quantum_layer = EnhancedQuantumLayer()
        self.classifier = nn.Sequential(
            nn.Linear(num_qubits, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    def forward(self,x):
        f = self.backbone(x)
        q_in = self.dim_reduction(f)
        q_out = self.quantum_layer(q_in)
        return self.classifier(q_out)

model = QuantumHybridModel(len(train_dataset.class_names)).to(device)
criterion = nn.CrossEntropyLoss(weight=loss_weight)
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': args.lr/20},
    {'params': model.dim_reduction.parameters(), 'lr': args.lr},
    {'params': model.quantum_layer.parameters(), 'lr': args.lr*5},
    {'params': model.classifier.parameters(), 'lr': args.lr}
], lr=args.lr, weight_decay=1e-5)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type=='cuda'))

history = {k: [] for k in ['train_loss','train_acc','val_loss','val_acc','train_f1','val_f1','lr']}

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total=0; correct=0; run_loss=0.0; all_preds=[]; all_labels=[]
    ctx = nullcontext()
    with torch.set_grad_enabled(train):
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True) if train else None
            amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if (args.amp and device.type=='cuda') else ctx
            with amp_ctx:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            if train:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            run_loss += loss.item()
            _,pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            all_preds.extend(pred.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    acc = correct/total
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return run_loss/len(loader), acc, prec, rec, f1

num_epochs = args.epochs
print(f"Baseline training for {num_epochs} epochs (batch={batch_size}, amp={args.amp})...")
best_val_f1 = float('-inf'); early_patience=8; patience_counter=0
for epoch in range(num_epochs):
    t0=time.time()
    tr_loss, tr_acc, tr_prec, tr_rec, tr_f1 = run_epoch(train_loader, train=True)
    val_loss, val_acc, val_prec, val_rec, val_f1 = run_epoch(valid_loader, train=False)
    scheduler.step(val_loss)
    history['train_loss'].append(tr_loss); history['train_acc'].append(tr_acc)
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
    history['train_f1'].append(tr_f1); history['val_f1'].append(val_f1)
    history['lr'].append(optimizer.param_groups[0]['lr'])
    dt=time.time()-t0
    print(f"Epoch {epoch+1}/{num_epochs} | {dt:.2f}s")
    print(f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} F1 {tr_f1:.4f}")
    print(f"Valid Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}")
    if val_f1>best_val_f1:
        best_val_f1=val_f1; patience_counter=0
        torch.save({'epoch':epoch+1,'model_state_dict':model.state_dict(),'val_f1':val_f1,'class_names':train_dataset.class_names},'best_quantum_lung_cancer_model.pth')
        print('✅ Improved & saved')
    else:
        patience_counter+=1
        print(f"⚠️ No improvement for {patience_counter} epochs")
        if patience_counter>=early_patience:
            print('⛔ Early stopping'); break
    print('-'*60)

print('\nEvaluating baseline best on test set...')
try:
    ckpt=torch.load('best_quantum_lung_cancer_model.pth'); model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded best from epoch {ckpt['epoch']} (val F1 {ckpt['val_f1']:.4f})")
except Exception as e:
    print(f"Could not load checkpoint: {e}")

# Test evaluation
model.eval();
all_preds=[]; all_labels=[]; test_loss=0.0
with torch.no_grad():
    for x,y in test_loader:
        x=x.to(device); y=y.to(device)
        out=model(x); loss=criterion(out,y); test_loss+=loss.item()
        _,p=out.max(1); all_preds.extend(p.cpu().numpy()); all_labels.extend(y.cpu().numpy())
import numpy as np
prec=precision_score(all_labels,all_preds,average='weighted',zero_division=0)
rec=recall_score(all_labels,all_preds,average='weighted',zero_division=0)
f1=f1_score(all_labels,all_preds,average='weighted',zero_division=0)
acc=(np.array(all_preds)==np.array(all_labels)).mean()
cm=confusion_matrix(all_labels,all_preds)
print(f"Test Loss {test_loss/len(test_loader):.4f} Acc {acc:.4f} Precision {prec:.4f} Recall {rec:.4f} F1 {f1:.4f}")

# Save confusion matrix
try:
    plt.figure(figsize=(8,6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.class_names, yticklabels=train_dataset.class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Baseline Confusion Matrix'); plt.tight_layout(); plt.savefig('confusion_matrix.png')
    print('Confusion matrix saved (baseline)')
except Exception as e:
    print(f'Could not save confusion matrix: {e}')

print('Baseline training/evaluation complete.')
