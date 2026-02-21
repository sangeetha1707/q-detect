import os
import glob
import argparse
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns

# Basic dataset for inference
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, class_to_idx=None, class_names=None):
        self.transform = transform
        self.paths = []
        self.labels = []
        self.class_names = [] if class_names is None else list(class_names)
        self.class_to_idx = None if class_to_idx is None else dict(class_to_idx)
        if self.class_to_idx is None:
            cancer_types = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))]
            self.class_names = sorted(cancer_types)
            self.class_to_idx = {c:i for i,c in enumerate(self.class_names)}
        else:
            if not self.class_names:
                inv = sorted(self.class_to_idx.items(), key=lambda kv: kv[1])
                self.class_names = [k for k,_ in inv]
        aliases = {cn.split('_')[0]: cn for cn in self.class_names}
        for d in os.listdir(root):
            full = os.path.join(root,d)
            if not os.path.isdir(full):
                continue
            key = d if d in self.class_to_idx else aliases.get(d.split('_')[0])
            if key is None: continue
            idx = self.class_to_idx[key]
            for f in os.listdir(full):
                p = os.path.join(full,f)
                if p.lower().endswith(('png','jpg','jpeg','tif','tiff')):
                    self.paths.append(p)
                    self.labels.append(idx)
        if len(self.paths)==0:
            raise RuntimeError(f"No images found in {root}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

# Quantum layer reconstruction (must match training hyperparams)
num_qubits = 6
num_layers = 6
try:
    dev = qml.device('lightning.gpu', wires=num_qubits)
except Exception:
    dev = qml.device('default.qubit', wires=num_qubits)

def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

weight_shapes = {"weights": (num_layers, num_qubits, 3)}
qnode = qml.QNode(quantum_circuit, dev, interface='torch')

class EnhancedQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.scale = nn.Parameter(torch.tensor(0.1))
    def forward(self, x):
        return self.q_layer(x * self.scale)

class QuantumHybridModel(nn.Module):
    def __init__(self, num_classes, use_quantum=True):
        super().__init__()
        from torchvision import models
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet50(weights=None)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.bottleneck = nn.Sequential(
            nn.Linear(feat, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
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
    def forward(self,x):
        f = self.backbone(x)
        z = self.bottleneck(f)
        if self.use_quantum:
            q_in = self.to_q(z)
            q_out = self.quantum_layer(q_in)
            return self.classifier(q_out)
        else:
            return self.classifier_noq(z)

# Expected calibration file
CALIB_FILE = 'calibration.json'

def load_temperature():
    if os.path.isfile(CALIB_FILE):
        try:
            with open(CALIB_FILE,'r') as f:
                return json.load(f).get('temperature',1.0)
        except Exception:
            pass
    return 1.0

def ece_score(probs, labels, n_bins=15):
    # Expected Calibration Error
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    confidences = probs.max(1)
    predictions = probs.argmax(1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)

def tta_batch(model, imgs, tta_transforms):
    # imgs: tensor BxCxHxW
    augmented_logits = []
    for t in tta_transforms:
        aug = t(imgs)
        augmented_logits.append(model(aug))
    return torch.stack(augmented_logits, dim=0)  # T x B x C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=str, default='Data/test')
    ap.add_argument('--checkpoints', type=str, default='best_quantum_lung_cancer_model.pth', help='Comma separated list or glob pattern')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--no-quantum', action='store_true')
    ap.add_argument('--tta', action='store_true', help='Enable test-time augmentation averaging')
    ap.add_argument('--tta-count', type=int, default=4, help='Number of stochastic augmentations')
    ap.add_argument('--mc-dropout', action='store_true', help='Enable MC dropout for uncertainty (multiple forward passes)')
    ap.add_argument('--mc-samples', type=int, default=5)
    ap.add_argument('--save-json', type=str, default='ensemble_results.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = SimpleDataset(os.path.abspath(args.data_dir), base_transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type=='cuda'))

    # Resolve checkpoints list
    ckpt_paths = []
    if ',' in args.checkpoints:
        ckpt_paths = [p.strip() for p in args.checkpoints.split(',') if p.strip()]
    elif any(ch in args.checkpoints for ch in ['*','?']):
        ckpt_paths = glob.glob(args.checkpoints)
    else:
        ckpt_paths = [args.checkpoints]
    ckpt_paths = [p for p in ckpt_paths if os.path.isfile(p)]
    if len(ckpt_paths)==0:
        raise RuntimeError('No checkpoint files found to ensemble.')
    print(f'Ensembling {len(ckpt_paths)} checkpoints:')
    for p in ckpt_paths: print(' -', p)

    # Build model architecture once (class count from dataset)
    model = QuantumHybridModel(len(dataset.class_names), use_quantum=(not args.no_quantum)).to(device)
    temperature = load_temperature()
    print(f'Using temperature scale: {temperature:.3f}')

    all_logits_accum = []

    # TTA transforms (operate on batch tensor directly)
    tta_transforms = []
    if args.tta:
        for _ in range(args.tta_count):
            tta_transforms.append(lambda x: x.flip(-1) if np.random.rand() < 0.5 else x)
            tta_transforms.append(lambda x: x)
    if len(tta_transforms)==0:
        tta_transforms = [lambda x: x]

    with torch.no_grad():
        for ckpt in ckpt_paths:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state['model_state_dict'], strict=False)
            model.eval()
            logits_list = []
            for imgs, _ in loader:
                imgs = imgs.to(device)
                if args.mc_dropout:
                    model.train()
                    for m_ in model.modules():
                        if isinstance(m_, nn.BatchNorm1d) or isinstance(m_, nn.BatchNorm2d):
                            m_.eval()
                batch_logits = []
                # MC dropout loops
                mc_iters = args.mc_samples if args.mc_dropout else 1
                for _ in range(mc_iters):
                    if args.mc_dropout:
                        # Dropout active
                        pass
                    if args.tta:
                        tta_logits = tta_batch(model, imgs, tta_transforms)  # T x B x C
                        batch_logits.append(tta_logits.mean(0))
                    else:
                        batch_logits.append(model(imgs))
                stacked = torch.stack(batch_logits, dim=0)  # M x B x C
                logits_list.append(stacked.mean(0))
            ckpt_logits = torch.cat(logits_list, dim=0) / temperature
            all_logits_accum.append(ckpt_logits)
    # Ensemble average
    ensemble_logits = torch.stack(all_logits_accum, dim=0).mean(0)
    probs = torch.softmax(ensemble_logits, dim=1).cpu().numpy()
    labels = np.array(dataset.labels)
    preds = probs.argmax(1)

    acc = (preds == labels).mean()
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    # Per-class metrics
    per_class = []
    for i, cname in enumerate(dataset.class_names):
        mask = labels == i
        tp = np.sum((preds==i) & mask)
        fp = np.sum((preds==i) & (~mask))
        fn = np.sum((preds!=i) & mask)
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1c = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        per_class.append({'class': cname, 'precision': precision, 'recall': recall, 'f1': f1c})

    cm = confusion_matrix(labels, preds)
    ece = ece_score(probs, labels)

    print(f"Ensemble Results | Acc {acc:.4f} Prec {prec:.4f} Rec {rec:.4f} F1 {f1:.4f} ECE {ece:.4f}")
    for pc in per_class:
        print(f"  Class {pc['class'][:22]:22s} | P {pc['precision']:.3f} R {pc['recall']:.3f} F1 {pc['f1']:.3f}")

    # Save confusion matrix figure
    try:
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.class_names, yticklabels=dataset.class_names)
        plt.title('Ensemble Confusion Matrix')
        plt.tight_layout()
        plt.savefig('ensemble_confusion_matrix.png')
        print('Saved ensemble_confusion_matrix.png')
    except Exception as e:
        print(f'Failed to save confusion matrix plot: {e}')

    out = {
        'accuracy': acc,
        'precision_weighted': prec,
        'recall_weighted': rec,
        'f1_weighted': f1,
        'ece': ece,
        'per_class': per_class,
        'class_names': dataset.class_names,
        'checkpoints': ckpt_paths,
        'temperature': temperature
    }
    with open(args.save_json,'w') as f:
        json.dump(out,f,indent=2)
    print(f'Results saved to {args.save_json}')

if __name__ == '__main__':
    main()
