import os
import json
import csv
from typing import List, Dict

import torch
import torch.nn as nn
try:
    from captum.attr import LayerGradCam
    _CAPTUM_AVAILABLE = True
except Exception:
    _CAPTUM_AVAILABLE = False
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pennylane as qml
from sklearn.metrics import classification_report, confusion_matrix

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid')
TEST_DIR = os.path.join(DATA_DIR, 'test')
CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_quantum_lung_cancer_model.pth')

# Transforms (align with train.py)
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset (copied from train.py with __len__/__getitem__)
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

        for d in os.listdir(base_path):
            class_dir = os.path.join(base_path, d)
            if not os.path.isdir(class_dir):
                continue
            mapped = aliases.get(d)
            if mapped is None or mapped not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[mapped]
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {base_path}!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, img_path

# Model definitions (minimal copy to load checkpoint)
num_qubits = 6
num_layers = 6

try:
    pl_dev = qml.device("lightning.gpu", wires=num_qubits)
except Exception:
    pl_dev = qml.device("default.qubit", wires=num_qubits)

def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(num_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

weight_shapes = {"weights": (num_layers, num_qubits, 3)}
qnode = qml.QNode(quantum_circuit, pl_dev, interface="torch")

class EnhancedQuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        scaled_input = x * self.scale
        return self.q_layer(scaled_input)

class QuantumHybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        try:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            self.backbone = models.resnet50(weights=None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dim_reduction = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
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

    def forward(self, x):
        features = self.backbone(x)
        quantum_input = self.dim_reduction(features)
        quantum_output = self.quantum_layer(quantum_input)
        logits = self.classifier(quantum_output)
        return logits


def count_per_class(base_dir: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not os.path.isdir(base_dir):
        return counts
    for d in os.listdir(base_dir):
        full = os.path.join(base_dir, d)
        if not os.path.isdir(full):
            continue
        c = 0
        for f in os.listdir(full):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                c += 1
        counts[d] = c
    return counts


def save_matrix_csv(path: str, mat, headers: List[str]):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + headers)
        for i, row in enumerate(mat):
            writer.writerow([headers[i]] + list(row))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--gradcam', action='store_true', help='Generate Grad-CAM heatmaps (requires captum)')
    parser.add_argument('--gradcam-samples', type=int, default=8, help='Number of samples for Grad-CAM')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Torch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', device)

    # Build datasets using train mapping
    train_ds = LungCancerDataset(TRAIN_DIR, transform=transform_val)
    fixed_class_to_idx = train_ds.class_to_idx
    class_names = train_ds.class_names

    test_ds = LungCancerDataset(TEST_DIR, transform=transform_val, class_to_idx=fixed_class_to_idx, class_names=class_names)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size if device.type == 'cuda' else min(args.batch_size,16), shuffle=False, num_workers=0)

    # Build and load model
    model = QuantumHybridModel(len(class_names)).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Inference
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    paths: List[str] = []
    with torch.no_grad():
        for x, y, p in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            prob = torch.softmax(logits, dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(y.tolist())
            y_prob.extend(prob)
            paths.extend(list(p))

    # Metrics
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True)

    # Print summary
    print('\nPer-class metrics:')
    for cls in class_names:
        r = report.get(cls, {})
        print(f"  {cls}: precision={r.get('precision', 0):.3f}, recall={r.get('recall', 0):.3f}, f1={r.get('f1-score', 0):.3f}")

    print('\nConfusion matrix:')
    print(cm)

    # Save artifacts
    out_dir = os.path.dirname(os.path.dirname(__file__))
    save_matrix_csv(os.path.join(out_dir, 'confusion_matrix.csv'), cm, class_names)

    with open(os.path.join(out_dir, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Class distributions
    train_counts = count_per_class(TRAIN_DIR)
    valid_counts = count_per_class(VALID_DIR)
    test_counts = count_per_class(TEST_DIR)
    with open(os.path.join(out_dir, 'class_distribution.json'), 'w') as f:
        json.dump({'train': train_counts, 'valid': valid_counts, 'test': test_counts}, f, indent=2)

    # Save misclassified images
    mis_dir = os.path.join(out_dir, 'misclassified_samples')
    os.makedirs(mis_dir, exist_ok=True)
    from shutil import copy2
    for t, pcls, path in zip(y_true, y_pred, paths):
        if t != pcls:
            tname = class_names[t]
            pname = class_names[pcls]
            base = os.path.basename(path)
            dst_name = f"pred_{pname}__true_{tname}__{base}"
            copy2(path, os.path.join(mis_dir, dst_name))

    # PR and ROC curves per class
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        y_prob_np = np.array(y_prob)

        # PR Curves
        plt.figure(figsize=(10, 8))
        for i, cname in enumerate(class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob_np[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_prob_np[:, i])
            plt.plot(recall, precision, label=f"{cname} (AP={ap:.3f})")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Per-class Precision-Recall Curves')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'pr_curves.png'))
        plt.close()

        # ROC Curves
        plt.figure(figsize=(10, 8))
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_np[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Per-class ROC Curves')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'roc_curves.png'))
        plt.close()
    except Exception as e:
        print(f"Could not generate PR/ROC curves: {e}")

    # Grad-CAM generation (backbone last conv) if requested
    if args.gradcam and _CAPTUM_AVAILABLE:
        try:
            target_layer = None
            for name, module in model.backbone.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module  # last conv encountered
            if target_layer is not None:
                gc = LayerGradCam(model, target_layer)
                import numpy as np
                sel = np.random.choice(len(paths), size=min(args.gradcam_samples, len(paths)), replace=False)
                heatmaps = []
                cam_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'gradcam')
                os.makedirs(cam_dir, exist_ok=True)
                preprocess = transforms.Compose([
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
                for idx in sel:
                    img = Image.open(paths[idx]).convert('RGB')
                    tens = preprocess(img).unsqueeze(0).to(device)
                    pred_class = y_pred[idx]
                    attr = gc.attribute(tens, target=pred_class)
                    cam = attr.squeeze().cpu().numpy()
                    # Normalize heatmap 0-1
                    cam_min, cam_max = cam.min(), cam.max()
                    if cam_max - cam_min > 0:
                        cam = (cam - cam_min) / (cam_max - cam_min)
                    heatmaps.append({'image_path': paths[idx], 'pred_class': class_names[pred_class], 'heatmap': cam.tolist()})
                with open(os.path.join(cam_dir, 'gradcam.json'), 'w') as f:
                    json.dump({'samples': heatmaps}, f, indent=2)
                print(f"Grad-CAM heatmaps saved to {cam_dir}")
            else:
                print('Grad-CAM: No convolutional layer found.')
        except Exception as e:
            print(f'Grad-CAM generation failed: {e}')
    elif args.gradcam and not _CAPTUM_AVAILABLE:
        print('Grad-CAM requested but captum not available. Install captum to enable.')

    print('\nSaved: confusion_matrix.csv, classification_report.json, class_distribution.json, misclassified_samples/, pr_curves.png, roc_curves.png')
    if args.gradcam and _CAPTUM_AVAILABLE:
        print('Grad-CAM: gradcam/gradcam.json')
    print('Classes:', class_names)


if __name__ == '__main__':
    main()
