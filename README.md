# Quantum-Inspired Lung Cancer Detection Model

This project implements a hybrid classical–quantum pipeline for lung cancer subtype classification using histopathology image data. It combines a ResNet50 feature extractor with a small quantum circuit (AngleEmbedding + StronglyEntanglingLayers) and modern training enhancements (AMP, focal loss, EMA, SWA, warmup, temperature scaling, ensemble inference, test-time augmentation, uncertainty estimation).

## Key Components
- Backbone: `ResNet50` pretrained weights (fallback to random if unavailable).
- Bottleneck: Linear -> BN -> ReLU -> Dropout -> Linear mapping to quantum inputs.
- Quantum Layer: 6 qubits, 6 entangling layers; TorchLayer interface via PennyLane.
- Classification Head: Dense -> ReLU -> Dropout -> Dense.
- Ablation Path: Optional pure-classical head without quantum (`--no-quantum`).

## Training Enhancements
| Feature | Flag | Description |
|---------|------|-------------|
| Mixed Precision | `--amp` | Faster & memory-efficient on CUDA |
| Focal Loss | `--use-focal --gamma` | Focus on hard/under-represented samples |
| Class Balancing | `--balance sampler|weights|none` | Weighted sampler or class weights |
| Quantum Warmup | `--warmup-epochs N` | Freeze quantum layer initially |
| EMA | `--ema --ema-decay` | Smooth parameter trajectory |
| SWA | `--swa --swa-start-epoch` | Averages weights for flatter minima |
| Gradient Clipping | `--clip-grad` | Stabilize training |
| Temperature Scaling | `--temp-scale` | Calibrate probability outputs |
| Targeted Augmentation | `--augment-adenocarcinoma` | Extra transforms for specific class |
| MC Dropout | `--mc-dropout` | Uncertainty via stochastic dropout |
| Scheduler | `--scheduler plateau|cosine` | ReduceLROnPlateau or CosineAnnealingWarmRestarts |
| Seed | `--seed` | Deterministic reproducibility |
| Bottleneck Dropout | `--bottleneck-dropout` | Regularization intensity |

## Ensemble & Evaluation
Use `ensemble_predict.py` to average multiple checkpoints with optional:
- Test-Time Augmentation (`--tta --tta-count`) 
- MC Dropout sampling (`--mc-dropout --mc-samples`) 
- Temperature scaling auto-loaded from `calibration.json` if present.

Outputs: `ensemble_results.json`, `ensemble_confusion_matrix.png` and per-class metrics.

## Example Workflow

### Phase 1: Stable Backbone + Warmup
```powershell
python .\quantum\train.py --epochs 40 --warmup-epochs 5 --scheduler cosine --amp --balance sampler --clip-grad 1.0 --seed 2025 --ema --ema-decay 0.999 --bottleneck-dropout 0.3
```

### Phase 2: Focused Fine-Tune with Focal + SWA
```powershell
python .\quantum\train.py --finetune --epochs 15 --use-focal --gamma 1.25 --amp --balance sampler --clip-grad 0.75 --scheduler plateau --patience 6 --seed 2025 --ema --swa --swa-start-epoch 8
```

### Calibration
```powershell
python .\quantum\train.py --finetune --epochs 1 --temp-scale --seed 2025
```

### Ensemble Inference
```powershell
python .\quantum\ensemble_predict.py --data-dir Data/test --checkpoints best_quantum_lung_cancer_model.pth --tta --tta-count 4 --mc-dropout --mc-samples 5
```

## Uncertainty & Reliability
- MC Dropout: Provides predictive variance; examine logit std across samples.
- ECE (Expected Calibration Error): Lower is better; stored in `ensemble_results.json`.

## Explainability (Planned)
`captum` is included for future integration (e.g., Grad-CAM on backbone feature maps). This will be added to `eval.py` for generating heatmaps to highlight salient regions.

## Data Assumptions
Expected folder structure:
```
Data/
  train/ CLASS_FOLDERS...
  valid/ CLASS_FOLDERS...
  test/  CLASS_FOLDERS...
```
Class folder naming may include extended descriptors; mapping uses leading token (e.g., `adenocarcinoma_*`).

## Medical Disclaimer
This model is a research prototype. It is NOT validated for clinical decision-making, diagnosis, or patient management. Outputs must be reviewed by qualified medical professionals. Do not deploy in real-world clinical settings without rigorous regulatory evaluation, multi-institutional validation, and ethical oversight.

## Tips for Optimization
- Monitor per-class recall; if one subtype lags, switch from sampler to class weights or enable targeted augmentation.
- Start with `gamma=1.0` for focal loss if instability; raise gradually.
- If cosine scheduler causes premature LR drops, increase `T_0` (e.g., 15) or switch to plateau.
- Use multiple different seeds + ensemble for robustness.
- Keep quantum qubit count modest to avoid training noise; current (6) is a balance.

## Reproducibility
Set `--seed` and disable non-deterministic cuDNN (handled internally when seed is set). Exact reproducibility may still vary with different GPU architectures or library versions.

## Future Extensions
- Grad-CAM / Integrated Gradients for interpretability.
- Multi-scale tiling if source images are large WSIs.
- Active learning loop to select uncertain samples (`MC variance`).
- Federated training across institutions.

## License
No explicit license provided; treat as internal research code.

---
For questions or enhancement requests, open an issue or extend the scripts directly.
