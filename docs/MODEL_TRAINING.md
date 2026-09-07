# 🧠 Model Training Guide — SpaceNetra

> Step-by-step guide to training, validating, and evaluating change detection models

---

## Table of Contents

- [Training Overview](#training-overview)
- [Model Architectures](#model-architectures)
- [Loss Functions](#loss-functions)
- [Training Configuration](#training-configuration)
- [Training Loop](#training-loop)
- [Validation Strategy](#validation-strategy)
- [Overfitting Prevention](#overfitting-prevention)
- [Evaluation Metrics](#evaluation-metrics)
- [Visual Evaluation](#visual-evaluation)
- [Model Improvement Path](#model-improvement-path)
- [Fine-Tuning on Indian Data](#fine-tuning-on-indian-data)

---

## Training Overview

```
┌────────────────────────────────────────────────────────┐
│                 TRAINING PIPELINE                       │
│                                                        │
│   LEVIR-CD Dataset                                     │
│        │                                               │
│        ▼                                               │
│   ┌──────────┐                                         │
│   │ DataLoader│  Patches (256×256), Augmented          │
│   └────┬─────┘                                         │
│        │                                               │
│        ▼                                               │
│   ┌──────────┐     ┌──────────┐                        │
│   │  Model   │────▶│Prediction│                        │
│   │(Siamese) │     │  Mask    │                        │
│   └──────────┘     └────┬─────┘                        │
│                         │                              │
│                    ┌────▼─────┐                         │
│   Ground Truth ───▶│  Loss    │                        │
│                    │ BCE+Dice │                        │
│                    └────┬─────┘                         │
│                         │                              │
│                    ┌────▼─────┐                         │
│                    │Backprop  │                        │
│                    │+ Update  │                        │
│                    └──────────┘                         │
│                                                        │
│   Repeat for N epochs                                  │
│   Save best checkpoint based on validation F1          │
└────────────────────────────────────────────────────────┘
```

---

## Model Architectures

### Architecture 1: Siamese U-Net (Baseline)

**Concept**: Two images pass through the same encoder. The difference in features is decoded into a change mask.

```
Input: T1 (256×256×3), T2 (256×256×3)

ENCODER (shared weights — ResNet-18/34 backbone):
  Block 1: 3 → 64 channels,   128×128
  Block 2: 64 → 128 channels,  64×64
  Block 3: 128 → 256 channels, 32×32
  Block 4: 256 → 512 channels, 16×16

FEATURE COMPARISON:
  diff = |encoder(T1) - encoder(T2)|    # or concatenation

DECODER (U-Net style with skip connections):
  UpBlock 4: 512 → 256, 32×32  + skip
  UpBlock 3: 256 → 128, 64×64  + skip
  UpBlock 2: 128 → 64,  128×128 + skip
  UpBlock 1: 64 → 32,   256×256 + skip

OUTPUT:
  Conv 1×1: 32 → 1 channel
  Sigmoid: → probability map (0.0 - 1.0)
  Threshold at 0.5 → binary mask
```

**Parameters**: ~11M (with ResNet-18 backbone)

### Architecture 2: ChangeFormer (Advanced)

**Concept**: Hierarchical transformer encoder replaces CNN for stronger global context.

```
Input: T1 (256×256×3), T2 (256×256×3)

ENCODER (Hierarchical Transformer — MiT backbone):
  Stage 1: Patch embed (4×4) → H/4 × W/4, C1 channels
  Stage 2: Downsample      → H/8 × W/8, C2 channels
  Stage 3: Downsample      → H/16 × W/16, C3 channels
  Stage 4: Downsample      → H/32 × W/32, C4 channels

  Each stage: Multi-head Self-Attention + FFN

DIFFERENCE MODULE:
  Multi-scale feature differencing at each stage

DECODER (MLP-based):
  Upsample + Fuse features from all 4 stages
  MLP layers → final prediction

OUTPUT:
  Binary change mask (256×256)
```

**Parameters**: ~41M (MiT-b2 backbone)

### Architecture Comparison

| Property | Siamese U-Net | ChangeFormer |
|----------|--------------|--------------|
| Backbone | ResNet-18/34 | MiT-b2 |
| Parameters | ~11M | ~41M |
| Context | Local (CNN) | Global (Transformer) |
| Training speed | Fast | Slower |
| GPU memory | ~4 GB | ~8-12 GB |
| Expected F1 | ~85-88% | ~90-92% |
| Best for | Baseline, resource-limited | Best accuracy |

---

## Loss Functions

### Problem: Severe Class Imbalance

In LEVIR-CD, typically **< 5%** of pixels are "change" class.

```
┌────────────────────────────────────────────┐
│          TYPICAL LEVIR-CD MASK             │
│                                            │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ░░░░░░░░░░██░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                            │
│  ░ = No change (95%+)   █ = Change (<5%)   │
└────────────────────────────────────────────┘
```

### Solution: Combined Loss

```
Total Loss = α × Weighted BCE + (1 - α) × Dice Loss
```

**Default α = 0.5**

### Weighted Binary Cross-Entropy (BCE)

```
BCE = -[w_pos × y × log(ŷ) + w_neg × (1-y) × log(1-ŷ)]

where:
  w_pos = N_total / (2 × N_change)    ← higher weight for change class
  w_neg = N_total / (2 × N_nochange)
```

### Dice Loss

```
Dice = 1 - (2 × |P ∩ G| + ε) / (|P| + |G| + ε)

where:
  P = predicted change pixels
  G = ground truth change pixels
  ε = smoothing factor (1e-6)
```

**Why Dice?** Directly optimizes spatial overlap, insensitive to class imbalance.

---

## Training Configuration

### Recommended Hyperparameters

```yaml
# configs/baseline.yaml

# Model
model:
  architecture: "siamese_unet"
  backbone: "resnet18"
  pretrained: true          # ImageNet pre-trained encoder
  in_channels: 3
  out_channels: 1

# Data
data:
  dataset: "levir_cd"
  patch_size: 256
  batch_size: 8             # Adjust based on GPU memory
  num_workers: 4
  
# Training
training:
  epochs: 100
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.01
  
# Loss
loss:
  type: "combined"
  bce_weight: 0.5
  dice_weight: 0.5
  pos_weight: auto          # Computed from dataset statistics

# Scheduler
scheduler:
  type: "cosine_annealing"
  T_max: 100
  eta_min: 0.00001

# Early Stopping
early_stopping:
  patience: 15
  monitor: "val_f1"
  mode: "max"

# Checkpointing
checkpoint:
  save_best: true
  monitor: "val_f1"
  save_last: true
```

---

## Training Loop

### Pseudocode

```python
best_val_f1 = 0
patience_counter = 0

for epoch in range(num_epochs):
    # ──── TRAINING ────
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        t1, t2, mask = batch
        t1, t2, mask = t1.cuda(), t2.cuda(), mask.cuda()
        
        # Forward pass
        prediction = model(t1, t2)
        loss = criterion(prediction, mask)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # ──── VALIDATION ────
    model.eval()
    val_metrics = evaluate(model, val_loader)
    
    # ──── LOGGING ────
    log(epoch, train_loss, val_metrics)
    
    # ──── CHECKPOINT ────
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        save_checkpoint(model, 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    # ──── EARLY STOPPING ────
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    # ──── LR SCHEDULE ────
    scheduler.step()
```

### Expected Training Progression

```
Epoch   Train Loss   Val Loss   Val F1    Val IoU   LR
─────   ──────────   ────────   ──────    ───────   ──────
  1       0.712       0.685     0.312     0.185    1.0e-3
  5       0.458       0.441     0.621     0.450    9.7e-4
 10       0.287       0.301     0.756     0.607    8.8e-4
 20       0.193       0.215     0.834     0.715    6.5e-4
 30       0.147       0.189     0.867     0.765    4.0e-4
 50       0.098       0.172     0.885     0.794    1.5e-4
 70       0.071       0.168     0.891     0.804    3.7e-5
100       0.055       0.170     0.889     0.801    1.0e-5 *
                                                    ↑
                                            * may early stop
```

---

## Validation Strategy

### Validation During Training

Every epoch, evaluate on the validation set:

```
┌─────────────────────────────────────────────┐
│           VALIDATION PIPELINE                │
│                                             │
│  Val T1 + Val T2                            │
│       │                                     │
│       ▼                                     │
│  Model (no gradient)                        │
│       │                                     │
│       ▼                                     │
│  Predictions                                │
│       │                                     │
│       ▼                                     │
│  Compare with Val Ground Truth              │
│       │                                     │
│       ▼                                     │
│  Metrics: F1, IoU, Precision, Recall        │
│       │                                     │
│       ▼                                     │
│  Decision:                                  │
│    Best F1? → Save checkpoint               │
│    No improvement for N epochs? → Stop      │
└─────────────────────────────────────────────┘
```

### K-Fold Cross-Validation (Optional)

For more robust estimates, use 5-fold cross-validation on train+val, then final test on held-out test set.

---

## Overfitting Prevention

### Warning Signs

```
                    OVERFITTING
                    ↓
Training F1:  ████████████████████  98%
Validation F1: ████████████         70%
                              ↑
                         GAP > 15%
```

### Countermeasures

| Technique | Implementation | Effect |
|-----------|---------------|--------|
| **Data Augmentation** | Flips, rotations, jitter | Increases effective dataset size |
| **Early Stopping** | Stop if val F1 doesn't improve for 15 epochs | Prevents overtraining |
| **Learning Rate Schedule** | Cosine annealing from 1e-3 to 1e-5 | Gentler optimization |
| **Weight Decay** | L2 regularization (0.01) | Penalizes large weights |
| **Dropout** | In decoder (p=0.3) | Reduces feature co-adaptation |
| **Best Checkpoint** | Save model with best val F1, not lowest train loss | Uses best generalization |
| **Batch Normalization** | In all conv blocks | Stabilizes training |

### Monitoring Dashboard

Track these curves during training:

```
     Loss                          F1 Score
1.0 ┤                        1.0 ┤
    │╲                           │              ___─────
    │ ╲    train                 │           __╱    val
0.5 ┤  ╲___───────              0.5 ┤       _╱╱  
    │       ╲___──── val             │     _╱╱ train
    │                                │   ╱╱
0.0 ┤                        0.0 ┤──╱
    └───────────────               └───────────────
    0    50   100 epoch            0    50   100 epoch

    ✅ Healthy: train and val converge
    ❌ Overfitting: train improves, val plateaus/worsens
```

---

## Evaluation Metrics

### Primary Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Precision** | TP / (TP + FP) | Of detected changes, how many are real? |
| **Recall** | TP / (TP + FN) | Of actual changes, how many did we find? |
| **F1 Score** | 2 × (P × R) / (P + R) | Harmonic mean of P and R |
| **IoU (Jaccard)** | TP / (TP + FP + FN) | Overlap between prediction and truth |

Where:
- **TP** = True Positive (correctly detected change)
- **FP** = False Positive (detected change that isn't real)
- **FN** = False Negative (missed actual change)
- **TN** = True Negative (correctly identified no-change)

### Confusion Matrix

```
                    PREDICTED
                 Change    No Change
              ┌──────────┬──────────┐
  ACTUAL      │          │          │
  Change      │    TP    │    FN    │
              │          │          │
              ├──────────┼──────────┤
  No Change   │          │          │
              │    FP    │    TN    │
              │          │          │
              └──────────┴──────────┘
```

### Target Metrics (LEVIR-CD)

| Metric | Baseline Target | Advanced Target |
|--------|----------------|-----------------|
| F1 | > 0.85 | > 0.90 |
| IoU | > 0.75 | > 0.82 |
| Precision | > 0.85 | > 0.90 |
| Recall | > 0.83 | > 0.88 |

---

## Visual Evaluation

### Comparison Grid

For each test sample, generate:

```
┌────────────┬────────────┬────────────┬────────────┐
│            │            │            │            │
│   T1       │    T2      │ Ground     │ Prediction │
│ (Before)   │  (After)   │ Truth      │  (Model)   │
│            │            │            │            │
│            │            │ ■■         │ ■■         │
│            │            │ ■■■■       │ ■■■        │
│            │            │            │            │
└────────────┴────────────┴────────────┴────────────┘
                                ↑              ↑
                           Actual change   AI prediction
                           (from dataset)  (from model)
```

### Error Analysis Visualization

```
┌────────────┬────────────┬────────────┐
│            │            │            │
│   T1       │ Overlay    │  Error Map │
│            │ (pred on   │            │
│            │  T2)       │ ■ TP       │
│            │            │ ■ FP       │
│            │            │ ■ FN       │
│            │            │            │
└────────────┴────────────┴────────────┘

Color code:
  GREEN = True Positive  (correctly detected)
  RED   = False Positive (false alarm)
  BLUE  = False Negative (missed change)
```

---

## Model Improvement Path

### Progressive Architecture Upgrade

```
Step 1: Siamese U-Net (ResNet-18)
           │
           │  Evaluate F1, IoU
           │  If F1 < 0.85 → investigate data issues first
           │
           ▼
Step 2: Siamese U-Net (ResNet-34/50)
           │
           │  Deeper encoder, more capacity
           │
           ▼
Step 3: ChangeFormer (MiT-b1)
           │
           │  Transformer-based, global context
           │
           ▼
Step 4: ChangeFormer (MiT-b2)
           │
           │  Larger transformer, best accuracy
           │
           ▼
Step 5: Ensemble (optional)
           │
           │  Combine predictions from multiple models
           │
           ▼
       FINAL MODEL
```

### Principle

> **Use the simplest model that gives reliable results for our data and hardware.**
> 
> Don't use ChangeFormer if Siamese U-Net already achieves F1 > 0.90.
> Complexity costs: training time, inference time, deployment size, debugging difficulty.

---

## Fine-Tuning on Indian Data

### Why Fine-Tuning is Critical

| Aspect | LEVIR-CD | Indian Sentinel-2 |
|--------|---------|-------------------|
| Resolution | ~0.5 m | 10 m |
| Bands | RGB | 4-13 bands |
| Geography | Texas, USA | India |
| Building types | American suburban | Indian urban/rural |
| Vegetation | Temperate | Tropical/arid/mixed |
| Seasons | North American | Monsoon pattern |

Training only on LEVIR-CD and claiming an Indian defence system would be a major technical weakness.

### Fine-Tuning Strategy

```
┌──────────────────────────────────────────────┐
│ STEP 1: Pre-train on LEVIR-CD                │
│         → Learn general change detection      │
│         → Establish baseline metrics          │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 2: Prepare Indian Dataset                │
│         → Sentinel-2 pairs over Indian AOIs   │
│         → Manual or semi-automated labels     │
│         → Preprocess with satellite pipeline   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 3: Fine-tune                             │
│         → Load LEVIR-CD checkpoint             │
│         → Freeze encoder (initially)           │
│         → Train decoder on Indian data         │
│         → Gradually unfreeze encoder           │
│         → Lower learning rate (1e-4 → 1e-5)   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 4: Evaluate on Indian test data          │
│         → Must demonstrate performance on      │
│           Indian geography to be credible      │
└──────────────────────────────────────────────┘
```

### Input Adaptation

If moving from 3-channel (RGB) to 4-channel (RGB+NIR):

```
Option A: Modify first conv layer from 3→64 to 4→64
          Re-initialize only the first layer weights
          
Option B: Use 3-channel (RGB only) from Sentinel-2
          Simpler, no architecture change needed
          
Option C: Use separate NIR processing branch
          More complex, potentially more powerful
```

**Recommendation**: Start with Option B (RGB only), then experiment with Option A if needed.
