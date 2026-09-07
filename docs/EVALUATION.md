# 📊 Evaluation Methodology — SpaceNetra

> How we measure, validate, and report system performance

---

## Table of Contents

- [Evaluation Strategy](#evaluation-strategy)
- [Model Evaluation](#model-evaluation)
- [System Evaluation](#system-evaluation)
- [Benchmark Results Template](#benchmark-results-template)
- [Visual Evaluation Protocol](#visual-evaluation-protocol)
- [Confidence Engine Evaluation](#confidence-engine-evaluation)
- [Search Quality Evaluation](#search-quality-evaluation)
- [End-to-End Evaluation](#end-to-end-evaluation)

---

## Evaluation Strategy

### Three Levels of Evaluation

```
┌─────────────────────────────────────────────────────────────┐
│                 EVALUATION PYRAMID                           │
│                                                             │
│                    ┌───────────┐                             │
│                    │  END-TO-  │  Full pipeline test         │
│                    │   END     │  User scenarios             │
│                    └─────┬─────┘                             │
│                          │                                   │
│               ┌──────────▼──────────┐                        │
│               │  SYSTEM EVALUATION  │  Latency, throughput   │
│               │                     │  Storage, scalability  │
│               └──────────┬──────────┘                        │
│                          │                                   │
│          ┌───────────────▼───────────────┐                   │
│          │     MODEL EVALUATION          │  F1, IoU          │
│          │                               │  Precision, Recall│
│          │     (Per-model metrics)        │  Inference time   │
│          └───────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Evaluation

### Change Detection Metrics

#### Primary Metrics

| Metric | Formula | Range | Target |
|--------|---------|-------|--------|
| **Precision** | TP / (TP + FP) | [0, 1] | > 0.85 |
| **Recall** | TP / (TP + FN) | [0, 1] | > 0.83 |
| **F1 Score** | 2·P·R / (P + R) | [0, 1] | > 0.85 |
| **IoU (Jaccard)** | TP / (TP + FP + FN) | [0, 1] | > 0.75 |
| **Overall Accuracy** | (TP + TN) / Total | [0, 1] | > 0.95 |
| **Kappa** | (OA - PE) / (1 - PE) | [-1, 1] | > 0.80 |

#### Confusion Matrix

```
                    PREDICTED
                 Change    No Change
              ┌──────────┬──────────┐
  ACTUAL      │          │          │
  Change      │  TP      │  FN      │
              │          │          │
              ├──────────┼──────────┤
  No Change   │          │          │
              │  FP      │  TN      │
              │          │          │
              └──────────┴──────────┘

Precision = TP / (TP + FP)       "Of what I detected, how much is real?"
Recall    = TP / (TP + FN)       "Of what exists, how much did I find?"
F1        = harmonic mean         "Balance of both"
IoU       = TP / (TP + FP + FN)  "How well do regions overlap?"
```

#### Per-Class Analysis

```
                CHANGE CLASS                    NO-CHANGE CLASS
         ┌─────────────────────┐         ┌─────────────────────┐
         │ Precision: XX%      │         │ Precision: XX%      │
         │ Recall:    XX%      │         │ Recall:    XX%      │
         │ F1:        XX%      │         │ F1:        XX%      │
         │ Support:   XXXX px  │         │ Support:   XXXXX px │
         └─────────────────────┘         └─────────────────────┘
```

### Evaluation Protocol

```
1. Train on: LEVIR-CD train set (445 pairs × 16 patches = 7,120 patches)
2. Validate on: LEVIR-CD val set (64 pairs × 16 patches = 1,024 patches)
3. Test on: LEVIR-CD test set (128 pairs × 16 patches = 2,048 patches)

CRITICAL RULE: Test set is NEVER used during model development.
               Only evaluate ONCE as the final step.
```

### Inference Speed

| Measurement | Method |
|-------------|--------|
| **Per-tile latency** | Average over 1000 tiles (excluding first 10 warmup) |
| **Throughput** | Tiles/second at batch size 8 |
| **GPU memory** | Peak VRAM during inference |
| **CPU fallback** | Latency without GPU |

---

## System Evaluation

### Performance Benchmarks

| Metric | How Measured | Target |
|--------|-------------|--------|
| **Search latency** | Time from query to results (p50, p95, p99) | < 100 ms (p95) |
| **CD inference** | Time per tile pair (model only) | < 500 ms |
| **Timeline query** | Time to retrieve temporal series | < 200 ms |
| **Page load** | Dashboard initial render | < 2 seconds |
| **Scene ingestion** | Full pipeline for new scene | < 10 minutes |
| **Index rebuild** | Full FAISS/Qdrant rebuild | Measured |

### Scalability Benchmarks

| Metric | How Measured |
|--------|-------------|
| **Index size vs. tiles** | Storage growth per 1000 tiles |
| **Search latency vs. index size** | Query time at 10K, 100K, 1M tiles |
| **Ingestion throughput** | Tiles/minute during bulk ingest |
| **Concurrent users** | Max users before degradation |

### Resource Utilization

```
┌──────────────────────────────────────────────┐
│         RESOURCE MONITORING                   │
│                                              │
│  CPU Usage:     ████████░░  80%              │
│  RAM Usage:     ██████░░░░  60%              │
│  GPU Usage:     ███████░░░  70%              │
│  GPU Memory:    █████░░░░░  50%              │
│  Disk I/O:      ████░░░░░░  40%              │
│  Storage Used:  ████████░░  82 GB / 100 GB   │
│                                              │
│  Active Services: 5/5                        │
│  Queue Depth:     12 tiles                   │
└──────────────────────────────────────────────┘
```

---

## Benchmark Results Template

### Model Evaluation Report

```
═══════════════════════════════════════════════
         MODEL EVALUATION REPORT
         SpaceNetra Change Detection
═══════════════════════════════════════════════

Date:           YYYY-MM-DD
Model:          Siamese U-Net v1.0.0
Backbone:       ResNet-18 (ImageNet pretrained)
Checkpoint:     best_model_epoch_XX.pth
Hash:           sha256:XXXXXXXXXXXX

───────────────────────────────────────────────
DATASET
───────────────────────────────────────────────
Name:           LEVIR-CD
Split:          Test (128 pairs)
Patch Size:     256 × 256
Total Patches:  2,048
Change Ratio:   ~4.7% of pixels

───────────────────────────────────────────────
METRICS
───────────────────────────────────────────────
                Value       Target      Status
Precision       XX.XX%      > 85%       ✅/❌
Recall          XX.XX%      > 83%       ✅/❌
F1 Score        XX.XX%      > 85%       ✅/❌
IoU             XX.XX%      > 75%       ✅/❌
Overall Acc     XX.XX%      > 95%       ✅/❌
Kappa           X.XXX       > 0.80      ✅/❌

───────────────────────────────────────────────
INFERENCE PERFORMANCE
───────────────────────────────────────────────
Hardware:       NVIDIA RTX XXXX (XX GB)
Batch Size:     8
Latency (p50):  XX ms/tile
Latency (p95):  XX ms/tile
Throughput:     XX tiles/sec
GPU Memory:     XX MB peak
CPU Fallback:   XX ms/tile

───────────────────────────────────────────────
TRAINING DETAILS
───────────────────────────────────────────────
Epochs:         XX (early stopped at XX)
Best Val F1:    XX.XX% at epoch XX
Training Time:  XX hours
Optimizer:      AdamW (lr=1e-3, wd=0.01)
Loss:           BCE(0.5) + Dice(0.5)
Augmentations:  HFlip, VFlip, Rot90, Jitter

═══════════════════════════════════════════════
```

### System Evaluation Report

```
═══════════════════════════════════════════════
         SYSTEM EVALUATION REPORT
         SpaceNetra v1.0.0
═══════════════════════════════════════════════

Date:           YYYY-MM-DD
Deployment:     Docker Compose (5 services)
Hardware:       [Server specifications]

───────────────────────────────────────────────
DATA SCALE
───────────────────────────────────────────────
Indexed Area:           XXX km²
Satellite Scenes:       XXXX
Total Tiles:            XXXXXX
Date Range:             YYYY — YYYY
Sensors:                Sentinel-2

───────────────────────────────────────────────
STORAGE
───────────────────────────────────────────────
Tile Storage:           XX.X GB
Vector Index:           XX.X MB
PostgreSQL DB:          XX.X MB
Total Footprint:        XX.X GB

───────────────────────────────────────────────
PERFORMANCE
───────────────────────────────────────────────
                        p50     p95     p99
Search Latency:         XX ms   XX ms   XX ms
CD Inference:           XX ms   XX ms   XX ms
Timeline Query:         XX ms   XX ms   XX ms
Event Retrieval:        XX ms   XX ms   XX ms

Scene Ingestion:        XX minutes
Index Build Time:       XX minutes

───────────────────────────────────────────────
RELIABILITY
───────────────────────────────────────────────
Uptime:                 XX.XX%
Failed Queries:         XX / XXXX (X.XX%)
Model Load Time:        XX seconds
Cold Start Time:        XX seconds

═══════════════════════════════════════════════
```

---

## Visual Evaluation Protocol

### Sample Grid Generation

For every test sample, generate a 4-panel comparison:

```
┌────────────────┬────────────────┬────────────────┬────────────────┐
│                │                │                │                │
│   T1 (Before)  │  T2 (After)   │ Ground Truth   │  Prediction    │
│                │                │                │                │
│                │                │  ██            │  ██            │
│                │   ▓▓▓▓        │  ████          │  ███           │
│                │   ▓▓▓▓        │                │                │
│                │                │                │                │
└────────────────┴────────────────┴────────────────┴────────────────┘
     (a)              (b)             (c)              (d)

Caption: Sample #XXX — F1: X.XX, IoU: X.XX
```

### Error Analysis Grid

```
┌────────────────┬────────────────┬────────────────┐
│                │                │                │
│  T2 (After)    │  Overlay       │  Error Map     │
│                │ (pred on T2)   │                │
│                │                │  ■ = TP (green)│
│                │                │  ■ = FP (red)  │
│                │                │  ■ = FN (blue) │
│                │                │                │
└────────────────┴────────────────┴────────────────┘
```

### Failure Case Analysis

Identify and categorize failure modes:

| Failure Mode | Example | Mitigation |
|-------------|---------|------------|
| **Missed small changes** | Single building | Lower threshold or multi-scale |
| **False alarm on shadows** | Cloud shadow → "change" | Quality/confidence filtering |
| **Edge bleeding** | Change mask bleeds beyond building | Post-processing refinement |
| **Misregistration artifacts** | Shift → false change strip | Registration quality check |
| **Seasonal confusion** | Green→brown → "change" | Temporal persistence check |

---

## Confidence Engine Evaluation

### Calibration Analysis

Does confidence 0.90 mean the prediction is correct 90% of the time?

```
Confidence Range    Actual Accuracy    Calibration Error
─────────────────   ───────────────    ─────────────────
0.0 — 0.2           XX%                XX%
0.2 — 0.4           XX%                XX%
0.4 — 0.6           XX%                XX%
0.6 — 0.8           XX%                XX%
0.8 — 1.0           XX%                XX%
```

### False Alarm Suppression Rate

```
                    Without Confidence     With Confidence
                    Engine                 Engine
──────────────────  ──────────────────     ──────────────
True Positives:     XXXX                   XXXX
False Positives:    XXXX                   XXX  (↓ XX%)
False Negatives:    XXX                    XXXX (↑ slight)
Precision:          XX%                    XX%  (↑ XX%)
Recall:             XX%                    XX%  (↓ slight)
F1:                 XX%                    XX%  (↑ net)
```

---

## Search Quality Evaluation

### Retrieval Metrics

| Metric | Definition |
|--------|-----------|
| **Recall@K** | Fraction of relevant tiles in top-K results |
| **Precision@K** | Fraction of top-K results that are relevant |
| **MRR** | Mean Reciprocal Rank of first relevant result |
| **NDCG@K** | Normalized Discounted Cumulative Gain |

### Test Queries

| Query | Expected Result Type | Evaluation |
|-------|---------------------|-----------|
| "construction activity" | Building sites | Recall@10, Precision@10 |
| "deforestation" | Forest clearing | Recall@10, Precision@10 |
| "urban expansion" | New development | Recall@10, Precision@10 |
| "water body change" | Reservoir/flood | Recall@10, Precision@10 |
| "agricultural land" | Farmland | Recall@10, Precision@10 |

---

## End-to-End Evaluation

### Scenario-Based Testing

| Scenario | Steps | Success Criteria |
|----------|-------|-----------------|
| **New scene** | Ingest → Preprocess → Tile → Embed → Index | All tiles indexed, < 10 min |
| **Search** | Query → Results → View tile → View metadata | Relevant results in < 100 ms |
| **Change detect** | Select pair → Run CD → View mask → Confidence | Accurate mask, < 500 ms |
| **Timeline** | Select location → View temporal series → Fingerprint | Complete timeline, < 200 ms |
| **Feedback** | View event → Accept/Reject → Verify stored | Decision recorded |
| **Provenance** | Select event → View full chain | All sources traceable |

### Demo Readiness Checklist

- [ ] 5+ successful search queries with relevant results
- [ ] 3+ change detection examples with visual comparison
- [ ] 1+ complete timeline showing multi-year evolution
- [ ] Confidence scoring demonstration (high vs low confidence)
- [ ] Provenance chain demonstration
- [ ] Analyst feedback workflow demonstration
- [ ] System health dashboard
- [ ] All latency targets met
