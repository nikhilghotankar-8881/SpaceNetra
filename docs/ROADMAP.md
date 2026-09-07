# 🗺️ Development Roadmap — SpaceNetra

> Complete 32-phase roadmap from zero to deployed satellite intelligence system

---

## Overview

This roadmap is organized into **5 milestones**, each building on the previous one. The principle is:

> **Make the AI work first. Then build the application around it.**

```
                 START
                   │
    ═══════════════╪═══════════════
    MILESTONE 1:   │  AI Foundation
    ═══════════════╪═══════════════
                   │
                   ▼
        ┌─────────────────────┐
        │ Phase 0-1            │
        │ Planning & Setup     │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ Phase 2-6            │
        │ Data Preparation     │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ Phase 7-13           │
        │ Train & Evaluate     │
        └──────────┬──────────┘
                   │
    ═══════════════╪═══════════════
    MILESTONE 2:   │  Real-World Data
    ═══════════════╪═══════════════
                   │
                   ▼
        ┌─────────────────────┐
        │ Phase 14-17          │
        │ Satellite Pipeline   │
        └──────────┬──────────┘
                   │
    ═══════════════╪═══════════════
    MILESTONE 3:   │  Semantic Search
    ═══════════════╪═══════════════
                   │
                   ▼
        ┌─────────────────────┐
        │ Phase 18-20          │
        │ Search & Temporal    │
        └──────────┬──────────┘
                   │
    ═══════════════╪═══════════════
    MILESTONE 4:   │  Intelligence
    ═══════════════╪═══════════════
                   │
                   ▼
        ┌─────────────────────┐
        │ Phase 21-23          │
        │ Confidence & Events  │
        └──────────┬──────────┘
                   │
    ═══════════════╪═══════════════
    MILESTONE 5:   │  Application
    ═══════════════╪═══════════════
                   │
                   ▼
        ┌─────────────────────┐
        │ Phase 24-32          │
        │ Deploy & Secure      │
        └──────────┬──────────┘
                   │
                   ▼
                 DONE
```

---

## Milestone 1 — AI Foundation

> **Goal**: Train a change detection model on LEVIR-CD and produce working change maps

### Phase 0 — Define What We're Building

**Status**: 📋 Planning

Three distinct AI problems to solve:

| Model | Input | Output | Candidate Architecture |
|-------|-------|--------|----------------------|
| **Change Detection** | Image pair (T1, T2) | Binary change mask | Siamese U-Net → ChangeFormer |
| **Semantic Retrieval** | Text query or image | Top-K similar tiles | RemoteCLIP |
| **Credibility Engine** | Raw change + metadata | Confidence score | Rule-based + learned |

**Deliverable**: Clear problem definitions and success criteria for each model.

---

### Phase 1 — Setup Development Environment

**Status**: 📋 Planning

**Project structure:**
```
SpaceNetra/
├── data/
├── models/
├── checkpoints/
├── outputs/
├── notebooks/
├── src/
├── train.py
├── evaluate.py
├── inference.py
├── requirements.txt
└── README.md
```

**Environment setup:**
```bash
python -m venv venv
.\venv\Scripts\activate

# Core ML
pip install torch torchvision torchaudio
pip install numpy pandas pillow
pip install matplotlib opencv-python
pip install scikit-learn tqdm

# Geospatial
pip install rasterio geopandas shapely
pip install torchgeo

# Later
pip install faiss-cpu
```

**Deliverable**: Working Python environment with all dependencies.

---

### Phase 2 — Understand Training Data

**Status**: 📋 Planning

**Dataset**: LEVIR-CD
- 637 image pairs
- RGB imagery
- ~1024 × 1024 pixels each
- Binary change masks (building changes)

**One training sample:**
```
T1 Image (2020)  +  T2 Image (2025)  +  Ground Truth Mask
                                          0 = No change
                                          255 = Change
```

**Deliverable**: Downloaded and verified LEVIR-CD dataset.

---

### Phase 3 — Inspect the Data

**Status**: 📋 Planning

Before training, verify:

- [x] Images load correctly
- [x] Dimensions match (T1 = T2 = Mask)
- [x] Image pairs correspond (same filename in A/, B/, label/)
- [x] Masks are valid (only 0 and 255 values)
- [x] No corrupted files
- [x] Labels contain change pixels (not all-zero)
- [x] Train/val/test splits are correct

**Visual check**: Side-by-side display of T1 | T2 | Ground Truth

**Deliverable**: Data inspection notebook with validation results.

---

### Phase 4 — Build Dataset Loader

**Status**: 📋 Planning

PyTorch Dataset class that maps:
```
A/xxx.png  → T1 image (old)
B/xxx.png  → T2 image (new)
label/xxx.png → change mask (ground truth)
```

Returns tensors:
```
PNG → PIL/OpenCV → NumPy → Tensor → PyTorch Model
```

Each sample: `(t1_tensor, t2_tensor, mask_tensor)`

**Deliverable**: Working `LEVIRCDDataset` class with unit tests.

---

### Phase 5 — Split / Patch Images

**Status**: 📋 Planning

Original 1024×1024 images → 256×256 patches:

```
1024 × 1024
       ↓
┌────┬────┬────┬────┐
│256 │256 │256 │256 │
├────┼────┼────┼────┤
│256 │256 │256 │256 │
├────┼────┼────┼────┤
│256 │256 │256 │256 │
├────┼────┼────┼────┤
│256 │256 │256 │256 │
└────┴────┴────┴────┘
= 16 patches per image pair
```

Patch size 256 is recommended by TorchGeo for LEVIR-CD and compatible with most segmentation architectures (multiples of 32).

**Deliverable**: Tiling utility and patched dataset.

---

### Phase 6 — Data Augmentation

**Status**: 📋 Planning

Augmentations (applied identically to T1, T2, and mask):

| Augmentation | Purpose |
|-------------|---------|
| Horizontal flip | Orientation invariance |
| Vertical flip | Orientation invariance |
| Random rotation (90°) | Rotation invariance |
| Brightness jitter (±10%) | Lighting invariance |
| Contrast jitter (±10%) | Sensor invariance |

> ⚠️ **Caution**: Remote sensing imagery has geographic meaning. Augmentations must be physically plausible. No extreme color shifts, no elastic deformations.

**Deliverable**: Augmentation pipeline integrated into DataLoader.

---

### Phase 7 — Build Baseline Model

**Status**: 📋 Planning

**Architecture**: Siamese U-Net

```
T1 → Encoder (shared weights) → Features A
                                      ↓
                                   Compare (|A-B| or concat)
                                      ↑
T2 → Encoder (shared weights) → Features B
                                      ↓
                                   Decoder
                                      ↓
                                 Change Mask
```

The encoder extracts features; the decoder produces pixel-level change predictions. Weight sharing ensures both images are processed identically.

**Deliverable**: `SiameseUNet` model class.

---

### Phase 8 — Loss Function

**Status**: 📋 Planning

**Problem**: Change pixels << No-change pixels (severe class imbalance)

**Solution**: Combined loss

```
Loss = α × WeightedBCE + (1 - α) × DiceLoss
```

| Loss | Purpose |
|------|---------|
| Weighted BCE | Pixel-level classification, upweights rare class |
| Dice Loss | Directly optimizes overlap (IoU-like) |

**Deliverable**: `CombinedLoss` class with configurable weights.

---

### Phase 9 — Train

**Status**: 📋 Planning

**Training loop:**
```python
for epoch in range(num_epochs):
    for t1, t2, mask in train_loader:
        prediction = model(t1, t2)
        loss = criterion(prediction, mask)
        loss.backward()
        optimizer.step()
```

**Expected progression:**
```
Epoch  1 → Loss: ~0.71
Epoch  5 → Loss: ~0.46
Epoch 10 → Loss: ~0.28
Epoch 20 → Loss: ~0.19
```

> ⚠️ Lower training loss does NOT guarantee success. Must validate.

**Deliverable**: Trained model checkpoint.

---

### Phase 10 — Validation

**Status**: 📋 Planning

**Metrics tracked during training:**

| Metric | What It Measures |
|--------|-----------------|
| **Precision** | Of detected changes, how many are real? |
| **Recall** | Of actual changes, how many did we find? |
| **F1 Score** | Balance of precision and recall |
| **IoU (Jaccard)** | Overlap between predicted and actual regions |

**Deliverable**: Validation metrics logged per epoch, best model saved.

---

### Phase 11 — Prevent Overfitting

**Status**: 📋 Planning

**Warning sign:**
```
Training F1  = 98%
Validation F1 = 70%
```

**Countermeasures:**

| Technique | Implementation |
|-----------|---------------|
| Data augmentation | Phase 6 augmentations |
| Validation monitoring | Early stopping on val loss |
| Learning rate scheduling | ReduceLROnPlateau or CosineAnnealing |
| Regularization | Dropout, weight decay |
| Best checkpoint | Save model with best val F1 |

**Deliverable**: Training with early stopping and best-checkpoint saving.

---

### Phase 12 — Test on Unseen Data

**Status**: 📋 Planning

Test set is **never touched** during development.

**Final evaluation:**
```
Test T1 + Test T2 → Trained Model → Predicted Change → Compare with Ground Truth
```

**Report:**
| Metric | Value |
|--------|-------|
| F1 Score | TBD |
| IoU | TBD |
| Precision | TBD |
| Recall | TBD |
| False Positives | TBD |
| False Negatives | TBD |
| Inference Time | TBD ms/tile |

**Deliverable**: Test evaluation report.

---

### Phase 13 — Visual Evaluation

**Status**: 📋 Planning

Generate comparison grids for SIH demo:

```
┌────────────┬────────────┬────────────┬────────────┐
│   T1       │    T2      │ Ground     │ Prediction │
│ (Before)   │  (After)   │ Truth      │  (AI)      │
└────────────┴────────────┴────────────┴────────────┘
```

This allows judges to immediately verify: **Before → After → Actual Change → AI Prediction**

**Deliverable**: Visual evaluation notebook with sample grids.

---

## Milestone 2 — Real-World Satellite Data

> **Goal**: Move from benchmark data to actual Indian satellite imagery

### Phase 14 — Improve the Model

**Status**: 📋 Planning

After baseline, experiment with stronger architectures:

```
Baseline Siamese U-Net → ChangeFormer (Transformer-based)
```

**Principle**: Use the simplest model that gives reliable results for our data and hardware.

**Deliverable**: Comparative evaluation of model architectures.

---

### Phase 15 — Indian Satellite Data

**Status**: 📋 Planning

**Critical**: Cannot claim an Indian defence system trained only on Texas RGB imagery.

**Approach:**
```
LEVIR-CD → Baseline model → Indian EO data → Fine-tuning → Indian validation
```

**Sentinel-2 bands at 10m resolution:**
| Band | Name | Use |
|------|------|-----|
| B02 | Blue | RGB composite |
| B03 | Green | RGB composite |
| B04 | Red | RGB composite |
| B08 | NIR | Vegetation, change sensitivity |

**Deliverable**: Fine-tuned model on Indian Sentinel-2 data.

---

### Phase 16 — Satellite Preprocessing Pipeline

**Status**: 📋 Planning

```
Raw Scene → Cloud Mask → Shadow Mask → Haze Check → Radiometric Norm
    → Geometric Alignment → Resampling → AOI Extraction → Tiling → AI
```

Sentinel-2 Scene Classification Layer (SCL) provides:
- Cloud detection classes
- Shadow classes
- Cirrus detection

**Deliverable**: Complete preprocessing pipeline.

---

### Phase 17 — Geo-Registration

**Status**: 📋 Planning

**Problem**: Misaligned images create false changes.

```
T1:  🏠                    (Building at position X)
T2:    🏠                  (Same building, shifted by registration error)
AI:  "Building changed!"   (FALSE — image shifted)
```

**Solution**:
- Coregistration to reference frame
- Sub-pixel alignment verification
- Registration quality score
- Reject pairs with low registration quality

**Deliverable**: Geo-registration module with quality scoring.

---

## Milestone 3 — Semantic Search

> **Goal**: Search satellite imagery using natural language

### Phase 18 — Semantic Search with RemoteCLIP

**Status**: 📋 Planning

```
"construction activity"  →  Text Encoder  →  [0.21, 0.73, ...]
Satellite tile           →  Image Encoder →  [0.20, 0.71, ...]
                                    ↓
                            Cosine Similarity
                                    ↓
                              Top-K Results
```

RemoteCLIP is designed for remote-sensing image/text retrieval and zero-shot tasks.

**Deliverable**: Working text-to-satellite-tile search.

---

### Phase 19 — Vector Database

**Status**: 📋 Planning

**Each tile stored with:**
```json
{
  "tile_id": "IND_000451",
  "vector": [0.21, 0.73, ...],     // → FAISS/Qdrant
  "latitude": 19.87,
  "longitude": 75.34,
  "date": "2025-04-12",
  "sensor": "Sentinel-2",
  "resolution_m": 10.0,
  "cloud_pct": 3.2,
  "quality_score": 94.5
}
```

- **Embeddings** → FAISS / Qdrant
- **Metadata** → PostgreSQL + PostGIS

**Deliverable**: Indexed vector database with metadata.

---

### Phase 20 — Multi-Temporal Engine

**Status**: 📋 Planning

For a location, retrieve all temporal observations:

```
Timeline:
2022 ── Stable
2023 ── Stable
2024 ── Clearing
2025 ── Construction
2026 ── Expansion
```

This produces a **Change Fingerprint** — the time-based signature of how a location evolved.

**Deliverable**: Temporal analysis engine with change fingerprinting.

---

## Milestone 4 — Intelligence Engine

> **Goal**: Suppress false alarms, create meaningful change events

### Phase 21 — False Alarm Suppression

**Status**: 📋 Planning

Not every visual difference is real change. Causes of false alarms:
- Cloud / shadow
- Seasonal variation
- Different sensors
- Registration errors

**Credibility scoring:**
```
AI Change Score:      91%
Image Quality:        96%
Alignment Score:      94%
Temporal Persistence: 4/5 sessions
Cloud Score:          Low risk

→ Final Confidence:   HIGH
```

**Deliverable**: Multi-signal credibility engine.

---

### Phase 22 — Change Event Creation

**Status**: 📋 Planning

**Aggregate detections into meaningful events:**

```
50 tile-level detections → Group spatially/temporally → 1 Change Event
```

**Change Event:**
```
CHANGE EVENT #021
────────────────────────
Location:        28.61°N, 77.21°E
First observed:  2024-03-15
Latest observed: 2026-01-20
Type:            Construction
Confidence:      HIGH
Evidence:        5 image pairs across 3 dates
```

Much more useful than raw segmentation masks.

**Deliverable**: Event aggregation and formatting system.

---

### Phase 23 — Similar-Site Discovery

**Status**: 📋 Planning

```
Selected location → Embedding → Vector Search → Similar locations
```

Find locations with similar visual/temporal patterns. Powerful demo feature.

**Deliverable**: Similar-site search API.

---

## Milestone 5 — Full Application

> **Goal**: Deploy the complete system as an offline GIS application

### Phase 24 — Backend (FastAPI)

**Status**: 📋 Planning

**Key APIs:**
```
POST /api/search           → Semantic search
POST /api/change-detect    → Run change detection
GET  /api/imagery/{id}     → Serve imagery tiles
GET  /api/timeline/{loc}   → Location timeline
GET  /api/change-events    → List change events
GET  /api/provenance/{id}  → Full provenance chain
POST /api/feedback         → Analyst feedback
```

**Deliverable**: FastAPI backend with all endpoints.

---

### Phase 25 — Frontend (React + GIS)

**Status**: 📋 Planning

**Dashboard layout:**
```
┌───────────────────────────────────────────────────┐
│ SPACENETRA — Satellite Intelligence Engine        │
├───────────────────────────────────────────────────┤
│ Search: [ construction activity near...     ] 🔍  │
├───────────────────────────────────────────────────┤
│                                                   │
│              GIS MAP                              │
│              (Leaflet / MapLibre)                  │
│       ● Change Event #021                         │
│             ●  #035                               │
│                         ● #042                    │
│                                                   │
├───────────────────────────────────────────────────┤
│ RESULT                                            │
│ Before       After        Change Mask             │
│ [IMAGE]      [IMAGE]      [OVERLAY]               │
│                                                   │
│ Confidence: 93%    Type: Construction             │
│ First observed: 2024-03-15                        │
│ [ACCEPT]  [REJECT]  [NEED REVIEW]                 │
└───────────────────────────────────────────────────┘
```

**Deliverable**: React GIS dashboard.

---

### Phase 26 — Provenance

**Status**: 📋 Planning

Every result answers: **"Where did this come from?"**

Stored for each result:
- Source scene ID & sensor
- Acquisition date
- Processing pipeline version
- Model version & hash
- Preprocessing steps applied
- Query that triggered the result
- Confidence calculation details
- Timestamp

**Deliverable**: Full provenance chain for every detection.

---

### Phase 27 — Analyst Feedback Loop

**Status**: 📋 Planning

```
AI Prediction → Analyst Review → [ACCEPT / REJECT / REVIEW]
                                        ↓
                              Stored as training signal
                                        ↓
                              Future model improvement
```

Human-in-the-loop: AI is not the final authority.

**Deliverable**: Feedback capture, storage, and reporting.

---

### Phase 28 — Incremental Ingestion

**Status**: 📋 Planning

When new satellite scenes arrive:
```
NEW SCENE → Quality check → Preprocess → Tile → Embed → Index → Temporal analysis
```

No full archive rebuild. Incremental updates only.

**Deliverable**: Incremental ingestion pipeline.

---

### Phase 29 — Offline Deployment

**Status**: 📋 Planning

```
┌────────────────────────────────────────┐
│         LOCAL SERVER (Air-gapped)       │
│                                        │
│  React UI → FastAPI → AI Models        │
│     → Vector DB → PostGIS              │
│     → Satellite Archive                │
│                                        │
│         NO CLOUD DEPENDENCY            │
└────────────────────────────────────────┘
```

Internet needed only for authorized data/model synchronization.

**Deliverable**: Fully offline-capable deployment.

---

### Phase 30 — Docker Compose

**Status**: 📋 Planning

```yaml
services:
  frontend:       # React (Nginx)
  backend:        # FastAPI
  model-service:  # PyTorch inference
  qdrant:         # Vector database
  postgres:       # PostgreSQL + PostGIS
```

**Deliverable**: `docker-compose.yml` with all services.

---

### Phase 31 — Security

**Status**: 📋 Planning

| Security Layer | Implementation |
|---------------|---------------|
| No external AI API | All models run locally |
| No telemetry | Zero external connections |
| Authentication | JWT / API keys |
| Authorization | Role-based access control |
| Audit logging | All operations logged |
| Encrypted storage | At-rest encryption |
| Network isolation | Air-gap capable |
| Model integrity | Checksum verification |
| Input validation | All inputs sanitized |
| Container isolation | Minimal base images |

**Deliverable**: Security hardening across all layers.

---

### Phase 32 — Final Evaluation & Testing

**Status**: 📋 Planning

**Model Evaluation Report:**
```
Dataset:         LEVIR-CD
Image Pairs:     637
Patch Size:      256 × 256
IoU:             TBD
F1:              TBD
Precision:       TBD
Recall:          TBD
Inference:       TBD ms/tile
Hardware:        TBD
```

**System Evaluation Report:**
```
Indexed Area:    TBD km²
Scenes:          TBD
Tiles:           TBD
Index Build:     TBD min
Storage:         TBD GB
Search Latency:  TBD ms
Change Detection: TBD ms
```

**Deliverable**: Complete evaluation report ready for SIH submission.

---

## Development Priority Summary

```
                 START
                   │
                   ▼
        ┌─────────────────────┐
        │ 1. Environment      │  ← We start here
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 2. LEVIR-CD Data    │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 3. Dataset Loader   │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 4. Baseline Model   │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 5. Train            │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 6. Evaluate         │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 7. Improve          │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 8. Indian EO Data   │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 9. Semantic Search  │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 10. Vector DB       │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 11. Temporal Engine │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 12. Confidence      │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 13. FastAPI Backend │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 14. GIS Dashboard   │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 15. Docker Deploy   │
        └──────────┬──────────┘
                   ▼
        ┌─────────────────────┐
        │ 16. Offline Deploy  │
        └──────────┬──────────┘
                   ▼
                 DONE
```
