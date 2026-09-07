# 🛰️ SpaceNetra — Satellite Intelligence Engine

> **AI-Powered Satellite Imagery Analysis for Change Detection, Semantic Search & Geospatial Intelligence**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [AI Models](#ai-models)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Development Phases](#development-phases)
- [Tech Stack](#tech-stack)
- [Hardware Requirements](#hardware-requirements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**SpaceNetra** is an offline-first, AI-driven satellite intelligence platform that combines:

1. **Change Detection** — Automatically identify what changed between two satellite acquisitions
2. **Semantic Retrieval** — Search satellite imagery using natural language queries
3. **Temporal Intelligence** — Track how locations evolve over time with confidence scoring
4. **Analyst Dashboard** — Interactive GIS interface with provenance and evidence chains

The system is designed for **on-premise deployment** with no cloud dependency during operation, making it suitable for defence and intelligence applications.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   SATELLITE DATA    │
                    │ Sentinel / Other EO │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  DATA INGESTION     │
                    │ GeoTIFF / COG       │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌──────────────────────────┐
                 │ PREPROCESSING             │
                 │ • Cloud masking           │
                 │ • Haze/quality filtering  │
                 │ • Radiometric norm.       │
                 │ • Geo-registration        │
                 │ • Tiling (256×256)        │
                 └────────────┬─────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
       ┌─────────────────┐         ┌─────────────────┐
       │ SEMANTIC MODEL  │         │ CHANGE MODEL    │
       │ Image/Text      │         │ T1 + T2         │
       │ Embeddings      │         │ → Change Map    │
       └────────┬────────┘         └────────┬────────┘
                │                           │
                ▼                           ▼
       ┌─────────────────┐         ┌─────────────────┐
       │ VECTOR DATABASE │         │ CHANGE EVENTS   │
       │ FAISS / Qdrant  │         │ + Confidence    │
       └────────┬────────┘         └────────┬────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ ANALYST DASHBOARD   │
                    │ Map + Timeline      │
                    │ Before / After      │
                    │ Evidence chains     │
                    │ Full Provenance     │
                    └─────────────────────┘
```

---

## AI Models

### Model 1 — Change Detection (Siamese U-Net → ChangeFormer)

| Aspect | Detail |
|--------|--------|
| **Input** | Image pair (T1, T2) |
| **Output** | Binary change mask with confidence |
| **Baseline** | Siamese U-Net |
| **Advanced** | ChangeFormer (Transformer-based) |
| **Training Data** | LEVIR-CD → Indian EO (Sentinel-2) |
| **Patch Size** | 256 × 256 |

### Model 2 — Semantic Retrieval (RemoteCLIP)

| Aspect | Detail |
|--------|--------|
| **Input** | Text query or satellite tile |
| **Output** | Top-K similar tiles from vector index |
| **Model** | RemoteCLIP (remote-sensing CLIP) |
| **Index** | FAISS / Qdrant |

### Model 3 — Credibility / Confidence Engine

| Aspect | Detail |
|--------|--------|
| **Input** | Raw change detection + metadata |
| **Output** | Calibrated confidence score |
| **Signals** | Image quality, alignment, temporal persistence, cloud, cross-sensor consistency |

---

## Project Structure

```
SpaceNetra/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── requirements-dev.txt               # Development dependencies
├── setup.py                           # Package setup
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── Makefile                           # Common commands
├── docker-compose.yml                 # Multi-container deployment
├── Dockerfile                         # Main application container
│
├── docs/                              # Documentation
│   ├── ARCHITECTURE.md                # System architecture deep-dive
│   ├── ROADMAP.md                     # Development roadmap & phases
│   ├── DATA_PIPELINE.md              # Data ingestion & preprocessing
│   ├── MODEL_TRAINING.md            # Training guide
│   ├── DEPLOYMENT.md                 # Deployment guide
│   ├── API_REFERENCE.md             # API documentation
│   ├── SECURITY.md                   # Security architecture
│   ├── EVALUATION.md                 # Evaluation methodology
│   └── CONTRIBUTING.md              # Contribution guidelines
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                           # Raw satellite scenes
│   ├── processed/                     # Preprocessed tiles
│   ├── levir_cd/                      # LEVIR-CD benchmark dataset
│   └── embeddings/                    # Precomputed embeddings
│
├── models/                            # Model definitions
│   ├── __init__.py
│   ├── siamese_unet.py              # Siamese U-Net for change detection
│   ├── change_former.py             # ChangeFormer architecture
│   ├── remote_clip.py               # RemoteCLIP wrapper
│   ├── confidence.py                # Credibility/confidence engine
│   └── losses.py                    # Loss functions (BCE + Dice)
│
├── checkpoints/                       # Saved model weights (gitignored)
│
├── outputs/                           # Training outputs & visualizations
│   ├── predictions/                   # Predicted change masks
│   ├── metrics/                       # Training/eval metrics
│   └── visualizations/               # Side-by-side comparisons
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_inspection.ipynb      # Dataset exploration
│   ├── 02_training_experiments.ipynb # Training experiments
│   ├── 03_evaluation.ipynb           # Model evaluation
│   └── 04_semantic_search.ipynb      # Semantic search demo
│
├── src/                               # Core source code
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   │
│   ├── data/                         # Data loading & preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py               # PyTorch dataset classes
│   │   ├── datamodule.py            # Data module (train/val/test)
│   │   ├── transforms.py           # Augmentations & transforms
│   │   ├── preprocessing.py        # Cloud masking, normalization
│   │   ├── tiling.py               # Image tiling (1024→256)
│   │   └── geo_registration.py     # Geometric alignment
│   │
│   ├── training/                     # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py               # Training loop
│   │   ├── scheduler.py            # LR scheduling
│   │   └── callbacks.py            # Early stopping, checkpointing
│   │
│   ├── evaluation/                   # Evaluation & metrics
│   │   ├── __init__.py
│   │   ├── metrics.py              # F1, IoU, Precision, Recall
│   │   └── visualizer.py           # Visual evaluation
│   │
│   ├── retrieval/                    # Semantic search
│   │   ├── __init__.py
│   │   ├── embeddings.py           # Embedding generation
│   │   ├── vector_store.py         # FAISS/Qdrant interface
│   │   └── search.py               # Search engine
│   │
│   ├── temporal/                     # Temporal intelligence
│   │   ├── __init__.py
│   │   ├── timeline.py             # Multi-temporal analysis
│   │   ├── change_events.py        # Event aggregation
│   │   └── fingerprint.py          # Change fingerprints
│   │
│   ├── confidence/                   # Confidence engine
│   │   ├── __init__.py
│   │   ├── quality.py              # Image quality scoring
│   │   ├── alignment.py            # Registration quality
│   │   └── scoring.py              # Final confidence scoring
│   │
│   └── ingestion/                    # Data ingestion pipeline
│       ├── __init__.py
│       ├── sentinel.py             # Sentinel-2 data handler
│       ├── geotiff.py              # GeoTIFF/COG reader
│       └── incremental.py          # Incremental ingestion
│
├── backend/                           # FastAPI backend
│   ├── main.py                       # Application entry point
│   ├── api/                          # API routes
│   │   ├── __init__.py
│   │   ├── search.py               # Semantic search endpoints
│   │   ├── change.py               # Change detection endpoints
│   │   ├── imagery.py              # Imagery serving
│   │   ├── events.py               # Change events
│   │   ├── timeline.py             # Timeline queries
│   │   └── provenance.py           # Provenance chain
│   │
│   ├── services/                     # Business logic
│   │   ├── __init__.py
│   │   ├── retrieval.py            # Search service
│   │   ├── change_detection.py     # CD inference service
│   │   ├── quality.py              # Quality assessment
│   │   ├── confidence.py           # Confidence scoring
│   │   └── feedback.py             # Analyst feedback
│   │
│   ├── database/                     # Database layer
│   │   ├── __init__.py
│   │   ├── models.py               # SQLAlchemy models
│   │   ├── connection.py           # DB connections
│   │   └── migrations/             # Alembic migrations
│   │
│   └── middleware/                   # Middleware
│       ├── auth.py                  # Authentication
│       ├── audit.py                 # Audit logging
│       └── security.py             # Security headers
│
├── frontend/                          # React frontend
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Map/                 # GIS map (Leaflet/MapLibre)
│   │   │   ├── Search/              # Search interface
│   │   │   ├── Timeline/            # Temporal timeline
│   │   │   ├── Comparison/          # Before/After viewer
│   │   │   ├── Results/             # Results panel
│   │   │   └── Feedback/            # Analyst feedback
│   │   ├── services/                # API client
│   │   └── utils/                   # Utilities
│   └── public/
│
├── scripts/                           # Utility scripts
│   ├── download_levir_cd.py         # Download LEVIR-CD dataset
│   ├── download_sentinel.py         # Download Sentinel-2 scenes
│   ├── build_vector_index.py        # Build FAISS/Qdrant index
│   └── evaluate_model.py           # Run full evaluation suite
│
├── tests/                             # Test suite
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_training.py
│   ├── test_inference.py
│   ├── test_search.py
│   └── test_api.py
│
├── train.py                           # Main training entry point
├── evaluate.py                        # Main evaluation entry point
└── inference.py                       # Main inference entry point
```

---

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd SpaceNetra

python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

pip install -r requirements.txt
```

### 2. Download LEVIR-CD Dataset

```bash
python scripts/download_levir_cd.py --output data/levir_cd/
```

### 3. Inspect Data

```bash
jupyter notebook notebooks/01_data_inspection.ipynb
```

### 4. Train Baseline Model

```bash
python train.py --config configs/baseline.yaml
```

### 5. Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth --dataset data/levir_cd/test/
```

### 6. Run Inference

```bash
python inference.py --t1 path/to/image1.tif --t2 path/to/image2.tif --output outputs/predictions/
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Deep Learning** | PyTorch, TorchVision, TorchGeo |
| **Change Detection** | Siamese U-Net, ChangeFormer |
| **Semantic Search** | RemoteCLIP |
| **Vector Database** | FAISS / Qdrant |
| **Geospatial** | Rasterio, GeoPandas, Shapely, PostGIS |
| **Backend** | FastAPI, SQLAlchemy, Alembic |
| **Frontend** | React, Leaflet / MapLibre GL |
| **Database** | PostgreSQL + PostGIS |
| **Containerization** | Docker, Docker Compose |
| **ML Tracking** | TensorBoard / MLflow |

---

## Hardware Requirements

### Minimum (Training)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **RAM**: 16 GB
- **Storage**: 50 GB free (dataset + checkpoints)
- **CPU**: 8 cores

### Recommended (Full Pipeline)
- **GPU**: NVIDIA RTX 3080/4080 or better (16GB+ VRAM)
- **RAM**: 32 GB
- **Storage**: 200 GB+ SSD
- **CPU**: 12+ cores

### Deployment (Inference Only)
- **GPU**: Optional (CPU inference supported, slower)
- **RAM**: 16 GB
- **Storage**: Based on satellite archive size

---

## Development Phases

See [docs/ROADMAP.md](docs/ROADMAP.md) for the complete 32-phase development roadmap.

**Current Priority Order:**

| Priority | Milestone | Description |
|----------|-----------|-------------|
| 🔴 **1** | Change Detection | LEVIR-CD → Train → Evaluate → Change Maps |
| 🟠 **2** | Indian EO Data | Sentinel-2 preprocessing → Fine-tune |
| 🟡 **3** | Semantic Search | RemoteCLIP + FAISS vector index |
| 🟢 **4** | Confidence Engine | Quality + temporal + alignment scoring |
| 🔵 **5** | Full Application | FastAPI + React + GIS Dashboard |

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **LEVIR-CD Dataset** — Chen & Shi, 2020
- **ChangeFormer** — Bandara & Patel, 2022
- **RemoteCLIP** — Liu et al., 2023
- **TorchGeo** — Stewart et al., 2022
- **Sentinel-2** — European Space Agency (ESA)
