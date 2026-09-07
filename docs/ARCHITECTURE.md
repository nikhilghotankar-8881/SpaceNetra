# 🏗️ System Architecture — SpaceNetra

> Deep-dive into the technical architecture of the Satellite Intelligence Engine

---

## Table of Contents

- [High-Level Architecture](#high-level-architecture)
- [AI Model Architecture](#ai-model-architecture)
- [Data Flow Pipeline](#data-flow-pipeline)
- [Component Architecture](#component-architecture)
- [Database Architecture](#database-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Security Architecture](#security-architecture)

---

## High-Level Architecture

SpaceNetra is composed of **five core subsystems** that operate in a pipeline:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  INGESTION   │────▶│ PREPROCESS   │────▶│  AI MODELS   │────▶│  INTELLIGENCE│────▶│  DASHBOARD   │
│              │     │              │     │              │     │   ENGINE     │     │              │
│ • Sentinel-2 │     │ • Cloud mask │     │ • Change Det │     │ • Temporal   │     │ • GIS Map    │
│ • GeoTIFF    │     │ • Normalize  │     │ • Semantic   │     │ • Confidence │     │ • Timeline   │
│ • COG        │     │ • Register   │     │ • Embeddings │     │ • Events     │     │ • Evidence   │
│ • Incremental│     │ • Tile       │     │              │     │ • Search     │     │ • Provenance │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

---

## AI Model Architecture

### Model 1 — Change Detection (Siamese U-Net)

The baseline architecture uses a **weight-shared Siamese encoder** with a **U-Net decoder**.

```
           T1 IMAGE (256×256×3)              T2 IMAGE (256×256×3)
                    │                                 │
                    ▼                                 ▼
           ┌────────────────┐                ┌────────────────┐
           │   ENCODER      │                │   ENCODER      │
           │ (Shared Weights)│               │ (Shared Weights)│
           │                │                │                │
           │ Conv Block 1   │                │ Conv Block 1   │
           │   64 filters   │                │   64 filters   │
           │ ──────────────│                │ ──────────────│
           │ Conv Block 2   │                │ Conv Block 2   │
           │  128 filters   │                │  128 filters   │
           │ ──────────────│                │ ──────────────│
           │ Conv Block 3   │                │ Conv Block 3   │
           │  256 filters   │                │  256 filters   │
           │ ──────────────│                │ ──────────────│
           │ Conv Block 4   │                │ Conv Block 4   │
           │  512 filters   │                │  512 filters   │
           └───────┬────────┘                └───────┬────────┘
                   │                                 │
                   │         Features A              │        Features B
                   │                                 │
                   └─────────────┬───────────────────┘
                                │
                         ┌──────▼──────┐
                         │  DIFFERENCE │
                         │  |A - B|    │
                         │  or         │
                         │  concat(A,B)│
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │   DECODER   │
                         │             │
                         │ UpConv 4    │ ← Skip Connection
                         │ UpConv 3    │ ← Skip Connection
                         │ UpConv 2    │ ← Skip Connection
                         │ UpConv 1    │ ← Skip Connection
                         │             │
                         │ Conv 1×1    │
                         │ Sigmoid     │
                         └──────┬──────┘
                                │
                                ▼
                     CHANGE MASK (256×256×1)
                     0.0 = No change
                     1.0 = Change
```

### Model 2 — ChangeFormer (Advanced)

After baseline validation, we upgrade to a **transformer-based** architecture:

```
       T1 IMAGE                    T2 IMAGE
          │                           │
          ▼                           ▼
   ┌──────────────┐            ┌──────────────┐
   │ Hierarchical │            │ Hierarchical │
   │ Transformer  │            │ Transformer  │
   │ Encoder      │            │ Encoder      │
   │              │            │              │
   │ Stage 1 (H/4)│           │ Stage 1 (H/4)│
   │ Stage 2 (H/8)│           │ Stage 2 (H/8)│
   │ Stage 3(H/16)│           │ Stage 3(H/16)│
   │ Stage 4(H/32)│           │ Stage 4(H/32)│
   └──────┬───────┘            └──────┬───────┘
          │                           │
          └───────────┬───────────────┘
                      │
               ┌──────▼──────┐
               │  Difference │
               │  Module     │
               └──────┬──────┘
                      │
               ┌──────▼──────┐
               │  MLP Decoder│
               │  Multi-scale│
               └──────┬──────┘
                      │
                      ▼
               CHANGE MASK
```

### Model 3 — Semantic Retrieval (RemoteCLIP)

```
   TEXT QUERY                    SATELLITE TILE
   "construction"                    │
        │                            │
        ▼                            ▼
   ┌──────────┐              ┌──────────────┐
   │  TEXT     │              │   IMAGE      │
   │  ENCODER  │              │   ENCODER    │
   │ (CLIP-   │              │  (ViT /      │
   │  based)  │              │   ResNet)    │
   └────┬─────┘              └──────┬───────┘
        │                           │
        ▼                           ▼
   [Text Embedding]         [Image Embedding]
   dim = 512/768            dim = 512/768
        │                           │
        │    ┌──────────────────┐   │
        └───▶│ COSINE SIMILARITY│◀──┘
             └────────┬─────────┘
                      │
                      ▼
              Similarity Score
              (0.0 → 1.0)
```

### Model 4 — Credibility Engine

```
   RAW CHANGE DETECTION OUTPUT
              │
              ▼
   ┌──────────────────────────────────────────┐
   │         CREDIBILITY ENGINE               │
   │                                          │
   │  ┌────────────┐   ┌────────────────┐    │
   │  │ Image      │   │ Registration   │    │
   │  │ Quality    │   │ Quality        │    │
   │  │ Score      │   │ Score          │    │
   │  │ (0-100)    │   │ (0-100)        │    │
   │  └─────┬──────┘   └──────┬─────────┘    │
   │        │                 │               │
   │  ┌─────▼──────┐   ┌─────▼──────────┐    │
   │  │ Cloud      │   │ Temporal       │    │
   │  │ Score      │   │ Persistence    │    │
   │  │ (0-100)    │   │ (n/N sessions) │    │
   │  └─────┬──────┘   └──────┬─────────┘    │
   │        │                 │               │
   │        └────────┬────────┘               │
   │                 │                        │
   │          ┌──────▼──────┐                 │
   │          │  WEIGHTED   │                 │
   │          │  AGGREGATION│                 │
   │          └──────┬──────┘                 │
   │                 │                        │
   └─────────────────┼────────────────────────┘
                     │
                     ▼
            FINAL CONFIDENCE
            LOW / MEDIUM / HIGH
```

---

## Data Flow Pipeline

### Ingestion → Processing → AI → Dashboard

```
SATELLITE SCENE (Raw)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: INGESTION                                          │
│                                                             │
│  • Download / receive GeoTIFF or COG                        │
│  • Parse metadata (sensor, date, bounds, CRS)               │
│  • Register in metadata database                            │
│  • Check for duplicates                                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: PREPROCESSING                                      │
│                                                             │
│  2a. Cloud Detection                                        │
│      Scene Classification Band → cloud/shadow/cirrus mask   │
│                                                             │
│  2b. Quality Assessment                                     │
│      Haze detection, noise estimation, usable area %        │
│      REJECT if cloud_pct > threshold                        │
│                                                             │
│  2c. Radiometric Normalization                              │
│      DN → TOA Reflectance → Surface Reflectance             │
│      Histogram matching for cross-temporal consistency       │
│                                                             │
│  2d. Geometric Registration                                 │
│      Coregistration to reference frame                      │
│      Sub-pixel alignment verification                       │
│      Registration quality score                             │
│                                                             │
│  2e. Tiling                                                 │
│      Full scene → 256×256 pixel patches                     │
│      With overlap (stride configurable)                     │
│      Georeference preserved per tile                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: AI INFERENCE                                       │
│                                                             │
│  PATH A: Semantic Embedding                                 │
│    Tile → RemoteCLIP Image Encoder → 512-d vector           │
│    → Insert into FAISS/Qdrant index                         │
│                                                             │
│  PATH B: Change Detection                                   │
│    (T1 tile, T2 tile) → Siamese Model → Change probability  │
│    → Binary mask at threshold (e.g., 0.5)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: INTELLIGENCE                                       │
│                                                             │
│  4a. False Alarm Suppression                                │
│      Quality + Cloud + Alignment + Temporal → Confidence    │
│                                                             │
│  4b. Change Event Aggregation                               │
│      50 tile-level detections → 1 event with evidence chain │
│                                                             │
│  4c. Temporal Fingerprinting                                │
│      Location timeline: Stable → Clearing → Construction    │
│                                                             │
│  4d. Similar-Site Discovery                                 │
│      Embedding similarity → locations with similar patterns │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 5: PRESENTATION                                       │
│                                                             │
│  • GIS Map with change event markers                        │
│  • Before / After comparison viewer                         │
│  • Timeline visualization                                   │
│  • Confidence scoring display                               │
│  • Full provenance chain                                    │
│  • Analyst Accept / Reject / Review workflow                 │
│  • Natural language search interface                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### Backend (FastAPI)

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │                  API Layer                        │   │
│  │                                                  │   │
│  │  POST /api/search          → Semantic search     │   │
│  │  POST /api/change-detect   → Run CD inference    │   │
│  │  GET  /api/imagery/{id}    → Serve imagery       │   │
│  │  GET  /api/timeline/{loc}  → Location timeline   │   │
│  │  GET  /api/change-events   → List events         │   │
│  │  GET  /api/provenance/{id} → Provenance chain    │   │
│  │  POST /api/feedback        → Analyst feedback    │   │
│  └───────────────────┬──────────────────────────────┘   │
│                      │                                   │
│  ┌───────────────────▼──────────────────────────────┐   │
│  │               Service Layer                       │   │
│  │                                                  │   │
│  │  RetrievalService  → RemoteCLIP + Vector DB      │   │
│  │  ChangeService     → Siamese Model inference     │   │
│  │  QualityService    → Image quality assessment    │   │
│  │  ConfidenceService → Credibility scoring         │   │
│  │  FeedbackService   → Analyst feedback loop       │   │
│  │  TimelineService   → Temporal analysis           │   │
│  └───────────────────┬──────────────────────────────┘   │
│                      │                                   │
│  ┌───────────────────▼──────────────────────────────┐   │
│  │               Data Layer                          │   │
│  │                                                  │   │
│  │  PostgreSQL + PostGIS  → Metadata, events, geo   │   │
│  │  FAISS / Qdrant        → Vector embeddings       │   │
│  │  File System           → Imagery tiles, models   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Middleware                            │   │
│  │                                                  │   │
│  │  Authentication  → JWT / API Key                 │   │
│  │  Audit Logging   → All API calls logged          │   │
│  │  Security        → CORS, rate limiting, headers  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Frontend (React)

```
┌─────────────────────────────────────────────────────────┐
│                   React Application                      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Header / Navigation                              │   │
│  │  [SpaceNetra Logo]    [Search]    [Settings]      │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────┐  ┌─────────────────────────┐  │
│  │                     │  │                         │  │
│  │    GIS MAP          │  │   RESULTS PANEL         │  │
│  │    (Leaflet /       │  │                         │  │
│  │     MapLibre)       │  │   • Search Results      │  │
│  │                     │  │   • Change Events       │  │
│  │    • Tile layers    │  │   • Confidence Score    │  │
│  │    • Event markers  │  │   • Before/After        │  │
│  │    • AOI drawing    │  │   • Provenance          │  │
│  │    • Heatmaps       │  │   • Timeline            │  │
│  │                     │  │                         │  │
│  └─────────────────────┘  │   [ACCEPT] [REJECT]     │  │
│                           │   [NEED REVIEW]         │  │
│  ┌─────────────────────┐  └─────────────────────────┘  │
│  │  TIMELINE BAR       │                                │
│  │  2022──2023──2024──2025──2026                        │
│  └─────────────────────┘                                │
└─────────────────────────────────────────────────────────┘
```

---

## Database Architecture

### PostgreSQL + PostGIS Schema

```sql
-- Satellite scenes metadata
CREATE TABLE scenes (
    id              UUID PRIMARY KEY,
    scene_id        VARCHAR(255) UNIQUE NOT NULL,
    sensor          VARCHAR(50) NOT NULL,         -- 'Sentinel-2', 'Landsat-8'
    acquisition_date TIMESTAMP NOT NULL,
    bounds          GEOMETRY(Polygon, 4326),       -- PostGIS geometry
    cloud_cover_pct FLOAT,
    quality_score   FLOAT,
    crs             VARCHAR(20),
    resolution_m    FLOAT,
    file_path       TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Individual tiles (patches)
CREATE TABLE tiles (
    id              UUID PRIMARY KEY,
    tile_id         VARCHAR(255) UNIQUE NOT NULL,
    scene_id        UUID REFERENCES scenes(id),
    center          GEOMETRY(Point, 4326),
    bounds          GEOMETRY(Polygon, 4326),
    width_px        INT DEFAULT 256,
    height_px       INT DEFAULT 256,
    cloud_pct       FLOAT,
    file_path       TEXT,
    embedding_id    VARCHAR(255),                  -- Reference to vector DB
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Change events
CREATE TABLE change_events (
    id              UUID PRIMARY KEY,
    location        GEOMETRY(Point, 4326),
    bounds          GEOMETRY(Polygon, 4326),
    first_observed  TIMESTAMP,
    last_observed   TIMESTAMP,
    change_type     VARCHAR(100),                  -- 'construction', 'demolition'
    confidence      FLOAT,
    confidence_level VARCHAR(20),                  -- 'LOW', 'MEDIUM', 'HIGH'
    tile_ids        UUID[],
    evidence_count  INT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Provenance records
CREATE TABLE provenance (
    id              UUID PRIMARY KEY,
    event_id        UUID REFERENCES change_events(id),
    source_scene_id VARCHAR(255),
    sensor          VARCHAR(50),
    acquisition_date TIMESTAMP,
    processing_version VARCHAR(50),
    model_version   VARCHAR(50),
    model_hash      VARCHAR(64),
    preprocessing_version VARCHAR(50),
    query_text      TEXT,
    timestamp       TIMESTAMP DEFAULT NOW()
);

-- Analyst feedback
CREATE TABLE analyst_feedback (
    id              UUID PRIMARY KEY,
    event_id        UUID REFERENCES change_events(id),
    analyst_id      VARCHAR(255),
    decision        VARCHAR(20),                   -- 'ACCEPT', 'REJECT', 'REVIEW'
    notes           TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Spatial indexes
CREATE INDEX idx_scenes_bounds ON scenes USING GIST(bounds);
CREATE INDEX idx_tiles_center ON tiles USING GIST(center);
CREATE INDEX idx_tiles_bounds ON tiles USING GIST(bounds);
CREATE INDEX idx_events_location ON change_events USING GIST(location);
CREATE INDEX idx_events_bounds ON change_events USING GIST(bounds);
```

### Vector Database Schema (FAISS / Qdrant)

```json
{
  "collection": "satellite_embeddings",
  "vector_size": 512,
  "distance": "cosine",
  
  "point_schema": {
    "id": "string (tile_id)",
    "vector": "float[512]",
    "payload": {
      "tile_id": "string",
      "scene_id": "string",
      "latitude": "float",
      "longitude": "float",
      "date": "ISO 8601 string",
      "sensor": "string",
      "resolution_m": "float",
      "cloud_pct": "float",
      "quality_score": "float"
    }
  }
}
```

---

## Deployment Architecture

### Offline Deployment (Target)

```
┌─────────────────────────────────────────────────────────────┐
│                     LOCAL SERVER                             │
│                     (Air-gapped)                             │
│                                                             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│  │  React    │  │  FastAPI  │  │  Model    │              │
│  │  Frontend │──│  Backend  │──│  Service  │              │
│  │  :3000    │  │  :8000    │  │  (PyTorch)│              │
│  └───────────┘  └─────┬─────┘  └───────────┘              │
│                       │                                     │
│  ┌───────────┐  ┌─────▼─────┐  ┌───────────┐              │
│  │  Qdrant   │  │ PostgreSQL│  │  File     │              │
│  │  Vector DB│  │ + PostGIS │  │  Storage  │              │
│  │  :6333    │  │  :5432    │  │  (Tiles)  │              │
│  └───────────┘  └───────────┘  └───────────┘              │
│                                                             │
│        ════════════════════════════════════                  │
│                   NO INTERNET                               │
│                   DURING OPERATION                           │
│        ════════════════════════════════════                  │
└─────────────────────────────────────────────────────────────┘
```

### Docker Compose Services

```yaml
services:
  frontend:     # React app (Nginx)
  backend:      # FastAPI application
  model-service: # PyTorch inference server
  qdrant:       # Vector database
  postgres:     # PostgreSQL + PostGIS
```

---

## Security Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  SECURITY LAYERS                          │
│                                                          │
│  Layer 1: Network Isolation                              │
│  ├── Air-gapped deployment                               │
│  ├── No external API calls                               │
│  └── No telemetry / analytics                            │
│                                                          │
│  Layer 2: Authentication & Authorization                 │
│  ├── JWT-based authentication                            │
│  ├── Role-based access control (RBAC)                    │
│  └── API key management                                  │
│                                                          │
│  Layer 3: Data Protection                                │
│  ├── Encrypted storage (at rest)                         │
│  ├── TLS for inter-service communication                 │
│  └── No secrets in source code                           │
│                                                          │
│  Layer 4: Audit & Accountability                         │
│  ├── All API calls logged                                │
│  ├── Analyst decisions recorded                          │
│  ├── Model version tracking                              │
│  └── Full provenance chains                              │
│                                                          │
│  Layer 5: Container Isolation                            │
│  ├── Each service in separate container                  │
│  ├── Minimal base images                                 │
│  └── No root processes                                   │
│                                                          │
│  Layer 6: Model Integrity                                │
│  ├── Model checksum verification                         │
│  ├── Version-controlled checkpoints                      │
│  └── Input validation/sanitization                       │
└──────────────────────────────────────────────────────────┘
```
