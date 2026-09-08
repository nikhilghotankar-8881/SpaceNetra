# SpaceNetra Production Deployment Guide

Enterprise real-time satellite change detection & geospatial AI intelligence engine deployment documentation.

---

## 1. Prerequisites

- **Docker Engine**: v20.10+
- **Docker Compose**: v2.0+
- **NVIDIA Container Toolkit** (Optional for CUDA GPU acceleration)
- **Python**: 3.10+ (for local CLI administration)

---

## 2. Environment Configuration

Copy the example environment configuration template:

```bash
cp .env.example .env
```

Key environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql://spacenetra:spacenetra_pass@postgres:5432/spacenetra_db` | PostGIS database connection string |
| `QDRANT_HOST` | `qdrant` | Vector database host address |
| `QDRANT_PORT` | `6333` | Vector database port |
| `SPACENETRA_SECRET_KEY` | `spacenetra_prod_secret_key_2026` | Secret key for JWT HS256 signing |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID for PyTorch acceleration |

---

## 3. Container Stack Architecture

SpaceNetra orchestrates 5 containerized services:

1. **`frontend`** (`port 80`): Nginx web proxy serving static operational UI assets and proxying `/api` traffic.
2. **`backend`** (`port 8000`): FastAPI REST API backend handling change detection requests, DB ORM operations, and WebSocket streams.
3. **`model-service`** (`port 8001`): Dedicated PyTorch real-time inference worker (Siamese U-Net / ChangeFormer Vision Transformer).
4. **`qdrant`** (`port 6333`): Vector database indexing RemoteCLIP 512-dim multimodal embeddings.
5. **`postgres`** (`port 5432`): PostGIS geospatial database persisting scenes, tiles, change events, analyst feedback, and provenance logs.

---

## 4. Launching the Production Stack

### Build and Start Containers

```bash
docker-compose up --build -d
```

### Verify Container Health

```bash
python spacenetra_cli.py verify
```

### View Real-Time Service Logs

```bash
docker-compose logs -f backend
```

---

## 5. Monitoring & Telemetry Integration

- **Prometheus Metrics Endpoint**: `http://localhost:8000/api/metrics`
- **Deep Health Diagnostic Endpoint**: `http://localhost:8000/api/health/deep`
- **Live WebSocket Alert Stream**: `ws://localhost:8000/api/ws/alerts`

---

## 6. Enterprise CLI Administration

SpaceNetra provides a unified CLI orchestrator:

```bash
# Run Sentinel-2 Satellite Change Inference
python spacenetra_cli.py predict --t1 data/scene_t1.tif --t2 data/scene_t2.tif --arch changeformer

# Execute Multi-Temporal Trend Analysis
python spacenetra_cli.py analyze --images data/t1.tif data/t2.tif data/t3.tif --metric ndvi

# Perform RemoteCLIP Semantic Search
python spacenetra_cli.py search --query "industrial expansion" --top_k 5

# Start Application Server
python spacenetra_cli.py serve --port 8000
```
