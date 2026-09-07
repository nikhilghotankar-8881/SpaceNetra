# 🚀 Deployment Guide — SpaceNetra

> From development to offline, air-gapped deployment

---

## Table of Contents

- [Deployment Overview](#deployment-overview)
- [Deployment Modes](#deployment-modes)
- [Docker Architecture](#docker-architecture)
- [Docker Compose Configuration](#docker-compose-configuration)
- [Building Containers](#building-containers)
- [Offline Deployment](#offline-deployment)
- [Data Transfer Procedure](#data-transfer-procedure)
- [Health Checks](#health-checks)
- [Scaling Considerations](#scaling-considerations)
- [Troubleshooting](#troubleshooting)

---

## Deployment Overview

SpaceNetra is designed for **offline-first deployment**. The complete stack runs on a local server with no cloud dependencies during operation.

```
┌─────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT TARGET                        │
│                     Local Server (Air-gapped)                │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Docker Compose Stack                      │  │
│  │                                                       │  │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────────┐         │  │
│  │  │ Nginx   │  │ FastAPI │  │ Model Service│         │  │
│  │  │ (React) │──│ Backend │──│ (PyTorch)    │         │  │
│  │  │ :80     │  │ :8000   │  │ :8001        │         │  │
│  │  └─────────┘  └────┬────┘  └──────────────┘         │  │
│  │                    │                                  │  │
│  │  ┌─────────┐  ┌────▼────┐  ┌──────────────┐         │  │
│  │  │ Qdrant  │  │PostgreSQL│ │ Shared Volume│         │  │
│  │  │ :6333   │  │+ PostGIS │ │ (Tiles/Data) │         │  │
│  │  │         │  │ :5432   │  │              │         │  │
│  │  └─────────┘  └─────────┘  └──────────────┘         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ════════════════════════════════════════════════════════    │
│       NO INTERNET REQUIRED DURING OPERATION                  │
│  ════════════════════════════════════════════════════════    │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Modes

### Mode 1: Development

```bash
# Run individual services locally
python -m uvicorn backend.main:app --reload --port 8000
cd frontend && npm run dev
```

### Mode 2: Docker Compose (Recommended for Prototype)

```bash
docker-compose up -d
```

### Mode 3: Air-Gapped Production

```bash
# Pre-build on connected machine
docker-compose build
docker save spacenetra-backend spacenetra-frontend spacenetra-model > spacenetra.tar

# Transfer to air-gapped machine
docker load < spacenetra.tar
docker-compose up -d
```

---

## Docker Architecture

### Service Breakdown

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `frontend` | Nginx + React build | 80 | Serves React SPA, proxies API |
| `backend` | Python + FastAPI | 8000 | REST API, business logic |
| `model-service` | Python + PyTorch | 8001 | AI inference (CD, embeddings) |
| `qdrant` | qdrant/qdrant | 6333 | Vector similarity search |
| `postgres` | postgis/postgis | 5432 | Metadata, events, geospatial |

### Network Topology

```
                    External (Host)
                         │
                    Port 80 only
                         │
                         ▼
                  ┌──────────────┐
                  │   Nginx      │
                  │  (Frontend)  │
                  └──────┬───────┘
                         │
              spacenetra-network (internal)
                         │
            ┌────────────┼────────────┐
            │            │            │
            ▼            ▼            ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐
     │ Backend  │ │  Model   │ │  Qdrant  │
     │ :8000    │ │  :8001   │ │  :6333   │
     └────┬─────┘ └──────────┘ └──────────┘
          │
          ▼
     ┌──────────┐
     │PostgreSQL│
     │  :5432   │
     └──────────┘
```

---

## Docker Compose Configuration

### docker-compose.yml

```yaml
version: '3.8'

services:
  # ────────────────────────────────────────
  # Frontend (React + Nginx)
  # ────────────────────────────────────────
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
    networks:
      - spacenetra-network
    restart: unless-stopped

  # ────────────────────────────────────────
  # Backend (FastAPI)
  # ────────────────────────────────────────
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://spacenetra:${DB_PASSWORD}@postgres:5432/spacenetra
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - MODEL_SERVICE_URL=http://model-service:8001
      - TILE_STORAGE_PATH=/data/tiles
      - LOG_LEVEL=INFO
    volumes:
      - tile-data:/data/tiles
      - model-checkpoints:/app/checkpoints
    depends_on:
      postgres:
        condition: service_healthy
      qdrant:
        condition: service_started
      model-service:
        condition: service_started
    networks:
      - spacenetra-network
    restart: unless-stopped

  # ────────────────────────────────────────
  # Model Service (PyTorch Inference)
  # ────────────────────────────────────────
  model-service:
    build:
      context: .
      dockerfile: Dockerfile.model
    ports:
      - "8001:8001"
    environment:
      - DEVICE=cuda                        # or 'cpu'
      - CD_MODEL_PATH=/app/checkpoints/change_detection.pth
      - CLIP_MODEL_PATH=/app/checkpoints/remote_clip.pth
      - BATCH_SIZE=8
    volumes:
      - model-checkpoints:/app/checkpoints
      - tile-data:/data/tiles:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - spacenetra-network
    restart: unless-stopped

  # ────────────────────────────────────────
  # Vector Database (Qdrant)
  # ────────────────────────────────────────
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant-data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
    networks:
      - spacenetra-network
    restart: unless-stopped

  # ────────────────────────────────────────
  # Database (PostgreSQL + PostGIS)
  # ────────────────────────────────────────
  postgres:
    image: postgis/postgis:16-3.4
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=spacenetra
      - POSTGRES_USER=spacenetra
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U spacenetra"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - spacenetra-network
    restart: unless-stopped

# ────────────────────────────────────────
# Volumes
# ────────────────────────────────────────
volumes:
  postgres-data:
    driver: local
  qdrant-data:
    driver: local
  tile-data:
    driver: local
  model-checkpoints:
    driver: local

# ────────────────────────────────────────
# Networks
# ────────────────────────────────────────
networks:
  spacenetra-network:
    driver: bridge
```

---

## Building Containers

### Dockerfile (Backend)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src/ ./src/
COPY backend/ ./backend/
COPY models/ ./models/

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dockerfile.model (Model Service)

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# System dependencies for geospatial
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model code
COPY src/ ./src/
COPY models/ ./models/
COPY inference.py .

# Non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

CMD ["python", "-m", "uvicorn", "src.model_server:app", "--host", "0.0.0.0", "--port", "8001"]
```

### Dockerfile (Frontend)

```dockerfile
# Build stage
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

## Offline Deployment

### Preparation Phase (Connected Machine)

```bash
# Step 1: Build all images
docker-compose build

# Step 2: Save images to tarball
docker save \
  spacenetra-frontend \
  spacenetra-backend \
  spacenetra-model-service \
  qdrant/qdrant:latest \
  postgis/postgis:16-3.4 \
  | gzip > spacenetra-images.tar.gz

# Step 3: Package application files
tar -czf spacenetra-app.tar.gz \
  docker-compose.yml \
  .env \
  database/ \
  checkpoints/ \
  configs/

# Step 4: Package satellite data (separately)
tar -czf spacenetra-data.tar.gz data/
```

### Deployment Phase (Air-Gapped Machine)

```bash
# Step 1: Transfer files (USB, secure network, etc.)
# spacenetra-images.tar.gz
# spacenetra-app.tar.gz
# spacenetra-data.tar.gz

# Step 2: Load Docker images
gunzip -c spacenetra-images.tar.gz | docker load

# Step 3: Extract application files
tar -xzf spacenetra-app.tar.gz

# Step 4: Extract data
tar -xzf spacenetra-data.tar.gz

# Step 5: Start services
docker-compose up -d

# Step 6: Verify
docker-compose ps
curl http://localhost/api/health
```

---

## Data Transfer Procedure

### For Updates (Models + Data)

```
CONNECTED ENVIRONMENT                    AIR-GAPPED ENVIRONMENT
─────────────────────                    ─────────────────────

1. Train new model         ──transfer──▶  5. Load new checkpoint
2. Download new scenes     ──transfer──▶  6. Ingest new scenes
3. Package checkpoint      ──────────▶    7. Rebuild vector index
4. Package scenes          ──────────▶    8. Verify integrity

Transfer medium: Encrypted USB / Secure network segment
```

### Integrity Verification

```bash
# Generate checksums before transfer
sha256sum checkpoints/best_model.pth > checksums.txt
sha256sum data/new_scenes/*.tif >> checksums.txt

# Verify after transfer
sha256sum -c checksums.txt
```

---

## Health Checks

### Endpoint: `/api/health`

```json
{
  "status": "healthy",
  "services": {
    "backend": "ok",
    "database": "ok",
    "vector_db": "ok",
    "model_service": "ok"
  },
  "models": {
    "change_detection": {
      "loaded": true,
      "version": "1.0.0",
      "hash": "a3f2b1c..."
    },
    "remote_clip": {
      "loaded": true,
      "version": "1.0.0",
      "hash": "d4e5f6a..."
    }
  },
  "stats": {
    "total_tiles": 125000,
    "total_scenes": 450,
    "total_events": 312,
    "index_size_mb": 256
  }
}
```

### Monitoring Commands

```bash
# Check all containers
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f model-service

# Database status
docker-compose exec postgres pg_isready

# GPU utilization (if applicable)
nvidia-smi
```

---

## Scaling Considerations

### Resource Allocation

| Service | CPU | RAM | GPU | Storage |
|---------|-----|-----|-----|---------|
| Frontend | 1 core | 256 MB | — | 100 MB |
| Backend | 2 cores | 2 GB | — | 1 GB |
| Model Service | 4 cores | 8 GB | 1× GPU (8GB+) | 5 GB |
| Qdrant | 2 cores | 4 GB | — | Based on index |
| PostgreSQL | 2 cores | 4 GB | — | Based on data |

### Performance Targets

| Operation | Target Latency |
|-----------|---------------|
| Semantic search (top-10) | < 100 ms |
| Change detection (1 tile pair) | < 500 ms |
| Timeline query | < 200 ms |
| Dashboard page load | < 2 seconds |
| New scene ingestion | < 10 minutes |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|---------|
| Model service OOM | GPU memory exhausted | Reduce batch size, use CPU fallback |
| Slow search | Large index, no optimization | Rebuild FAISS index with IVF |
| Database connection refused | PostgreSQL not ready | Check health check, increase startup timeout |
| Tiles not loading | Volume mount issue | Verify shared volume paths |
| CORS errors | Frontend config | Check Nginx proxy configuration |
| GPU not detected | NVIDIA runtime missing | Install nvidia-container-toolkit |
