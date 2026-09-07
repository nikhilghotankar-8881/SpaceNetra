# 📡 API Reference — SpaceNetra

> Complete REST API documentation for the SpaceNetra backend

---

## Base URL

```
http://localhost:8000/api
```

---

## Authentication

All endpoints require authentication via JWT token or API key.

```
Authorization: Bearer <jwt_token>
```

or

```
X-API-Key: <api_key>
```

---

## Endpoints

### Health Check

#### `GET /api/health`

Check system health and service status.

**Response:**
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
      "hash": "a3f2b1c4d5e6..."
    },
    "remote_clip": {
      "loaded": true,
      "version": "1.0.0",
      "hash": "d4e5f6a7b8c9..."
    }
  },
  "uptime_seconds": 86400
}
```

---

### Semantic Search

#### `POST /api/search`

Search satellite imagery using natural language or image-based queries.

**Request Body:**
```json
{
  "query": "construction activity near industrial area",
  "top_k": 10,
  "filters": {
    "date_from": "2024-01-01",
    "date_to": "2026-01-01",
    "sensor": "Sentinel-2",
    "max_cloud_pct": 10.0,
    "bbox": {
      "min_lat": 18.5,
      "max_lat": 19.5,
      "min_lon": 72.5,
      "max_lon": 73.5
    }
  }
}
```

**Response:**
```json
{
  "query": "construction activity near industrial area",
  "results": [
    {
      "tile_id": "IND_000451",
      "similarity_score": 0.934,
      "latitude": 19.076,
      "longitude": 72.877,
      "date": "2025-04-12",
      "sensor": "Sentinel-2",
      "cloud_pct": 3.2,
      "thumbnail_url": "/api/imagery/IND_000451/thumbnail",
      "tile_url": "/api/imagery/IND_000451"
    }
  ],
  "total_results": 10,
  "search_time_ms": 45
}
```

---

### Change Detection

#### `POST /api/change-detect`

Run change detection between two tiles or uploaded image pair.

**Request Body:**
```json
{
  "tile_id_t1": "IND_000451_2024",
  "tile_id_t2": "IND_000451_2025",
  "threshold": 0.5,
  "model": "siamese_unet"
}
```

**Response:**
```json
{
  "change_detected": true,
  "change_percentage": 12.4,
  "change_mask_url": "/api/imagery/change_mask/IND_000451_2024_2025",
  "confidence": 0.91,
  "inference_time_ms": 234,
  "model_version": "1.0.0",
  "pixel_stats": {
    "total_pixels": 65536,
    "change_pixels": 8130,
    "no_change_pixels": 57406
  }
}
```

#### `POST /api/change-detect/upload`

Upload custom image pair for change detection.

**Request**: `multipart/form-data`
- `t1_image`: GeoTIFF or PNG file (before)
- `t2_image`: GeoTIFF or PNG file (after)
- `threshold`: float (optional, default 0.5)

**Response**: Same as above.

---

### Imagery

#### `GET /api/imagery/{tile_id}`

Retrieve a specific tile image.

**Response**: Image file (PNG/GeoTIFF)

#### `GET /api/imagery/{tile_id}/thumbnail`

Get a thumbnail version of the tile.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | 256 | Thumbnail width |
| `height` | int | 256 | Thumbnail height |

#### `GET /api/imagery/{tile_id}/metadata`

Get metadata for a tile.

**Response:**
```json
{
  "tile_id": "IND_000451",
  "scene_id": "S2B_MSIL2A_20250412T053629",
  "latitude": 19.076,
  "longitude": 72.877,
  "date": "2025-04-12",
  "sensor": "Sentinel-2",
  "resolution_m": 10.0,
  "cloud_pct": 3.2,
  "quality_score": 94.5,
  "crs": "EPSG:4326",
  "bounds": {
    "min_lat": 19.070,
    "max_lat": 19.082,
    "min_lon": 72.871,
    "max_lon": 72.883
  }
}
```

---

### Timeline

#### `GET /api/timeline/{location}`

Get temporal observations for a location.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `location` | string | Format: `lat,lon` (e.g., `19.076,72.877`) |

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius_km` | float | 1.0 | Search radius |
| `date_from` | string | — | ISO date |
| `date_to` | string | — | ISO date |

**Response:**
```json
{
  "location": {
    "latitude": 19.076,
    "longitude": 72.877
  },
  "timeline": [
    {
      "date": "2022-03-15",
      "tile_id": "IND_000451_2022",
      "status": "stable",
      "confidence": 0.95,
      "thumbnail_url": "/api/imagery/IND_000451_2022/thumbnail"
    },
    {
      "date": "2023-06-20",
      "tile_id": "IND_000451_2023",
      "status": "stable",
      "confidence": 0.93
    },
    {
      "date": "2024-09-10",
      "tile_id": "IND_000451_2024",
      "status": "clearing",
      "confidence": 0.87
    },
    {
      "date": "2025-04-12",
      "tile_id": "IND_000451_2025",
      "status": "construction",
      "confidence": 0.91
    }
  ],
  "change_fingerprint": "stable → stable → clearing → construction",
  "total_observations": 4
}
```

---

### Change Events

#### `GET /api/change-events`

List all detected change events.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `limit` | int | 20 | Results per page |
| `confidence_min` | float | 0.0 | Minimum confidence |
| `confidence_level` | string | — | `LOW`, `MEDIUM`, `HIGH` |
| `change_type` | string | — | Filter by type |
| `bbox` | string | — | Bounding box `min_lat,min_lon,max_lat,max_lon` |
| `date_from` | string | — | First observed after |
| `date_to` | string | — | First observed before |

**Response:**
```json
{
  "events": [
    {
      "id": "evt_021",
      "location": {
        "latitude": 28.612,
        "longitude": 77.209
      },
      "first_observed": "2024-03-15",
      "last_observed": "2026-01-20",
      "change_type": "construction",
      "confidence": 0.93,
      "confidence_level": "HIGH",
      "evidence_count": 5,
      "thumbnail_url": "/api/change-events/evt_021/thumbnail"
    }
  ],
  "total": 312,
  "page": 1,
  "limit": 20
}
```

#### `GET /api/change-events/{event_id}`

Get detailed information about a specific change event.

**Response:**
```json
{
  "id": "evt_021",
  "location": {
    "latitude": 28.612,
    "longitude": 77.209
  },
  "bounds": {
    "min_lat": 28.608,
    "max_lat": 28.616,
    "min_lon": 77.205,
    "max_lon": 77.213
  },
  "first_observed": "2024-03-15",
  "last_observed": "2026-01-20",
  "change_type": "construction",
  "confidence": 0.93,
  "confidence_level": "HIGH",
  "confidence_breakdown": {
    "ai_change_score": 0.91,
    "image_quality": 0.96,
    "alignment_score": 0.94,
    "temporal_persistence": "4/5",
    "cloud_risk": "low"
  },
  "evidence": [
    {
      "t1_tile_id": "IND_002341_2024",
      "t2_tile_id": "IND_002341_2025",
      "change_pct": 14.2,
      "date_t1": "2024-03-15",
      "date_t2": "2025-04-12"
    }
  ],
  "similar_sites": [
    {
      "event_id": "evt_045",
      "similarity": 0.87,
      "location": { "latitude": 28.701, "longitude": 77.156 }
    }
  ]
}
```

---

### Provenance

#### `GET /api/provenance/{event_id}`

Get full provenance chain for a change event.

**Response:**
```json
{
  "event_id": "evt_021",
  "provenance": {
    "source_scenes": [
      {
        "scene_id": "S2B_MSIL2A_20240315T053629",
        "sensor": "Sentinel-2B",
        "acquisition_date": "2024-03-15T05:36:29Z",
        "cloud_cover_pct": 4.2
      },
      {
        "scene_id": "S2A_MSIL2A_20250412T053629",
        "sensor": "Sentinel-2A",
        "acquisition_date": "2025-04-12T05:36:29Z",
        "cloud_cover_pct": 3.1
      }
    ],
    "processing": {
      "preprocessing_version": "1.2.0",
      "cloud_masking": "SCL-based",
      "registration_method": "SIFT + affine",
      "registration_rmse": 0.34,
      "normalization": "histogram_matching"
    },
    "model": {
      "architecture": "siamese_unet",
      "version": "1.0.0",
      "checkpoint": "best_model_epoch_47.pth",
      "hash": "sha256:a3f2b1c4d5e6f7a8b9c0d1e2f3a4b5c6",
      "training_dataset": "LEVIR-CD + Indian-Sentinel-v1",
      "threshold": 0.5
    },
    "confidence": {
      "method": "multi-signal aggregation",
      "version": "1.0.0",
      "signals": ["ai_score", "quality", "alignment", "temporal", "cloud"]
    },
    "timestamp": "2026-01-20T14:32:15Z"
  }
}
```

---

### Feedback

#### `POST /api/feedback`

Submit analyst feedback on a change event.

**Request Body:**
```json
{
  "event_id": "evt_021",
  "decision": "ACCEPT",
  "notes": "Confirmed construction activity visible in high-res imagery.",
  "analyst_id": "analyst_001"
}
```

**Response:**
```json
{
  "feedback_id": "fb_001234",
  "event_id": "evt_021",
  "decision": "ACCEPT",
  "timestamp": "2026-01-21T10:15:30Z",
  "status": "recorded"
}
```

**Valid decisions:** `ACCEPT`, `REJECT`, `NEED_REVIEW`

#### `GET /api/feedback/stats`

Get feedback statistics.

**Response:**
```json
{
  "total_feedback": 450,
  "accepted": 312,
  "rejected": 89,
  "needs_review": 49,
  "acceptance_rate": 0.693
}
```

---

### Similar Sites

#### `POST /api/similar-sites`

Find locations with similar visual/temporal patterns.

**Request Body:**
```json
{
  "reference_tile_id": "IND_000451",
  "top_k": 5,
  "filters": {
    "min_similarity": 0.7,
    "bbox": {
      "min_lat": 18.0,
      "max_lat": 22.0,
      "min_lon": 72.0,
      "max_lon": 78.0
    }
  }
}
```

**Response:**
```json
{
  "reference_tile_id": "IND_000451",
  "similar_sites": [
    {
      "tile_id": "IND_003782",
      "similarity": 0.912,
      "latitude": 19.234,
      "longitude": 73.012,
      "date": "2025-05-20",
      "thumbnail_url": "/api/imagery/IND_003782/thumbnail"
    }
  ],
  "search_time_ms": 32
}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "TILE_NOT_FOUND",
    "message": "Tile with ID 'IND_999999' not found.",
    "status": 404
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `TILE_NOT_FOUND` | 404 | Requested tile doesn't exist |
| `EVENT_NOT_FOUND` | 404 | Requested event doesn't exist |
| `INVALID_QUERY` | 400 | Malformed search query |
| `INVALID_BBOX` | 400 | Invalid bounding box |
| `MODEL_NOT_LOADED` | 503 | AI model not available |
| `VECTOR_DB_ERROR` | 503 | Vector database unavailable |
| `DB_ERROR` | 503 | PostgreSQL unavailable |
| `UNAUTHORIZED` | 401 | Missing or invalid auth |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| Search | 60 requests/minute |
| Change Detection | 20 requests/minute |
| Imagery | 120 requests/minute |
| Other | 60 requests/minute |
