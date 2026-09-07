# 📡 Data Pipeline — SpaceNetra

> Data ingestion, preprocessing, and preparation for AI inference

---

## Table of Contents

- [Overview](#overview)
- [Data Sources](#data-sources)
- [LEVIR-CD Benchmark Dataset](#levir-cd-benchmark-dataset)
- [Satellite Data Ingestion](#satellite-data-ingestion)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Tiling Strategy](#tiling-strategy)
- [Data Augmentation](#data-augmentation)
- [Quality Control](#quality-control)
- [Incremental Ingestion](#incremental-ingestion)

---

## Overview

The data pipeline handles two distinct data flows:

1. **Benchmark Data (LEVIR-CD)** — For initial model training and evaluation
2. **Operational Satellite Data (Sentinel-2)** — For real-world Indian imagery

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                            │
│                                                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐   │
│  │  BENCHMARK  │     │  SATELLITE  │     │   OUTPUT        │   │
│  │  (LEVIR-CD) │     │  (Sentinel) │     │                 │   │
│  │             │     │             │     │ • PyTorch Tensors│   │
│  │ • RGB pairs │     │ • GeoTIFF   │     │ • 256×256 tiles  │   │
│  │ • Masks     │     │ • 13 bands  │     │ • Metadata       │   │
│  │ • 1024×1024 │     │ • 10-60m    │     │ • Embeddings     │   │
│  └──────┬──────┘     └──────┬──────┘     └─────────────────┘   │
│         │                   │                      ▲             │
│         ▼                   ▼                      │             │
│  ┌──────────────────────────────────────┐          │             │
│  │          PREPROCESSING               │──────────┘             │
│  │ Cloud → Quality → Normalize → Tile   │                       │
│  └──────────────────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Benchmark: LEVIR-CD

| Property | Value |
|----------|-------|
| **Name** | LEVIR-CD (Large-scale Earth Visual Intelligence & Recognition Change Detection) |
| **Image Pairs** | 637 |
| **Resolution** | ~0.5 m/pixel |
| **Image Size** | 1024 × 1024 pixels |
| **Bands** | RGB (3 channels) |
| **Label Type** | Binary mask (0 = no change, 255 = change) |
| **Change Type** | Building construction/demolition |
| **Location** | Texas, USA |
| **Time Span** | 5–14 years between T1 and T2 |
| **Split** | Train: 445, Val: 64, Test: 128 |

### Operational: Sentinel-2

| Property | Value |
|----------|-------|
| **Operator** | European Space Agency (ESA) |
| **Revisit** | ~5 days (twin satellites) |
| **Swath** | 290 km |
| **Bands** | 13 spectral bands |
| **Coverage** | Global (including India) |
| **Cost** | Free and open |
| **Format** | GeoTIFF / JPEG2000 |

**Sentinel-2 Band Configuration:**

| Band | Name | Resolution | Use |
|------|------|-----------|-----|
| B01 | Coastal aerosol | 60 m | Atmospheric correction |
| B02 | Blue | 10 m | RGB composite, water |
| B03 | Green | 10 m | RGB composite, vegetation |
| B04 | Red | 10 m | RGB composite, soil |
| B05 | Red Edge 1 | 20 m | Vegetation classification |
| B06 | Red Edge 2 | 20 m | Vegetation classification |
| B07 | Red Edge 3 | 20 m | Vegetation classification |
| B08 | NIR | 10 m | Vegetation, change detection |
| B08A | Red Edge 4 | 20 m | Vegetation, water vapor |
| B09 | Water vapor | 60 m | Atmospheric correction |
| B10 | SWIR - Cirrus | 60 m | Cirrus cloud detection |
| B11 | SWIR 1 | 20 m | Snow/ice, moisture |
| B12 | SWIR 2 | 20 m | Snow/ice, moisture |

**Priority bands for change detection (10 m):** B02, B03, B04, B08

---

## LEVIR-CD Benchmark Dataset

### Directory Structure

```
data/levir_cd/
├── train/
│   ├── A/          # T1 images (old)
│   │   ├── train_1.png
│   │   ├── train_2.png
│   │   └── ...
│   ├── B/          # T2 images (new)
│   │   ├── train_1.png
│   │   ├── train_2.png
│   │   └── ...
│   └── label/      # Ground truth masks
│       ├── train_1.png
│       ├── train_2.png
│       └── ...
│
├── val/
│   ├── A/
│   ├── B/
│   └── label/
│
└── test/
    ├── A/
    ├── B/
    └── label/
```

### Data Loading Flow

```
A/train_001.png  ──┐
                   ├──→  (T1, T2, Mask)  ──→  Augment  ──→  Tensor  ──→  Model
B/train_001.png  ──┤
                   │
label/train_001.png┘
```

### Validation Checks

Before training, verify every sample:

```python
# Pseudo-code for data validation
for filename in all_filenames:
    t1 = load_image(f"A/{filename}")       # Must exist
    t2 = load_image(f"B/{filename}")       # Must exist
    mask = load_image(f"label/{filename}") # Must exist
    
    assert t1.shape == t2.shape == (1024, 1024, 3)
    assert mask.shape == (1024, 1024)
    assert set(np.unique(mask)).issubset({0, 255})
    assert not is_corrupted(t1)
    assert not is_corrupted(t2)
```

---

## Satellite Data Ingestion

### Sentinel-2 Download Pipeline

```
┌──────────────────┐
│ Define AOI       │  Area of Interest (GeoJSON polygon)
│ (Lat/Lon bounds) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Query Catalog    │  Copernicus Data Space / STAC API
│ Date range       │
│ Cloud cover max  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Filter Results   │  Sort by cloud cover, quality
│ Select best      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Download Bands   │  B02, B03, B04, B08 (10m priority)
│ GeoTIFF format   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Register in DB   │  Scene ID, bounds, date, sensor, path
└──────────────────┘
```

### Supported Formats

| Format | Extension | Description |
|--------|----------|-------------|
| GeoTIFF | `.tif` / `.tiff` | Standard geospatial raster |
| Cloud Optimized GeoTIFF (COG) | `.tif` | Optimized for cloud/streaming access |
| JPEG2000 | `.jp2` | Sentinel-2 native format |

---

## Preprocessing Pipeline

### Full Pipeline

```
Raw Satellite Scene
       │
       ▼
┌──────────────────────────────────────────────┐
│ STEP 1: Cloud Detection                       │
│                                              │
│ Input:  Raw scene + SCL band (Sentinel-2)    │
│ Method: Scene Classification Layer classes   │
│ Output: Binary cloud mask                    │
│                                              │
│ SCL Classes:                                 │
│   0  = No data           6 = Water           │
│   1  = Saturated/defect  7 = Unclassified    │
│   2  = Dark/shadow       8 = Cloud (medium)  │
│   3  = Cloud shadow      9 = Cloud (high)    │
│   4  = Vegetation       10 = Cirrus (thin)   │
│   5  = Bare soil        11 = Snow/ice        │
│                                              │
│ Cloud = SCL ∈ {8, 9, 10}                     │
│ Shadow = SCL ∈ {2, 3}                        │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 2: Quality Assessment                    │
│                                              │
│ Metrics:                                     │
│   • Cloud cover percentage                   │
│   • Usable area percentage                   │
│   • Haze/atmospheric quality                 │
│   • Noise level estimation                   │
│                                              │
│ Decision:                                    │
│   cloud_pct > 20% → REJECT scene             │
│   cloud_pct > 10% → FLAG for review          │
│   cloud_pct < 10% → ACCEPT                   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 3: Radiometric Normalization             │
│                                              │
│ Raw DN values → calibrated reflectance       │
│                                              │
│ For Sentinel-2:                              │
│   Reflectance = DN / 10000                   │
│                                              │
│ Cross-temporal normalization:                │
│   Histogram matching between T1 and T2       │
│   Ensures consistent brightness/contrast     │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 4: Geometric Registration                │
│                                              │
│ Goal: Sub-pixel alignment between T1 and T2  │
│                                              │
│ Method:                                      │
│   1. Feature matching (SIFT/ORB)             │
│   2. Affine or projective transform          │
│   3. Resample T2 to T1 reference frame       │
│   4. Calculate registration quality score    │
│                                              │
│ Quality check:                               │
│   RMSE < 0.5 pixels → GOOD                  │
│   RMSE 0.5-1.0       → ACCEPTABLE           │
│   RMSE > 1.0 pixels → REJECT pair           │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 5: AOI Extraction & Resampling           │
│                                              │
│ Clip to Area of Interest                     │
│ Resample all bands to target resolution      │
│ (e.g., 20m bands → 10m via bicubic)          │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│ STEP 6: Tiling                                │
│                                              │
│ Full scene → 256×256 pixel patches           │
│ With configurable overlap/stride             │
│ Preserve georeference per tile               │
│ Store tile metadata in database              │
└──────────────────────────────────────────────┘
```

---

## Tiling Strategy

### Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Tile size | 256 × 256 px | Compatible with most segmentation architectures |
| Stride | 256 (no overlap) for training | Can use overlap for inference |
| Overlap | 0-64 px for inference | Reduces edge artifacts |
| CRS preservation | Yes | Each tile retains georeference |

### Visualization

```
Full Scene (e.g., 10240 × 10240 pixels at 10m = 102.4 × 102.4 km)
┌────┬────┬────┬────┬────┬────┬────┬────┬ ─ ─ ┐
│ 01 │ 02 │ 03 │ 04 │ 05 │ 06 │ 07 │ 08 │     │
├────┼────┼────┼────┼────┼────┼────┼────┼ ─ ─ ┤
│ 09 │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │ 16 │     │
├────┼────┼────┼────┼────┼────┼────┼────┼ ─ ─ ┤
│    │    │    │    │    │    │    │    │     │
│    │         ...                │    │     │
│    │    │    │    │    │    │    │    │     │
└────┴────┴────┴────┴────┴────┴────┴────┴ ─ ─ ┘

Each tile: 256 × 256 pixels
40 × 40 = 1600 tiles per scene
```

### LEVIR-CD Tiling

```
1024 × 1024 → 256 × 256 patches
= 4 × 4 = 16 patches per image pair
= 637 pairs × 16 patches = 10,192 training patches
```

---

## Data Augmentation

### Strategy

Augmentations are applied **identically** to T1, T2, and mask (same random seed per sample).

### Augmentation Pipeline

```python
# Pseudo-code
augmentations = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation90(p=0.5),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    # NO elastic deform — physically meaningless for EO
    # NO extreme color shifts — breaks spectral properties
])

# Applied identically to all three:
t1_aug, t2_aug, mask_aug = augmentations(t1, t2, mask)
```

### Allowed vs Forbidden Augmentations

| ✅ Allowed | ❌ Forbidden | Reason |
|-----------|-------------|--------|
| Horizontal flip | Elastic deformation | Physically meaningless |
| Vertical flip | Heavy color shift | Breaks spectral properties |
| 90° rotation | CutMix/MixUp | Destroys spatial relationships |
| Brightness ±10% | Random crop (alone) | May crop out change areas |
| Contrast ±10% | Gaussian blur (heavy) | Destroys important detail |
| Gaussian noise (light) | Style transfer | Breaks physical meaning |

> ⚠️ **Key Principle**: Remote sensing augmentations must be physically plausible. A flipped satellite image is valid. A color-inverted one is not.

---

## Quality Control

### Automated Checks

```
┌─────────────────────────────────────────────────┐
│              QUALITY CONTROL GATES               │
│                                                 │
│  Gate 1: File Integrity                         │
│  ├── File exists and is readable                │
│  ├── Correct dimensions                         │
│  ├── Correct number of bands                    │
│  └── No corruption (can decode fully)           │
│                                                 │
│  Gate 2: Content Validity                       │
│  ├── Not all-black / all-white                  │
│  ├── Reasonable value range                     │
│  ├── Mask contains both 0 and 255               │
│  └── Change area > minimum threshold            │
│                                                 │
│  Gate 3: Pair Consistency                       │
│  ├── T1 and T2 dimensions match                 │
│  ├── T1 and T2 CRS match (satellite data)       │
│  ├── T1 and T2 bounds overlap (satellite data)   │
│  └── Registration quality above threshold       │
│                                                 │
│  Gate 4: Scene Quality (satellite only)         │
│  ├── Cloud cover below threshold                │
│  ├── Haze/atmospheric quality acceptable        │
│  └── No sensor artifacts                        │
└─────────────────────────────────────────────────┘
```

---

## Incremental Ingestion

### Design Principle

When new satellite scenes arrive, **do not rebuild the entire archive**.

### Process

```
NEW SCENE ARRIVES
       │
       ▼
┌──────────────────┐
│ 1. Duplicate check│  Has this scene been processed?
└────────┬─────────┘
         │ (New scene)
         ▼
┌──────────────────┐
│ 2. Quality check  │  Cloud, quality, usability
└────────┬─────────┘
         │ (Passed)
         ▼
┌──────────────────┐
│ 3. Preprocess     │  Cloud mask, normalize, register
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Tile           │  256×256 patches with metadata
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. Embed          │  RemoteCLIP → 512-d vector per tile
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. Index          │  ADD to FAISS/Qdrant (no rebuild)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 7. Metadata       │  INSERT into PostGIS database
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 8. Temporal check │  Compare with existing tiles at same location
└──────────────────┘    → Run change detection if T1 exists
```

### Scalability Targets

| Metric | Target |
|--------|--------|
| New scene processing | < 10 minutes |
| Vector index addition | < 1 second per tile |
| Metadata insertion | < 100 ms per tile |
| No downtime during ingestion | Required |
