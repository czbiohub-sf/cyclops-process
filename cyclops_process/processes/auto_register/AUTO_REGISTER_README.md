# Automatic Registration Algorithm

## Overview

Automatically registers ISS→Track and Pheno→Track using **PCC pre-alignment** → **Graph-based neighborhood matching** → **RANSAC affine estimation**.

## Algorithm Pipeline

### 1. PCC Pre-alignment (~50s)
- Downsample segmentation masks (8-32x depending on modality)
- Phase cross-correlation for coarse translation estimate
- **ISS→Track**: Uses full well (1.0) + 32x downsampling (sparse signal)
- **Pheno→Track**: Uses center 30% + 1x downsampling (dense signal)

### 2. Extract & Subsample Centroids (~30s)
- Extract cell centroids from segmentation masks
- Spatial grid subsampling: Select 100 random bins from 50×50 grid
- Typical: 450k cells → 25k cells (preserves spatial distribution)
- Filter by minimum area (100 pixels) to remove debris

### 3. Graph-Based Neighborhood Matching (~100-160s)
**Multi-stage filtering pipeline:**

**Stage 1: Wide spatial search**
- Find all cells within 600px radius (handles poor PCC alignment)

**Stage 2: Hu moment filtering**
- Score all candidates by shape similarity (7D Hu moments)
- Keep top 100-200 candidates per cell

**Stage 3: Graph consistency validation**
- Build k-NN graphs (k=8) for source and target cells
- Score each candidate using weighted components:
  - Individual Hu similarity (30%)
  - Neighbor Hu consistency (40%)
  - Edge length consistency (20%)
  - Edge angle consistency (10%)

**Stage 4: Hard quality threshold**
- Reject matches with score > 0.10 (very strict)
- Allow cells with no match if quality insufficient
- Fallback to top 10 matches globally if too few pass

**Stage 5: Greedy 1-to-1 matching**
- Enforce unique source→target correspondence
- Sort by score, greedily assign best matches first

**Result**: 30-100 high-quality matched cell pairs

### 4. RANSAC Affine Estimation (~0.5s)
- Robust affine fitting despite outliers
- Min 3 points, residual threshold 8px
- Typical inlier ratio: 10-20% (graph matching is very strict)
- Final accuracy: <5px compared to manual alignment

### 5. Validation & Output
**Saved files:**
- `A{well}_auto_{source}_register.yml` - 4×4 affine matrix (biahub format)
- `auto_register_metrics.csv` - Comprehensive metrics (timing, RANSAC, overlap, graph scores)
- Visualization overlays (PCC comparison, final alignment, graph matches)

**Quality check:**
- Overlap metric: >20% required (error if below)
- Visual inspection: Yellow overlap in aligned regions
- Comparison with manual (if available): <10px translation diff

## Key Parameters

```python
# Graph matching (strict quality over quantity)
"max_match_distance": 600.0          # Wide search radius (handles poor PCC)
"graph_top_k_candidates": 100-200    # Hu-filtered candidates per cell
"graph_k_neighbors": 8               # Neighborhood size
"graph_max_score_threshold": 0.10    # Hard quality gate (reject if >0.10)
"graph_min_total_matches": 10        # Minimum matches (fallback)

# RANSAC
"min_samples": 3                     # Minimum for 2D affine
"residual_threshold": 8.0            # Inlier threshold (pixels)
"max_trials": 50000                  # More trials for low inlier ratios

# Subsampling
"spatial_grid_size": 50              # 50×50 grid
"spatial_bins_to_select": 100        # Select 100 bins (~5% of cells)
```

## Usage

```bash
# ISS→Track registration
python -m cyclops_process.processes.auto_register iss-to-track \
    --experiment ops0031_20250424 --well 1

# Pheno→Track registration
python -m cyclops_process.processes.auto_register pheno-to-track \
    --experiment ops0031_20250424 --well 2

# All wells
python -m cyclops_process.processes.auto_register all \
    --experiment ops0031_20250424

# Debug mode (faster, central 30%)
python -m cyclops_process.processes.auto_register pheno-to-track \
    --experiment ops0031_20250424 --well 1 --center-fraction 0.3
```

## Performance

**Full well (~25k subsampled cells):**
- PCC: ~50s
- Centroid extraction: ~30s
- Graph matching: ~100-160s (most time here)
- RANSAC: ~0.5s
- Overlays: ~80s
- **Total: ~5 minutes**

**Debug mode (30% center):**
- Total: ~1-2 minutes

## Why Graph-Based Matching?

**Problem with simple k-NN Hu matching:**
- Only searches within 50px (fails when PCC has large errors)
- No neighborhood validation (high false positive rate)
- Low RANSAC inlier ratios (~3%)

**Graph-based solution:**
- 600px search radius (12× wider, handles PCC errors)
- Validates neighborhood structure consistency
- Multi-component scoring (shape + neighbors + edges)
- Hard quality thresholds (reject instead of force)
- High precision: 10-20% RANSAC inlier ratio with strict filtering

**Result**: <5px accuracy vs manual, robust to poor PCC alignment

## Validation

Check `auto_register_metrics.csv` for:
- `overlap_forward`: Should be >20% (error if below)
- `ransac_inlier_ratio`: Should be >10%
- `manual_translation_diff_px`: Should be <10px (if manual available)
- `graph_matches`: Number of high-quality matches found
- `time_total_s`: Total processing time

Check visualizations:
- `03_final_alignment_8x.png` - Full well overlay
- `00d_graph_matched_cells_100.png` - Shows matched cell pairs with graph edges
- `04_final_detail_grid_pos*.png` - Detailed views at 3 positions

## Cache Files

All cache files now have human-readable names:

**Centroids:** `track5x_A1_t0_cf1p0_a1b2c3d4.npy`
**PCC:** `pcc_pheno20x_to_track5x_A1_t0_ds8_cf0p3_e5f6g7h8.json`
**Hu moments:** `hu_moments_track5x_A1_t0_n27534_12345678.npy`

Format: `{modality}_{well}_{timepoint}_{params}_{hash8}.ext`

Cache location: `{experiment}/2-tracking/{cache_type}_cache/`
