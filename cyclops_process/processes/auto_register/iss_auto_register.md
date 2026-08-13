I need your help making a clear plan for fixing a new registration issue we noticed in our in situ sequencing data in which there are appears to be some ~5-20pixel drift from round to round of imaging our 10 rounds... we originally fixed this with a PCC method in @cyclops_process/processes/register.py correct cycle drift but that is proving to be not sufficient for a bunch of expeirments with more drift... we came up with a really nice registration method of nuclear shape between microscopes in @cyclops_process/processes/auto_register/auto_register.py and now id like to adapt this for our ISS data... note we don't have segmentations of our iss spots and we def will want to sum our channels which garuntees there will always be signal to match for our spots between rounds within the same cell... also round 1 is a bit more complicated in that it contains a zero channel nucleus which we shoule consider its own seperate 'round -1' that should be aligned indepdently to round 0 in addition to aligning all the other 10 rounds (0-9)... come up with a clear plan fo rme to review and incldue all details and ask any questions of informaiton you are missing

# Plan: Robust ISS Round-to-Round Drift Correction Using Spot-Based Registration

## Problem Summary
- Current PCC method (`correct_cycle_drift`) samples 5 tiles and applies mean drift to all tiles
- Insufficient for experiments with 5-20 pixel drift across 10 rounds
- Need adaptation of successful auto-registration approach from nuclear segmentation matching
- Special case: Round 0 contains nucleus channel (ch0) that should be treated as separate "Round -1"

## Questions Before Implementation

1. **ISS Data Structure**: 
   - Current shape appears to be `(rounds, channels, z, y, x)` from line 179 in register.py
   - How many channels per round? (excluding Round 0's nucleus channel) there are 5 channels but channel 0 is blank expecpt for round 0 so we just sum channels 1-4 in all rounds to get the spot intensities.
   - Is nucleus channel always channel 0 in Round 0? yes its the first round's channel 0

2. **Round 0 (Nucleus) Alignment**:
   - Should Round -1 (nucleus) be aligned to Round 0 spots, or vice versa? align to the spots
   - Should we use the nucleus segmentation if available for this step, or just the raw intensity? since we are aligning to spot the best thing to is get nucleur centroids and then use graph-matching with it neighbors to find its corrsponding 'spot' of that nucleus in round 0 data.

3. **Spot Detection Strategy**:
   - Should we detect spots per-channel then aggregate, or sum channels first then detect? - sum first then detect.
   - Any minimum spot intensity threshold preferences? - 400 is decent threshold but bimodal binning may be more robust... we don't need to detect every spot here just the really robust high signal ones to match.

4. **Output Requirements**:
   - Keep existing output structure (`iss_drift_corrected.zarr`)? def not... we will work in stitch space... see @ops_stitch.py and save a new zarr path in @experiment.py for iss_stitch_registered.py for @iss.py to call.
   - Should we output both Round -1 (nucleus) and Rounds 0-9 in same zarr, or separate? yes place round -1 nucleus back into the 0 channel of round 0 after registration

5. **Per-Tile vs Mean Drift**:
   - You mentioned current method applies mean drift - should new method apply per-tile transforms? we will work in stitch space like auto_register is set up to do and comput a full affine of the well for each round + nucleus.

## Proposed Architecture

### New Module: `cyclops_process/processes/iss_cycle_registration.py`

Similar structure to `auto_register.py` but adapted for spot-based matching:

```
iss_cycle_registration.py (main orchestration)
├── iss_cycle_pcc.py (PCC on summed spot channels)
├── iss_cycle_spots.py (spot detection & centroid extraction)
├── iss_cycle_ransac.py (reuse from auto_register)
└── iss_cycle_visualization.py (QA overlays per round)
```

### Core Algorithm (Per Well, Per Tile)

**Phase 1: Round 0 (Nucleus) → Round 1 Registration**
1. Extract nucleus channel from Round 1
2. Sum all spot channels from Round 1 → reference spots
3. PCC pre-alignment (if nucleus and spots have different distributions)
4. Detect spots in both as "pseudo-centroids"
5. KDTree matching + RANSAC affine
6. Output: `round0_to_round1_affine.yml`

**Phase 2: Round-to-Round Sequential Registration (Rounds 1→10)**
1. For each round R (2-10):
   - Sum all channels in round R-1 (reference)
   - Sum all channels in round R (moving)
   - PCC pre-alignment (coarse)
   - Detect spots in both rounds
   - KDTree matching on spot centroids
   - RANSAC affine estimation (similarity or euclidean transform)
   - Compose with previous rounds: `T_final[R] = T[R→R-1] ∘ T[R-1→R-2] ∘ ... ∘ T[2→1]`
2. Output: Per-round affine transforms + visualization overlays

**Phase 3: Apply Transforms & Write Output**
1. Apply composed affines to each round's raw data
2. Write to `iss_drift_corrected.zarr` with same structure
3. Generate drift trajectory plots (like current `plot_drift`)

## Implementation Plan

### Step 1: Core Spot Detection Module
**File**: `cyclops_process/processes/iss_cycle_spots.py`
- `detect_spots_in_round()` - sum channels, local maxima detection
- `extract_spot_centroids()` - similar to `extract_centroids_from_segmentation` but for spot peaks
- Caching mechanism (like centroid cache)

### Step 2: PCC Adaptation  
**File**: `cyclops_process/processes/iss_cycle_pcc.py`
- Reuse `estimate_translation_pcc()` but adapted for intensity images (not binary masks)
- Handle summed spot channels (continuous intensities)
- Cache results per (well, round_pair)

### Step 3: Main Registration Function
**File**: `cyclops_process/processes/iss_cycle_registration.py`
- `register_iss_round_to_round()` - main entry point
  - Handle Round 0 special case
  - Sequential round-to-round registration
  - Compose affine transforms
  - Apply transforms and write output
- Reuse RANSAC from `auto_register_ransac.py` (already robust)

### Step 4: Visualization & QA
**File**: `cyclops_process/processes/iss_cycle_visualization.py`
- Per-round alignment overlays (before/after)
- Drift trajectory plot (cumulative drift across rounds)
- Spot matching quality metrics (inlier ratios, residuals)

### Step 5: Integration & Testing
- Add CLI interface (similar to `auto_register.py` main)
- Update `OpsDataset` store paths if needed
- Test on experiments with known high drift
- Compare with current `correct_cycle_drift` outputs

## Key Design Decisions

1. **Reuse Proven Components**:
   - PCC module structure from `auto_register_pcc.py`
   - KDTree matching from `auto_register_ransac.py`
   - RANSAC affine estimation (already handles low inlier ratios)
   - Caching strategy for speed

2. **Spot Detection Approach**:
   - Sum all channels first (ensures signal in every round)
   - Use local maxima detection with minimum intensity threshold
   - Generate "pseudo-centroids" for RANSAC matching

3. **Transform Type**:
   - Start with `similarity` (isotropic scale + rotation + translation)
   - Can downgrade to `euclidean` (rotation + translation) if scale drift is minimal

4. **Validation Strategy**:
   - Automatic overlay generation per round-pair
   - Metrics CSV with inlier ratios, residuals, drift magnitude
   - Side-by-side comparison with current PCC method

5. **Backward Compatibility**:
   - Keep same output path (`iss_drift_corrected.zarr`)
   - Maintain same CLI interface in `register.py` (add `method` parameter)
   - Option to fall back to old PCC method

## Success Criteria

1. **Registration Quality**:
   - RANSAC inlier ratio > 30% per round-pair
   - Median residual < 2 pixels after registration
   - Maximum cumulative drift < 5 pixels across all 10 rounds

2. **Performance**:
   - < 5 minutes per well (similar to current method)
   - Efficient caching (spot detection + PCC)
   - Parallel tile processing where possible

3. **Validation**:
   - Visual overlays show good spot alignment across rounds
   - Drift plots show clear correction vs. uncorrected data
   - Downstream spot calling improves (fewer missed spots due to drift)

## Next Steps After Approval

1. Implement spot detection module with caching
2. Adapt PCC for continuous intensity images
3. Build main registration loop with Round 0 handling
4. Add visualization outputs
5. Test on high-drift experiment
6. Integrate with existing pipeline

**Estimated Implementation Time**: 2-3 days for core functionality + testing



...



# Revised Plan: ISS Round-to-Round Registration in Stitch Space

## Problem Summary (Updated)
- Current PCC method insufficient for 5-20px drift across 10 ISS rounds
- Working in **stitched well space** (not tiles) for full-well affine transforms
- Special handling: Round 0 has nucleus in ch0; Rounds 0-9 have spots in ch1-4
- Need to register Round -1 (nucleus, ch0 of Round 0) separately to spots of Round 0
- Output: `iss_stitch_registered.zarr` (new store path)

## Clarifications from User Responses

1. **Data Structure**: 
   - Shape: `(rounds, channels, z, y, x)` 
   - 5 channels total: ch0 = nucleus (Round 0 only), ch1-4 = spot channels (all rounds)
   - Sum channels 1-4 for spot intensities

2. **Round -1 (Nucleus) Strategy**:
   - Extract nucleus centroids from ISS segmentation (if exists) 
   - Use graph-matching with neighbors to find corresponding "spot" in Round 0 summed data
   - Align nucleus TO spots (not vice versa)

3. **Spot Detection**:
   - Sum channels 1-4 first, then detect peaks
   - Threshold: 400 (or bimodal binning for robustness)
   - Only need robust high-signal spots for matching

4. **Output Structure**:
   - New zarr: `iss_stitch_registered.zarr` 
   - Place registered nucleus back into ch0 of Round 0
   - Keep Rounds 0-9 with all channels

5. **Work in Stitch Space**:
   - Similar to `auto_register.py` (works on stitched wells)
   - Compute one affine per well per round
   - Input: `bc_stitched.zarr` (from existing stitch)
   - Output: `bc_stitched_registered.zarr`

## Revised Architecture

### Module Structure
```
cyclops_process/processes/auto_register/
├── iss_cycle_register.py (main entry point)
├── iss_cycle_spots.py (spot detection & centroid extraction)  
├── iss_cycle_ransac.py (reuse auto_register_ransac.py)
├── iss_cycle_pcc.py (PCC for intensity images)
└── iss_cycle_visualization.py (per-round QA overlays)
```

### Core Algorithm (Per Well)

#### Phase 1: Round -1 (Nucleus) → Round 0 (Spots) Registration

**Input**: 
- Round 0, ch0 (nucleus raw image)
- Round 0, ch1-4 summed (spot intensities)
- ISS segmentation (for nucleus centroids) - optional

**Steps**:
1. **Extract nucleus centroids**:
   - If `iss_segmentation` exists: extract centroids from segmentation
   - Otherwise: detect nucleus peaks from raw ch0 intensity
   
2. **Extract "pseudo-spot" centroids from Round 0**:
   - Sum channels 1-4 → spot image
   - Detect local maxima (threshold=400 or bimodal)
   - Extract (y, x) coordinates as centroids

3. **Graph-based matching**:
   - Build k-NN graphs for both nucleus and spot centroids
   - Use same graph-consistency matching as auto_register (Hu moments not needed)
   - Match nucleus → nearest spot cluster using spatial + neighborhood structure

4. **RANSAC affine estimation**:
   - Use matched pairs (nucleus centroids → spot centroids)
   - Estimate similarity/euclidean transform
   - Output: `nucleus_to_round0_affine.yml`

**Output**: Affine transform to register nucleus image to Round 0 coordinate system

---

#### Phase 2: Round-to-Round Sequential Registration (Rounds 0→1, 1→2, ..., 8→9)

**For each consecutive pair (R[i], R[i+1])**:

1. **Load and sum spot channels**:
   - Reference: Sum ch1-4 from Round i
   - Moving: Sum ch1-4 from Round i+1

2. **PCC coarse alignment** (optional but recommended):
   - Estimate translation-only shift using phase cross-correlation
   - Work on intensity images (not binary masks)
   - Downsample 16-32x for speed
   - Cache results

3. **Spot detection**:
   - Detect local maxima in reference (Round i summed)
   - Detect local maxima in moving (Round i+1 summed)
   - Apply PCC shift to moving centroids

4. **KDTree matching**:
   - Match spots from moving → reference using spatial distance
   - Filter by max distance (e.g., 600px search radius)

5. **RANSAC affine estimation**:
   - Use matched spot pairs
   - Estimate similarity or euclidean transform
   - Validate with inlier ratio > 30%, median residual < 2px

6. **Compose transforms**:
   - `T_cumulative[i+1] = T[i+1→i] ∘ T_cumulative[i]`
   - All rounds ultimately registered to Round 0 coordinate system

**Output**: 
- Per-round affines: `round{i}_to_round{i-1}_affine.yml`
- Cumulative affines: `round{i}_to_round0_affine.yml`

---

#### Phase 3: Apply Transforms & Write Registered Output

**Input**: `bc_stitched.zarr` (T, C=5, Z=1, Y, X) per well

**Steps**:
1. **Create output store**: `bc_stitched_registered.zarr`
   - Same shape as input
   - Layout: HCS with same wells/positions

2. **Apply affine transforms**:
   - For Round -1 (nucleus):
     - Load ch0 from Round 0
     - Apply `nucleus_to_round0_affine`
     - Write back to ch0 of Round 0
   
   - For each Round i (0-9):
     - Load all channels (1-4, and possibly 0)
     - Apply `round{i}_to_round0_affine`
     - Use scipy/cupy affine_transform with order=1 (linear interpolation)
     - Write to output[round_i, :, :, :, :]

3. **Handle edge cases**:
   - Padding if transforms shift content outside original bounds
   - Or crop to keep same output size (user preference?)

**Output**: `bc_stitched_registered.zarr` with all rounds aligned to Round 0

---

## Implementation Plan

### Step 1: Spot Detection Module
**File**: `cyclops_process/processes/auto_register/iss_cycle_spots.py`

**Functions**:
- `detect_spots_intensity(image, threshold, min_distance)` 
  - Local maxima detection on continuous intensity
  - Return (y, x) coordinates
  
- `extract_spot_centroids_from_round(zarr_path, position, round_idx, threshold)`
  - Load Round data, sum ch1-4, detect spots
  - Cache results (similar to centroid cache in auto_register)

- `extract_nucleus_centroids(seg_path, zarr_path, position, round_idx)`
  - Try segmentation first (if exists)
  - Fall back to intensity-based detection
  - Return (y, x) coordinates

**Cache Strategy**:
- Cache directory: `experiment/2-tracking/iss_spot_cache/`
- Key: `{well}_r{round}_ch{summed}_thr{threshold}_{hash}.npy`

---

### Step 2: PCC Adaptation for Intensity Images
**File**: `cyclops_process/processes/auto_register/iss_cycle_pcc.py`

**Function**: `estimate_translation_pcc_intensity()`
- Reuse structure from `auto_register_pcc.py`
- Remove binary mask conversion
- Apply directly to summed spot intensities
- Try multiple preprocessing: raw, gaussian blur, edges
- Cache per round-pair

---

### Step 3: Graph-Based Nucleus Matching
**File**: `cyclops_process/processes/auto_register/iss_cycle_register.py` (helper)

**Function**: `match_nucleus_to_spots_graph()`
- Build k-NN graphs for nucleus centroids
- Build k-NN graphs for spot centroids  
- Use spatial + neighborhood consistency (skip Hu moments - not needed)
- Return matched pairs for RANSAC

---

### Step 4: Main Registration Orchestrator
**File**: `cyclops_process/processes/auto_register/iss_cycle_register.py`

**Main Function**: `auto_register_iss_rounds()`

**Signature**:
```python
def auto_register_iss_rounds(
    experiment: str,
    well: int,
    input_zarr_path: Path = None,  # Override for bc_stitched.zarr
    output_zarr_path: Path = None,  # Override for bc_stitched_registered.zarr
    spot_threshold: float = 400,
    use_bimodal_threshold: bool = True,
    transform_type: str = "similarity",
    create_overlays: bool = True,
    verbose: bool = True,
) -> dict:
```

**Logic**:
1. Load dataset and paths
2. Register nucleus → Round 0
3. For each round pair: register sequentially
4. Compose all transforms
5. Apply transforms and write output
6. Generate QA overlays and metrics

**CLI Interface** (in same file):
```bash
# Single well
python -m cyclops_process.processes.auto_register.iss_cycle_register \
  --experiment ops0035_20250501 \
  --well 1

# All wells
python -m cyclops_process.processes.auto_register.iss_cycle_register \
  --experiment ops0035_20250501 \
  --well all
```

---

### Step 5: Visualization & QA
**File**: `cyclops_process/processes/auto_register/iss_cycle_visualization.py`

**Outputs** (saved to `experiment/2-tracking/iss_registration_overlays/A{well}/`):
1. **Per round-pair overlays** (10 pairs: nucleus→R0, R0→R1, ..., R8→R9):
   - Before alignment (red/green overlay)
   - After PCC (if used)
   - After final affine
   - Grid of 4-6 crop regions across well

2. **Drift trajectory plot**:
   - Line plot showing cumulative (dy, dx) across rounds
   - Compare before/after correction
   - Mark inlier ratios per round

3. **Metrics CSV**:
   - Per round-pair: inlier ratio, median residual, transform parameters
   - Similar structure to auto_register metrics

4. **Spot matching visualization**:
   - Sample 100 matched spot pairs
   - Show neighborhoods (like graph matching viz in auto_register)

---

### Step 6: Integration into Pipeline
**File**: `cyclops_process/processes/iss.py`

**Add function**:
```python
def register_iss_rounds_wrapper(experiment: str):
    """Call ISS round registration before spot calling"""
    from cyclops_process.processes.auto_register.iss_cycle_register import auto_register_iss_rounds
    
    for well in [1, 2, 3]:
        auto_register_iss_rounds(experiment, well)
```

**Update store paths** in `cyclops_process/data/experiment.py`:
```python
self.store_paths = {
    ...
    "iss_stitch_registered": self.preprocess_in_situ / "stitch/bc_stitched_registered.zarr",
    ...
}
```

---

## Key Design Decisions

### 1. Work in Stitched Well Space
- Input: `bc_stitched.zarr` (post-stitch, pre-segmentation)
- Compute one affine per well (not per tile)
- Simpler than tile-by-tile registration
- Consistent with auto_register approach

### 2. Spot Detection Strategy
- Sum ch1-4 first (guaranteed signal)
- Local maxima with configurable threshold
- Bimodal binning option for robustness
- Cache detected spots for speed

### 3. Nucleus Matching Strategy
- Use graph-based matching (not just distance)
- Spatial + neighborhood consistency
- Fall back to intensity peaks if no segmentation
- Skip Hu moments (not needed for point clouds)

### 4. Transform Composition
- Sequential registration: R[i+1] → R[i] → ... → R[0]
- Compose affines (matrix multiplication)
- All rounds in R[0] coordinate system
- Nucleus aligned to R[0] as well

### 5. Validation & QA
- Automatic overlays per round-pair
- Metrics tracking (inlier ratios, residuals)
- Drift trajectory plots
- Comparison with/without correction

---

## Success Criteria

### Registration Quality
- RANSAC inlier ratio > 30% per round-pair
- Median residual < 2 pixels after registration  
- Max cumulative drift < 5 pixels across all 10 rounds
- Nucleus alignment inlier ratio > 40%

### Performance
- < 10 minutes per well (includes spot detection + RANSAC)
- Efficient caching (spots detected once, reused)
- Parallel well processing (joblib)

### Validation
- Visual overlays confirm spot alignment
- Drift plots show improvement vs. uncorrected
- Downstream spot calling: fewer lost spots, better decoding

---

## Questions for User Confirmation

1. **Output Size**: Should registered zarr maintain same canvas size (with cropping), or expand to fit all transformed content (with padding)?

2. **Transform Type**: Start with "similarity" (scale+rotation+translation) or simpler "euclidean" (rotation+translation)?

3. **Threshold Strategy**: Fixed threshold (400) or auto bimodal binning as default?

4. **Nucleus Fallback**: If ISS segmentation doesn't exist, detect nucleus peaks from intensity - is this acceptable?

5. **Pipeline Integration**: Should this run automatically in iss.py pipeline, or as separate manual step initially?

---

## Implementation Timeline

1. **Day 1**: Spot detection module + caching (iss_cycle_spots.py)
2. **Day 2**: Main registration loop + nucleus matching (iss_cycle_register.py)  
3. **Day 3**: Apply transforms + write output, integrate PCC
4. **Day 4**: Visualization + QA overlays (iss_cycle_visualization.py)
5. **Day 5**: Testing on high-drift experiment, parameter tuning
6. **Day 6**: Integration into pipeline, documentation

**Total Estimated Time**: 6 days (includes testing & iteration)