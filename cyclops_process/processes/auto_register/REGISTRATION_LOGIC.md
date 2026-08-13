# ISS Registration Logic & Data Flow

This document describes the coordinate system logic used for ISS round-to-round registration, finalization, and data writing.

## 1. Coordinate System Definition

The registration pipeline unifies all imaging rounds into a single global coordinate system defined by the **Segmentation** (cells/nuclei masks).

- **Global Anchor**: Segmentation Mask (`bc_segmentation.zarr`)
- **Target Coordinate Space**: All images are transformed to align with the Segmentation.

## 2. Transform Chain

The registration is composed of a chain of affine transformations:

$$ \text{Spots}_{Round\_i} \xrightarrow{R_i \to R_{i-1}} \dots \xrightarrow{R_1 \to R_0} \text{Spots}_{Round\_0} \xrightarrow{Spots \to Nucleus} \text{Nucleus} \xrightarrow{Nucleus \to Seg} \text{Segmentation} $$

### A. Nucleus $\to$ Segmentation
*   **Goal**: Align the Nucleus channel (Round 0, ch0) to the Segmentation mask.
*   **Source**: `segmentation_to_nucleus.yaml`
    *   *YAML Content*: Stores $T_{Seg \to Nuc}$ (Forward).
    *   *Loading*: Loaded as $T_{Seg \to Nuc}$.
*   **Operation**: We use the loaded transform directly to get $T_{Nuc \to Seg}$ (which is actually what the value represents in the context of finalization logic).
    *   `affine_nuc_to_seg = loaded_seg_to_nuc` (Direct Use)

### B. Spots $\to$ Nucleus
*   **Goal**: Align Round 0 Spots (ch1-4) to the Nucleus channel (ch0).
*   **Method**: Phase Cross Correlation (PCC) on DAPI channels at full resolution.
*   **Source**: `nucleus_to_round0.yml`
    *   *YAML Content*: Stores $T_{Spots \to Nucleus}$ (inverted before saving).
    *   *Loading*: Loaded directly as $T_{Spots \to Nucleus}$.
*   **Sign Convention**:
    *   PCC computes raw shift (dy, dx) to align DAPI1→DAPI0 (spots→nucleus).
    *   **During storage** (lines 2228-2229): Y is stored positive, X is negated (`-dx`) to match Y convention for scipy.
    *   This ensures both axes can be applied with the same sign downstream.
    *   Rationale: `scipy.ndimage.affine_transform` uses inverse mapping (output→input), so consistent signs simplify application logic.
*   **Operation**: Use the loaded transform directly (no inversion needed).
    *   `affine_spots_to_nuc = loaded_spots_to_nuc` (Direct Use)

### C. Round $i \to$ Round $i-1$
*   **Goal**: Align Round $i$ to the previous Round $i-1$.
*   **Source**: `round{i}_to_round{i-1}.yml`
*   **Operation**: Composed sequentially.

## 3. Cumulative Transform Calculation

The cumulative transform for any given Round $i$ maps a pixel in that round's raw image to its corresponding location in the Segmentation (Anchor) space.

$$ T_{Round\_i \to Anchor} = T_{Nucleus \to Seg} \times T_{Spots \to Nucleus} \times T_{Round\_i \to Spots} $$

This is calculated in `iss_cycle_register_orchestrator.py` during the finalization job:

1.  **Key -1 (Nucleus)**:
    ```python
    affines_cumulative[-1] = affine_nuc_to_seg
    ```

2.  **Key 0 (Round 0 Spots)**:
    ```python
    affines_cumulative[0] = affine_nuc_to_seg @ affine_spots_to_nuc
    ```

3.  **Keys 1-9 (Subsequent Rounds)**:
    ```python
    affines_cumulative[i] = affines_cumulative[i-1] @ transform_i_to_prev
    ```

## 4. Application Logic Consistency

It is critical that the validation overlays and the final data writing use the exact same transforms.

### Visualization (`create_all_rounds_overlay`)
*   **Input**: `affines_cumulative` dictionary.
*   **Logic**:
    1.  Retrieves forward transform $M$ for the round (`affines_cumulative[r]`).
    2.  Adjusts for crop offset: $M_{local} = T_{offset}^{-1} \cdot M \cdot T_{offset}$.
    3.  Inverts for warping: $M_{scipy} = M_{local}^{-1}$.
    4.  Applies `ndi.affine_transform` using $M_{scipy}$.

### Data Writing (`apply_iss_transforms`)
*   **Input**: `affines_cumulative` dictionary.
*   **Logic**:
    1.  Retrieves forward transform $M$ for the round (`affines_cumulative[r]`).
    2.  Inverts for warping: $M_{inv} = M^{-1}$.
    3.  Adjusts for tiling/cropping: Computes `offset` and adjusts `matrix` relative to tile coordinates.
    4.  Applies `ndi.affine_transform` (or GPU equivalent) using $M_{inv}$.

**Conclusion**: Both processes use the identical `affines_cumulative` dictionary and apply the **Inverse** of those matrices to warp raw data into the anchor space. The visual overlays accurately represent the data written to the Zarr file.
