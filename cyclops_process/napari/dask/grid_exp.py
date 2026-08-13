from pathlib import Path
import os
import random

import numpy as np
from PIL import Image
from iohub import open_ome_zarr
from cyclops_utils.data.filesystem import vprintf
from cyclops_utils.data.filesystem import (
    _iter_experiment_configs,
    _extract_channels_from_config,
)


def _add_experiment_grids_in_viewer(
    v,
    source_store: Path | str,
    selected_positions: list[str],
    offsets_x: dict[str, int],
    label_font_size: int = 28,
    exp_grid_count: int = 50,
) -> None:
    """Add a synthetic grid of circular experiment placeholders with labels.

    - Circle radius equals half of the larger spatial dimension of the first loaded position
    - Circles are arranged in a grid below the real data, spaced like wells
    - One label per circle: experiment name and channel labels
    - Total circles = 3 * (# experiment config files)
    """
    try:
        if not selected_positions:
            return
        base_pos = selected_positions[0]
        with open_ome_zarr(source_store, mode="r") as store:
            fov = store[base_pos]["0"]
            y0, x0 = int(fov.shape[-2]), int(fov.shape[-1])
            try:
                base_scale = list(getattr(fov, "scale", []) or [])
            except Exception:
                base_scale = []
    except Exception:
        # Fallback to a conservative default size
        y0, x0 = 2048, 2048

    # Set circle diameter to ~well size so circles are clearly visible at current level
    marker_diameter = float(max(y0, x0))
    diameter = max(1.0, marker_diameter)
    radius = 0.5 * diameter
    spacing_x = diameter * 1.1
    spacing_y = diameter * 1.1
    # Center the experimental grid around the currently selected well (e.g., A/1)
    cx_center = float(offsets_x.get(base_pos, 0)) + (x0 * 0.5)
    cy_center = float(y0) * 0.5

    cfgs = _iter_experiment_configs()
    # Limit to exp_grid_count centered around the current experiment
    try:
        parts = Path(source_store).parts
        current_experiment_name = None
        if "ops" in parts:
            i_ops = parts.index("ops")
            if i_ops + 1 < len(parts):
                current_experiment_name = parts[i_ops + 1]
        names_all = [n for (n, _) in cfgs]
        if current_experiment_name in names_all:
            i0 = names_all.index(current_experiment_name)
            ordered: list[tuple[str, Path]] = []
            k = 0
            limit = int(max(1, exp_grid_count))
            while len(ordered) < limit and (i0 - k >= 0 or i0 + k < len(cfgs)):
                if k == 0:
                    ordered.append(cfgs[i0])
                else:
                    if i0 - k >= 0:
                        ordered.append(cfgs[i0 - k])
                    if len(ordered) >= limit:
                        break
                    if i0 + k < len(cfgs):
                        ordered.append(cfgs[i0 + k])
                k += 1
            cfgs = ordered[:limit]
        else:
            cfgs = cfgs[: int(max(1, exp_grid_count))]
    except Exception:
        cfgs = cfgs[: int(max(1, exp_grid_count))]
    # Show 3 wells per experiment
    num_items = 3 * len(cfgs)
    if num_items <= 0:
        vprintf("[exp-grids] No experiment configs; skipping overlay.")
        return

    # Grid layout (rows x cols) roughly square
    # Build a square odd-sized grid so there is a true center cell (which we leave empty)
    grid_side = int(np.ceil(np.sqrt(num_items)))
    # Force odd size to ensure a unique center
    if grid_side % 2 == 0:
        grid_side += 1
    rows = cols = grid_side
    mid_r = mid_c = rows // 2

    vprintf(
        "[exp-grids] Grid layout rows=%d cols=%d total=%d (from %d configs), spacing_scale=%.2f",
        int(rows),
        int(cols),
        int(num_items),
        int(len(cfgs)),
    )

    centers: list[tuple[float, float]] = []  # (y, x)
    labels: list[str] = []
    well_images: list[Path] = (
        []
    )  # Store randomly selected image paths for each experiment

    # Load available well images from directory
    well_images_dir = Path(os.environ.get("OPS_WELL_IMAGES_DIR", "ops_wells"))
    available_well_images = list(well_images_dir.glob("*.jpeg"))
    if not available_well_images:
        vprintf("[exp-grids] No well images found in %s", str(well_images_dir))
        return

    # Forbidden regions (avoid overlapping real wells): build rectangles around all selected wells
    forbidden_rects: list[tuple[float, float, float, float]] = (
        []
    )  # (ymin, ymax, xmin, xmax)
    try:
        for pos in selected_positions:
            cx_pos = float(offsets_x.get(pos, 0)) + (x0 * 0.5)
            cy_pos = float(y0) * 0.5
            half = diameter * 0.55
            forbidden_rects.append(
                (cy_pos - half, cy_pos + half, cx_pos - half, cx_pos + half)
            )
    except Exception:
        pass

    # Precompute grid indices ordered by distance from center (skip true center)
    all_cells: list[tuple[int, int, float]] = []
    for r in range(rows):
        for c in range(cols):
            if r == mid_r and c == mid_c:
                continue  # leave the A1 well position empty
            dr = float(r - mid_r)
            dc = float(c - mid_c)
            all_cells.append((r, c, dr * dr + dc * dc))
    # Sort so we fill nearest ring first, expanding outward
    all_cells.sort(key=lambda t: t[2])

    # Fill positions with (experiment, replicate) entries
    idx = 0
    for exp_idx, (exp_name, cfg_path) in enumerate(cfgs):
        ch_labels = _extract_channels_from_config(cfg_path)
        # Render channel list on multiple lines by replacing commas with newlines
        ch_text = ", ".join(ch_labels) if ch_labels else ""
        ch_text = ch_text.replace(",", "\n")
        # Place 3 wells per experiment, labeled 1, 2, 3
        for well_num in range(1, 4):
            label_text = f"{exp_name}\nWell {well_num}\n{ch_text}".strip()
            # Find the next cell that does not overlap any forbidden region
            placed = False
            while idx < len(all_cells) and not placed:
                r, c, _ = all_cells[idx]
                idx += 1
                row_offset = (r - mid_r) * spacing_y
                col_offset = (c - mid_c) * spacing_x
                cy = cy_center + row_offset
                cx = cx_center + col_offset
                # Check overlap
                overlaps = False
                for ymin, ymax, xmin, xmax in forbidden_rects:
                    if cy >= ymin and cy <= ymax and cx >= xmin and cx <= xmax:
                        overlaps = True
                        break
                if overlaps:
                    continue
                centers.append((float(cy), float(cx)))
                labels.append(label_text)
                # Randomly select a well image for this experiment
                well_images.append(random.choice(available_well_images))
                placed = True
            if not placed:
                break

    try:
        centers_np = np.asarray(centers, dtype=np.float32)
        # Promote 2D YX centers to match viewer dimensionality (place on displayed axes)
        try:
            nd = int(getattr(v.dims, "ndim", 2))
            disp = tuple(getattr(v.dims, "displayed", (max(0, nd - 2), max(1, nd - 1))))
            if len(disp) != 2:
                disp = (max(0, nd - 2), max(1, nd - 1))
            # Use strictly 2D coordinates for points (displayed axes only)
            coords_2d = centers_np.astype(np.float32)
            # Build 2D scale from image scale (Y,X)
            try:
                sy = (
                    float(base_scale[-2])
                    if isinstance(base_scale, (list, tuple)) and len(base_scale) >= 2
                    else 1.0
                )
                sx = (
                    float(base_scale[-1])
                    if isinstance(base_scale, (list, tuple)) and len(base_scale) >= 2
                    else 1.0
                )
            except Exception:
                sy, sx = 1.0, 1.0
            scale_2d = (sy if sy > 0 else 1.0, sx if sx > 0 else 1.0)
        except Exception:
            coords_2d = centers_np.astype(np.float32)
            scale_2d = (1.0, 1.0)

        # Replace circles with image stamps
        # Build a single composite image layer of stamps using a downscaled canvas
        # Load all the randomly selected well images
        stamp_images: list[np.ndarray] = []
        H0, W0 = 0, 0

        # First pass: determine target size from first image
        if well_images:
            try:
                first_img = Image.open(well_images[0]).convert("RGBA")
                H0, W0 = (
                    first_img.size[1],
                    first_img.size[0],
                )  # PIL size is (width, height)
            except Exception:
                H0, W0 = 512, 512

        # Second pass: load and resize all images to match target size, apply random transformations
        for well_img_path in well_images:
            try:
                stamp_img = Image.open(well_img_path).convert("RGBA")
                # Resize to match H0, W0
                if stamp_img.size != (W0, H0):
                    stamp_img = stamp_img.resize((W0, H0), Image.Resampling.LANCZOS)

                # Apply random transformations
                # 1. Random rotation (0, 90, 180, 270 degrees)
                rotation = random.choice([0, 90, 180, 270])
                if rotation != 0:
                    stamp_img = stamp_img.rotate(rotation, expand=False)

                # 2. Random horizontal flip
                if random.random() > 0.5:
                    stamp_img = stamp_img.transpose(Image.FLIP_LEFT_RIGHT)

                # 3. Random vertical flip
                if random.random() > 0.5:
                    stamp_img = stamp_img.transpose(Image.FLIP_TOP_BOTTOM)

                # Convert to numpy for color/brightness adjustments
                stamp_np = np.asarray(stamp_img).astype(np.float32)

                # Create mask for non-black pixels (where any RGB channel > threshold)
                non_black_mask = np.any(stamp_np[..., :3] > 10, axis=-1, keepdims=True)

                # 4. Desaturate colors for pastel effect
                # Convert to grayscale and blend with original (higher blend = more pastel)
                grayscale = np.mean(stamp_np[..., :3], axis=-1, keepdims=True)
                desaturation_factor = 0.6  # 60% desaturation for pastel colors
                stamp_np[..., :3] = np.where(
                    non_black_mask,
                    stamp_np[..., :3] * (1.0 - desaturation_factor) + grayscale * desaturation_factor,
                    stamp_np[..., :3],
                )

                # 5. Subtle brightness adjustment (90% to 110%)
                brightness_factor = random.uniform(0.9, 1.1)
                stamp_np[..., :3] = np.clip(
                    stamp_np[..., :3] * brightness_factor, 0, 255
                )

                # 6. Reduced contrast enhancement (100% to 110%)
                contrast_factor = random.uniform(1.0, 1.1)
                mean_rgb = np.mean(stamp_np[..., :3], axis=(0, 1), keepdims=True)
                stamp_np[..., :3] = np.clip(
                    mean_rgb + (stamp_np[..., :3] - mean_rgb) * contrast_factor, 0, 255
                )

                # 7. Subtle random color shift (±30 in each channel), preserving black corners
                color_shift = np.random.uniform(-30, 30, size=(1, 1, 3))
                stamp_np[..., :3] = np.where(
                    non_black_mask,
                    np.clip(stamp_np[..., :3] + color_shift, 0, 255),
                    stamp_np[..., :3],
                )

                # 8. Apply light gray-white overlay for mature, washed-out pastel look
                pastel_overlay = 200.0  # Light gray-white value
                pastel_alpha = 0.4  # Higher alpha for stronger pastel effect
                stamp_np[..., :3] = np.where(
                    non_black_mask,
                    np.clip(
                        stamp_np[..., :3] * (1.0 - pastel_alpha) + pastel_overlay * pastel_alpha,
                        0, 255
                    ),
                    stamp_np[..., :3],
                )

                stamp_images.append(stamp_np.astype(np.uint8))
            except Exception as _e:
                vprintf(
                    "[exp-grids] Failed to load well image %s: %s",
                    str(well_img_path),
                    str(_e),
                )
                # Use a blank image as fallback
                stamp_images.append(np.zeros((H0, W0, 4), dtype=np.uint8))

        if stamp_images and H0 > 0:
            # Stamps span the full world diameter in world coordinates
            # The scale factor determines how many pixels per world unit
            s = float(H0) / float(diameter)
            ys = np.asarray([c[0] for c in centers], dtype=np.float64)
            xs = np.asarray([c[1] for c in centers], dtype=np.float64)
            min_y = float(np.floor(np.min(ys) - diameter / 2.0))
            min_x = float(np.floor(np.min(xs) - diameter / 2.0))
            max_y = float(np.ceil(np.max(ys) + diameter / 2.0))
            max_x = float(np.ceil(np.max(xs) + diameter / 2.0))
            canvas_h = max(1, int(np.ceil((max_y - min_y) * s)))
            canvas_w = max(1, int(np.ceil((max_x - min_x) * s)))
            # Safety cap to avoid accidental huge allocations
            max_canvas_px = 16000
            if canvas_h > max_canvas_px or canvas_w > max_canvas_px:
                factor = max(canvas_h / max_canvas_px, canvas_w / max_canvas_px)
                s = s / float(factor)
                canvas_h = max(1, int(np.ceil((max_y - min_y) * s)))
                canvas_w = max(1, int(np.ceil((max_x - min_x) * s)))
                vprintf(
                    "[exp-grids] downscaled canvas to (hxw)=(%d,%d), scale=%.6f",
                    int(canvas_h),
                    int(canvas_w),
                    s,
                )

            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

            for i_pt, (yy, xx) in enumerate(centers):
                # Use the specific stamp image for this experiment
                base_np = stamp_images[i_pt]
                s_rgb = base_np[..., :3].astype(np.float32)
                s_a = (base_np[..., 3].astype(np.float32) / 255.0)[..., None]

                y0_pix = int(round((float(yy) - min_y) * s - H0 / 2.0))
                x0_pix = int(round((float(xx) - min_x) * s - W0 / 2.0))
                cy0 = max(0, y0_pix)
                cx0 = max(0, x0_pix)
                y1 = min(canvas_h, y0_pix + H0)
                x1 = min(canvas_w, x0_pix + W0)
                sy0 = max(0, -y0_pix)
                sx0 = max(0, -x0_pix)
                if cy0 >= y1 or cx0 >= x1:
                    continue
                c_slice = canvas[cy0:y1, cx0:x1]
                s_rgb_c = s_rgb[sy0 : sy0 + (y1 - cy0), sx0 : sx0 + (x1 - cx0)]
                s_a_c = s_a[sy0 : sy0 + (y1 - cy0), sx0 : sx0 + (x1 - cx0)]
                # No tinting - just use the image as-is
                dst_rgb = c_slice[..., :3].astype(np.float32)
                dst_a = (c_slice[..., 3].astype(np.float32) / 255.0)[..., None]
                out_a = s_a_c + dst_a * (1.0 - s_a_c)
                out_rgb = s_rgb_c * s_a_c + dst_rgb * dst_a * (1.0 - s_a_c)
                c_slice[..., :3] = np.where(
                    out_a > 0, (out_rgb / out_a).clip(0, 255), out_rgb
                ).astype(np.uint8)
                c_slice[..., 3] = (out_a[..., 0].clip(0.0, 1.0) * 255.0).astype(
                    np.uint8
                )

            try:
                stamps_layer = v.add_image(
                    canvas,
                    name="experiment-stamps",
                    translate=(min_y, min_x),
                    scale=(1.0 / s, 1.0 / s),
                    blending="translucent",
                    opacity=0.8,
                    visible=False,  # Default to hidden
                )
            except Exception as _e2:
                vprintf("[exp-grids] Failed to add composite stamps: %s", str(_e2))
        try:
            circ.blending = "additive"
            circ.visible = False  # Default to hidden
        except Exception:
            pass

        # Label overlay at centers
        vprintf("[exp-grids] Adding %d labels...", int(len(centers)))
        lbl = v.add_points(
            data=coords_2d,
            size=1,
            face_color=[0.0, 0.0, 0.0, 0.0],
            border_color="transparent",
            properties={"label": np.asarray(labels, dtype="U")},
            name="experiment-labels",
            visible=False,  # Default to hidden
            # no translate/scale
        )
        lbl.text = {
            "string": "{label}",
            "size": 12,
            "color": "white",
            "anchor": "center",
        }
        # Labels already set to visible=False in constructor
        try:
            vprintf(
                "[exp-grids] Layers present: %s",
                ", ".join(
                    [
                        str(getattr(_l, "name", "?"))
                        for _l in list(getattr(v, "layers", []))
                    ]
                ),
            )
        except Exception:
            pass
    except Exception as e:
        try:
            import traceback as _tb

            vprintf("[exp-grids] Exception type=%s msg=%s", str(type(e)), str(e))
            vprintf("[exp-grids] Traceback: %s", _tb.format_exc())
        except Exception:
            vprintf(
                "[exp-grids] Exception while adding experiment grids/labels: %s", str(e)
            )
        try:
            # Fallback: add a single visible debug marker at the base center
            try:
                nd = int(getattr(v.dims, "ndim", 2))
                cur = tuple(
                    getattr(v.dims, "current_step", tuple(0 for _ in range(nd)))
                )
                disp = tuple(
                    getattr(v.dims, "displayed", (max(0, nd - 2), max(1, nd - 1)))
                )
                dbg_pt = np.asarray(cur, dtype=np.float32)
                dbg_pt[int(disp[0])] = float(cy_center)
                dbg_pt[int(disp[1])] = float(cx_center)
                dbg_pts = dbg_pt.reshape(1, -1)
            except Exception:
                dbg_pts = np.asarray([[cy_center, cx_center]], dtype=np.float32)
            dbg = v.add_points(
                data=dbg_pts,
                size=max(32.0, diameter * 0.25),
                border_color="yellow",
                border_width=3,
                border_width_is_relative=False,
                face_color=[1.0, 1.0, 0.0, 0.0],
                name="experiment-grids-debug",
                translate=(0.0,) * int(getattr(v.dims, "ndim", 2)),
                scale=tuple(scale_nd.tolist()) if scale_nd is not None else None,
            )
            try:
                dbg.visible = True
            except Exception:
                pass
            vprintf("[exp-grids] Added fallback debug marker layer.")
        except Exception:
            vprintf("[exp-grids] Fallback debug marker also failed.")
