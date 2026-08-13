"""
Centralised contrast-limit (clim) computation for OME-Zarr pyramid channels.

Design
------
Every channel is matched to a ``ChannelClimProfile`` that fully describes how
its contrast limits are computed.  There are two flavours:

* **Fixed-range** profiles (``fixed_range`` is set) – the same lo/hi is used
  at every pyramid level with no sampling.  Phase-contrast, membrane
  prediction, and nuclei prediction fall into this category.
* **Data-driven** profiles – percentile-based statistics are computed from
  multi-point spatial samples, then scaled per pyramid level.

All magic numbers live in ``CHANNEL_PROFILES`` below.  Nothing is hardcoded
elsewhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Profile dataclass
# ---------------------------------------------------------------------------
@dataclass
class ChannelClimProfile:
    """Describes how to compute contrast limits for one channel type."""

    name: str = "unknown"

    # --- Fixed-range override (skips all sampling / scaling) ---------------
    # If set, this (lo, hi) is returned verbatim for every pyramid level.
    fixed_range: Optional[Tuple[float, float]] = None

    # --- Percentile-based computation (used when fixed_range is None) ------
    lo_percentile: float = 1.0
    hi_percentile: float = 99.5

    # Headroom multipliers applied to the *range* (hi_raw - lo_raw).
    # lo_headroom < 1 pushes lo further below lo_raw; hi_headroom > 1 pushes
    # hi further above hi_raw.
    lo_headroom: float = 0.9
    hi_headroom: float = 1.3

    # Hard floor / ceiling (dtype-aware defaults applied at runtime).
    lo_floor: Optional[float] = None
    hi_ceiling: Optional[float] = None

    # Per-level expansion factor:  how much the window grows when moving from
    # the reference (coarsest) level toward full resolution.
    level_expansion: float = 2.0

    # Minimum window width to prevent collapsed ranges.
    min_window: float = 1.0


# ---------------------------------------------------------------------------
# Profile registry – order matters (first match wins)
# ---------------------------------------------------------------------------
# Each entry is (matcher_fn, profile).  matcher_fn receives the *lowered*
# channel name and returns True when the profile applies.

CHANNEL_PROFILES: list[tuple[Callable[[str], bool], ChannelClimProfile]] = [
    # ── Fixed-range channels ──────────────────────────────────────────────
    # Phase contrast / focus – symmetric float range, same at every level
    (
        lambda n: any(k in n for k in ("phase2d", "phase_contrast", "phase contrast", "focus")),
        ChannelClimProfile(
            name="phase_contrast",
            fixed_range=(-0.5, 0.85),
        ),
    ),
    # Membrane prediction – fixed float range (exact match to avoid catching
    # cell painting membrane stains like CP1_plasma_membrane_WGA)
    (
        lambda n: n == "membrane_prediction",
        ChannelClimProfile(
            name="membrane_prediction",
            fixed_range=(-1.0, 3.0),
        ),
    ),
    # Nuclei prediction – fixed uint range (exact match to avoid catching
    # cell painting nuclear stains like CP1_nuclei_Hoechst or nucleoli)
    (
        lambda n: n == "nuclei_prediction",
        ChannelClimProfile(
            name="nuclei_prediction",
            fixed_range=(0.0, 50.0),
        ),
    ),

    # ── Data-driven channels ──────────────────────────────────────────────
    # DAPI / nuclear stain
    (
        lambda n: any(k in n for k in ("dapi", "hoechst")),
        ChannelClimProfile(
            name="nuclear_stain",
            lo_percentile=30.0,
            hi_percentile=99.8,
            lo_headroom=1.0,
            hi_headroom=1.3,
            lo_floor=0.0,
            level_expansion=1.3,
        ),
    ),
    # ISS base channels (MiSeq A/C/G/T)
    # Sparse signal: bright spots on dark background, most pixels are BG.
    # Use very high percentile to reach actual signal, not just background.
    (
        lambda n: "miseq" in n,
        ChannelClimProfile(
            name="iss_base",
            lo_percentile=30.0,
            hi_percentile=99.95,
            lo_headroom=1.0,
            hi_headroom=1.2,
            lo_floor=0.0,
            level_expansion=1.3,
        ),
    ),
    # Cell painting stains (CP1_*, CP2_*)
    # Fluorescent stains – same treatment as fluorescence profile.
    (
        lambda n: n.startswith(("cp1_", "cp2_")),
        ChannelClimProfile(
            name="cell_painting",
            lo_percentile=1.0,
            hi_percentile=99.5,
            lo_headroom=1.0,
            hi_headroom=1.0,
            lo_floor=0.0,
            level_expansion=1.3,
        ),
    ),
    # GFP / fluorescent protein channels
    # Sparse bright signal on dark background – keep hi close to actual
    # signal so the display doesn't wash out to dim.
    (
        lambda n: any(k in n for k in ("gfp", "egfp", "yfp", "cfp", "mcherry", "rfp", "tdtomato")),
        ChannelClimProfile(
            name="fluorescence",
            lo_percentile=1.0,
            hi_percentile=99.5,
            lo_headroom=1.0,
            hi_headroom=1.0,
            lo_floor=0.0,
            level_expansion=1.3,
        ),
    ),
]

# Fallback when no matcher hits
DEFAULT_PROFILE = ChannelClimProfile(name="unknown")


def match_profile(channel_name: str) -> ChannelClimProfile:
    """Return the first matching profile for *channel_name* (case-insensitive)."""
    name_lower = channel_name.lower()
    for matcher, profile in CHANNEL_PROFILES:
        if matcher(name_lower):
            return profile
    return DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Multi-point spatial sampling
# ---------------------------------------------------------------------------
def sample_channel_robust(
    arr_tc: "dask.array.Array | np.ndarray",
    n_samples: int = 5,
    window: int = 256,
) -> np.ndarray:
    """Sample pixel values from multiple spatial locations.

    Skips empty/uniform blocks so the returned sample is representative of
    actual tissue signal rather than background padding.

    Parameters
    ----------
    arr_tc : array-like
        2-D (Y, X) or 3-D (Z, Y, X) array for a single time-point / channel.
    n_samples : int
        Maximum number of windows to read.
    window : int
        Side length of each sample window in pixels.

    Returns
    -------
    np.ndarray
        1-D float32 array of sampled pixel values.
    """
    shape = tuple(int(s) for s in arr_tc.shape)
    h, w = shape[-2], shape[-1]

    # Build a grid of candidate centre points (centre, quadrants, thirds)
    points = [
        (h // 2, w // 2),
        (h // 4, w // 4),
        (h // 4, 3 * w // 4),
        (3 * h // 4, w // 4),
        (3 * h // 4, 3 * w // 4),
        (h // 3, w // 2),
        (2 * h // 3, w // 2),
    ]

    samples: list[np.ndarray] = []
    for cy, cx in points[:n_samples]:
        y0 = max(0, cy - window // 2)
        x0 = max(0, cx - window // 2)
        y1 = min(h, y0 + window)
        x1 = min(w, x0 + window)
        try:
            if len(shape) == 3:
                block = np.asarray(arr_tc[0, y0:y1, x0:x1], dtype=np.float32).ravel()
            else:
                block = np.asarray(arr_tc[y0:y1, x0:x1], dtype=np.float32).ravel()
        except Exception:
            continue
        finite = block[np.isfinite(block)]
        if finite.size > 0 and np.ptp(finite) > 0:
            samples.append(finite)
        if len(samples) >= n_samples:
            break

    if not samples:
        # Last-resort: sub-sampled full array
        try:
            return np.asarray(arr_tc, dtype=np.float32).ravel()[::256]
        except Exception:
            return np.zeros(1, dtype=np.float32)

    return np.concatenate(samples)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_channel_clims(
    sample: np.ndarray,
    profile: ChannelClimProfile,
) -> Tuple[float, float]:
    """Compute contrast limits from a pixel sample using *profile* parameters.

    If the profile has ``fixed_range`` set this function still respects it
    (returns immediately), but callers should typically check that first.
    """
    if profile.fixed_range is not None:
        return profile.fixed_range

    flat = sample.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0

    # Sub-sample for speed when the sample is very large
    step = max(1, flat.size // 500_000)
    flat = flat[::step]

    lo_raw = float(np.percentile(flat, profile.lo_percentile))
    hi_raw = float(np.percentile(flat, profile.hi_percentile))

    # Expand range outward by headroom
    window = hi_raw - lo_raw
    if window <= 0:
        window = max(abs(hi_raw), 1.0)

    lo = lo_raw - window * (1.0 - profile.lo_headroom)
    hi = hi_raw + window * (profile.hi_headroom - 1.0)

    # Clamp to floor / ceiling
    if profile.lo_floor is not None:
        lo = max(lo, profile.lo_floor)
    if profile.hi_ceiling is not None:
        hi = min(hi, profile.hi_ceiling)

    # Enforce minimum window
    if hi - lo < profile.min_window:
        mid = (lo + hi) / 2.0
        lo = mid - profile.min_window / 2.0
        hi = mid + profile.min_window / 2.0

    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-level scaling
# ---------------------------------------------------------------------------
def scale_clims_to_level(
    base_clims: Tuple[float, float],
    profile: ChannelClimProfile,
    base_level: int,
    target_level: int,
) -> Tuple[float, float]:
    """Scale *base_clims* (computed at *base_level*) to *target_level*.

    Fixed-range profiles return their fixed values regardless of level.
    Data-driven profiles expand the window toward higher resolution using
    ``profile.level_expansion``.
    """
    if profile.fixed_range is not None:
        return profile.fixed_range

    lo_b, hi_b = base_clims
    window = hi_b - lo_b

    # Positive steps = moving toward higher resolution (larger arrays)
    steps = max(0, base_level - target_level)
    scale = profile.level_expansion ** steps

    # Asymmetric expansion: anchor lo, grow hi upward only
    lo = lo_b
    hi = lo_b + window * scale

    # Re-apply floor / ceiling
    if profile.lo_floor is not None:
        lo = max(lo, profile.lo_floor)
    if profile.hi_ceiling is not None:
        hi = min(hi, profile.hi_ceiling)

    # Enforce minimum window
    if hi - lo < profile.min_window:
        hi = lo + profile.min_window

    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Convenience: compute clims for all channels of a position in one call
# ---------------------------------------------------------------------------
def compute_position_clims(
    fov,
    pos_path: str,
    source_store,
    channel_names: Sequence[str],
    levels_sorted: Sequence[int],
    scale_factor: float = 2.0,
) -> dict[int, list[Tuple[float, float] | None]]:
    """Return ``{level: [per_channel_clims]}`` for every level of a position.

    Parameters
    ----------
    fov
        Open iohub FOV handle (supports ``fov["0"]``, etc.).
    pos_path : str
        Position path inside the zarr store (e.g. ``"A/1/0"``).
    source_store : Path
        Root zarr store path (used for ``da.from_zarr``).
    channel_names : Sequence[str]
        Ordered list of channel names for this store.
    levels_sorted : Sequence[int]
        Sorted list of pyramid level indices present for this position.
    scale_factor : float
        Passed through to ``ChannelClimProfile.level_expansion`` override
        (currently unused – each profile carries its own expansion factor).

    Returns
    -------
    dict mapping level (int) to a list of ``(lo, hi)`` tuples (one per channel,
    ``None`` when computation failed).
    """
    import dask.array as da
    from pathlib import Path

    c_dim = len(channel_names) if channel_names else 1
    lvl_max = max(levels_sorted) if levels_sorted else 0

    # 1. Match each channel to a profile
    profiles: list[ChannelClimProfile] = []
    for c in range(c_dim):
        cname = channel_names[c] if c < len(channel_names) else ""
        profiles.append(match_profile(cname))

    # 2. Compute base clims per channel
    #    For data-driven profiles, sample from the coarsest level (most
    #    downsampled → smallest read).  Fixed-range profiles skip sampling.
    base_per_channel: list[Tuple[float, float] | None] = [None] * c_dim

    for c in range(c_dim):
        prof = profiles[c]

        # Fixed-range channels need no sampling
        if prof.fixed_range is not None:
            base_per_channel[c] = prof.fixed_range
            continue

        # Data-driven: read from coarsest level
        try:
            arr_lvl = da.from_zarr(
                str(source_store),
                component=str(Path(pos_path) / str(lvl_max)),
            )
            if arr_lvl.ndim >= 3:
                arr_tc = arr_lvl[0, int(c)]
            else:
                arr_tc = arr_lvl
            sample = sample_channel_robust(arr_tc)
            base_per_channel[c] = compute_channel_clims(sample, prof)
        except Exception:
            base_per_channel[c] = None

    # 3. Scale to every level
    per_level: dict[int, list[Tuple[float, float] | None]] = {}
    for lvl in levels_sorted:
        per_ch: list[Tuple[float, float] | None] = [None] * c_dim
        for c in range(c_dim):
            base = base_per_channel[c]
            if base is None:
                continue
            per_ch[c] = scale_clims_to_level(
                base_clims=base,
                profile=profiles[c],
                base_level=lvl_max,
                target_level=lvl,
            )
        per_level[int(lvl)] = per_ch

    return per_level


# ---------------------------------------------------------------------------
# Validation: detect stale / bad clims
# ---------------------------------------------------------------------------
def validate_clims(
    pos_path: "Path",
    clims_data: list[list[float] | None],
    threshold: float = 0.05,
) -> list[int]:
    """Return channel indices whose stored clims look stale.

    A channel is flagged when its clim range is less than *threshold* (default
    5%) of the actual data range sampled from level 0. This catches cases like
    clims stuck at 0-1 on data with values in the hundreds.

    Parameters
    ----------
    pos_path : Path
        Position directory inside the zarr store (e.g. ``store/A/1/0``).
    clims_data : list
        Per-channel ``[lo, hi]`` entries (from ``contrast_limits_per_channel``).
    threshold : float
        Ratio below which a clim range is considered stale.

    Returns
    -------
    list[int]
        Indices of channels with stale clims (empty list means all OK).
    """
    import dask.array as da

    stale: list[int] = []
    try:
        arr = da.from_zarr(str(pos_path.parent), component=f"{pos_path.name}/0")
        n_c = int(arr.shape[1]) if arr.ndim >= 2 else 1
        H, W = int(arr.shape[-2]), int(arr.shape[-1])
        ps = min(128, H, W)
        y0 = max(0, H // 2 - ps // 2)
        x0 = max(0, W // 2 - ps // 2)

        for c_idx in range(min(n_c, len(clims_data))):
            entry = clims_data[c_idx]
            if entry is None:
                continue
            clim_range = entry[1] - entry[0]

            # Read a small centre patch
            if arr.ndim == 5:
                patch = np.asarray(arr[0, c_idx, 0, y0:y0+ps, x0:x0+ps], dtype=np.float32)
            elif arr.ndim == 4:
                patch = np.asarray(arr[0, c_idx, y0:y0+ps, x0:x0+ps], dtype=np.float32)
            else:
                continue
            finite = patch[np.isfinite(patch)]
            if finite.size == 0:
                continue

            data_range = float(np.percentile(finite, 99.5)) - float(np.percentile(finite, 1))
            if data_range > 10 and clim_range < data_range * threshold:
                stale.append(c_idx)
    except Exception:
        pass
    return stale
