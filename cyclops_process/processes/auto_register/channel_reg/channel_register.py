"""Automatic fluorescence->Phase2D channel registration for the OPS pipeline.

Replaces the manual point-and-click bead registration with a best-effort
automatic affine per fluor channel, rendered for human review before the
pipeline continues (we don't fully trust it yet, so a review checkpoint follows).

Pure-`.venv` (no biahub, no antspyx): all numpy / scipy / skimage.

Method (validated in scripts/bead_registration/):
  1. Seed scale+rotation from the generic median affine (configs/affines, stable
     across experiments); these don't drift, so only the per-experiment
     translation needs solving.
  2. Pre-warp the fluor tile by the seed, then estimate the residual translation
     with phase cross-correlation on GRADIENT-MAGNITUDE low-passed images. Gradient
     magnitude is polarity-agnostic, so inversely-correlated fluor/phase aligns
     correctly without mutual-information / ANTs.
  3. Multi-tile + cross-validation: estimate on several tiles, score every
     candidate (plus the seed itself as baseline) by mean fluor<->phase NMI across
     all tiles, keep the best. Seed-in-pool guarantees no regression; a bad tile
     can't win; a legit large correction is kept only when it raises NMI.

Writes 3-assembly/lc_<ch>_register.yml (the pipeline-consumed affine; skipped if
one already exists so manual fixes are preserved) and a review overlay PNG to
3-assembly/channel_registration/<ch>_registration.png.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage
from iohub import open_ome_zarr
from skimage.filters import sobel
from skimage.metrics import normalized_mutual_information
from skimage.registration import phase_cross_correlation

from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.profiling.decorators import versioned_function
from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

plt.rcParams["pdf.fonttype"] = 42

# fluor channel-map label (lowercase) -> (zarr channel name, register yml stem)
_FLUOR_LABELS = {
    "gfp": ("GFP", "lc_GFP_register"),
    "mcherry": ("mCherry", "lc_mCherry_register"),
    "cy5": ("Cy5", "lc_Cy5_register"),
}

# ANTs method runs as a subprocess in an isolated env (antspyx pins scipy<1.16,
# so it cannot live in the shared .venv). Override via OPS_ANTS_ENV / BIAHUB_BEADS_ENV.
DEFAULT_ANTS_ENV = os.environ.get(
    "OPS_ANTS_ENV", os.environ.get("BIAHUB_BEADS_ENV",
                                   "biahub_beads_env"))
# Helper run in the antspyx env. fwdtransforms is the pull matrix (fixed->moving).
# NB: never name the temp file ants.py — it would shadow the antspyx package.
_ANTS_HELPER = '''
import json, sys, numpy as np, ants
fixed = np.load(sys.argv[1]).astype("float32"); moving = np.load(sys.argv[2]).astype("float32")
reg = ants.registration(fixed=ants.from_numpy(fixed), moving=ants.from_numpy(moving),
                        type_of_transform=sys.argv[3], aff_metric="mattes")
tx = ants.read_transform(reg["fwdtransforms"][0])
json.dump({"parameters": [float(x) for x in tx.parameters],
           "fixed": [float(x) for x in tx.fixed_parameters]}, open(sys.argv[4], "w"))
'''


# --------------------------------------------------------------------------- #
# affine + image helpers
# --------------------------------------------------------------------------- #
def _load_affine_zyx(yml_path) -> np.ndarray:
    d = yaml.safe_load(open(yml_path))
    return np.asarray(d["affine_transform_zyx"], dtype=float)


def _yx_pull(M_zyx) -> np.ndarray:
    return np.array([[M_zyx[1, 1], M_zyx[1, 2], M_zyx[1, 3]],
                     [M_zyx[2, 1], M_zyx[2, 2], M_zyx[2, 3]],
                     [0.0, 0.0, 1.0]])


def _norm(img, p_lo=1, p_hi=99.5):
    lo, hi = np.percentile(img, [p_lo, p_hi])
    return np.clip((img.astype(float) - lo) / max(hi - lo, 1e-6), 0, 1)


def _apply_yx(fluor_yx, M_yx_pull, out_shape):
    return scipy.ndimage.affine_transform(
        np.nan_to_num(fluor_yx.astype(np.float32)), M_yx_pull,
        output_shape=out_shape, order=1)


def _scale(M_zyx):
    F = np.linalg.inv(M_zyx)
    return float(np.hypot(F[1, 1], F[2, 1])), float(np.hypot(F[1, 2], F[2, 2]))


def _nmi(a_yx, b_yx, crop=1500):
    cy, cx = a_yx.shape[0] // 2, a_yx.shape[1] // 2
    h = crop // 2
    sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
    return float(normalized_mutual_information(_norm(a_yx[sl]), _norm(b_yx[sl])))


def _gradmag(img, sigma):
    """Low-pass to cell-shape scale, then gradient magnitude (polarity-agnostic)."""
    return sobel(scipy.ndimage.gaussian_filter(_norm(img).astype(np.float32), sigma))


def _residual_pcc(fixed_img, moving_img, sigma, upsample=10):
    """Residual translation (pull 4x4) via gradient-magnitude phase cross-correlation.

    Gradient magnitude is sign-independent, so inverse correlation is handled.
    Pure .venv. Returns a 4x4 pull matrix (translation only).
    """
    shift, _, _ = phase_cross_correlation(
        _gradmag(fixed_img, sigma), _gradmag(moving_img, sigma), upsample_factor=upsample)
    R = np.eye(4)
    R[1, 3] = -float(shift[0])   # registered(p)=moving(R@p) -> R trans = -shift
    R[2, 3] = -float(shift[1])
    return R


def _ants_to_pull4(params, fixed):
    """ANTs 2D MatrixOffset params (numpy YX axis order) -> 4x4 zyx pull matrix."""
    A = np.array(params[:4], float).reshape(2, 2)
    t = np.array(params[4:6], float)
    c = np.array(fixed, float)
    trans = t + c - A @ c
    M = np.eye(4)
    M[1:3, 1:3] = A
    M[1, 3], M[2, 3] = trans[0], trans[1]
    return M


def _sitk_to_pull4(transform):
    """Any SimpleITK 2D transform (fixed->moving, physical x,y) -> 4x4 zyx pull.

    Recover the affine numerically via basis points (robust to transform subtype /
    composite). GetImageFromArray maps numpy (Y,X) -> sitk (x=col=X, y=row=Y), so we
    swap (x,y)->(Y,X). x_m = A @ [x_f,y_f] + offset, with A columns = images of unit x,y.
    """
    o = np.array(transform.TransformPoint((0.0, 0.0)), float)
    ex = np.array(transform.TransformPoint((1.0, 0.0)), float)
    ey = np.array(transform.TransformPoint((0.0, 1.0)), float)
    A = np.column_stack([ex - o, ey - o])                      # 2x2 in (x,y)
    M = np.eye(4)
    M[1, 1], M[1, 2] = A[1, 1], A[1, 0]                         # (x,y) -> (y,x)
    M[2, 1], M[2, 2] = A[0, 1], A[0, 0]
    M[1, 3], M[2, 3] = o[1], o[0]
    return M


def _residual_itk(fixed_img, moving_img, sigma):
    """Residual (pull 4x4) via SimpleITK Mattes mutual-information multi-resolution
    Similarity registration on gaussian low-passed images. Pure .venv (no antspyx).
    MI is polarity-agnostic, so inverse correlation is handled.
    """
    import SimpleITK as sitk
    fx = sitk.GetImageFromArray(
        scipy.ndimage.gaussian_filter(_norm(fixed_img).astype(np.float32), sigma))
    mv = sitk.GetImageFromArray(
        scipy.ndimage.gaussian_filter(_norm(moving_img).astype(np.float32), sigma))
    init = sitk.CenteredTransformInitializer(
        fx, mv, sitk.Similarity2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.2, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=300,
        gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 0])
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    R.SetInitialTransform(init, inPlace=False)
    final = R.Execute(fx, mv)
    return _sitk_to_pull4(final)


def _residual_ants(fixed_img, moving_img, sigma, ants_env, transform_type="Similarity"):
    """Residual (pull 4x4) via ANTs mutual-information registration on gaussian
    low-passed images. Runs in an isolated antspyx env via subprocess. Handles
    inverse correlation (MI is polarity-agnostic) and can refine scale/rotation.
    """
    fixed = scipy.ndimage.gaussian_filter(_norm(fixed_img).astype(np.float32), sigma)
    moving = scipy.ndimage.gaussian_filter(_norm(moving_img).astype(np.float32), sigma)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fp, mp, op, hp = td / "f.npy", td / "m.npy", td / "o.json", td / "_antsreg.py"
        np.save(fp, fixed); np.save(mp, moving); hp.write_text(_ANTS_HELPER)
        cmd = [str(Path(ants_env) / "bin" / "python"), str(hp),
               str(fp), str(mp), transform_type, str(op)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not op.exists():
            raise RuntimeError(f"ANTs helper failed:\n{r.stderr[-400:]}")
        res = json.loads(op.read_text())
    return _ants_to_pull4(res["parameters"], res["fixed"])


def _load_tile_yx(store, tile, channel):
    with open_ome_zarr(store, mode="r") as ds:
        arr = ds[tile]
        ci = arr.channel_names.index(channel)
        data = np.asarray(arr.data[0, ci])
    return data[data.shape[0] // 2]


def _pick_tiles(dataset, channel, n_tiles, min_std=0.02):
    fluor_store = dataset.store_paths["lc_20x_fluor_2d_flatfield"]
    phase_store = dataset.store_paths["lc_20x_phase_2d_optimized"]
    with open_ome_zarr(fluor_store, mode="r") as f:
        fpos = {p[0] for p in f.positions()}
    with open_ome_zarr(phase_store, mode="r") as p:
        ppos = {q[0] for q in p.positions()}
    common = sorted(fpos & ppos)
    if not common:
        return [], fluor_store, phase_store
    step = max(1, len(common) // (n_tiles * 3))
    scored = []
    for t in common[::step]:
        try:
            s = float(_norm(_load_tile_yx(phase_store, t, "Phase2D")).std())
            if s >= min_std:
                scored.append((s, t))
        except Exception:
            continue
    scored.sort(reverse=True)
    return [t for _, t in scored[:n_tiles]], fluor_store, phase_store


# --------------------------------------------------------------------------- #
# core: multi-tile seeded translation refinement for one channel
# --------------------------------------------------------------------------- #
def _refine_channel(dataset, channel, seed_yml, n_tiles, sigma, max_residual_px,
                    method, ants_env, scale_band, verbose):
    seed4 = _load_affine_zyx(seed_yml)
    seed_yx = _yx_pull(seed4)
    tiles, fstore, pstore = _pick_tiles(dataset, channel, n_tiles)
    if not tiles:
        raise RuntimeError(f"No populated tiles for {dataset.experiment} {channel}")
    data = [(t, _load_tile_yx(fstore, t, channel), _load_tile_yx(pstore, t, "Phase2D"))
            for t in tiles]

    # candidates: the seed (baseline) + one residual per tile (method-dependent)
    candidates = [("seed", seed4)]
    for t, fl, ph in data:
        fluor_in = _apply_yx(fl, seed_yx, ph.shape)   # coarse-align by seed
        try:
            if method == "itk":
                R = _residual_itk(ph, fluor_in, sigma)
            elif method == "ants":
                R = _residual_ants(ph, fluor_in, sigma, ants_env)
            else:
                R = _residual_pcc(ph, fluor_in, sigma)
        except Exception as e:
            if verbose:
                print(f"    {t}: {method} residual failed ({e})")
            continue
        if _grid_resid(np.eye(3), _yx_pull(R)) > max_residual_px:
            continue                                  # implausible shift -> drop
        M = seed4 @ R
        sy, sx = _scale(M)                            # reject scale collapse (ANTs)
        if not (scale_band[0] <= sy <= scale_band[1] and scale_band[0] <= sx <= scale_band[1]):
            continue
        candidates.append((f"{method}[{t}]", M))

    # cross-validated selection: mean NMI across ALL tiles (seed in pool)
    def score(M):
        myx = _yx_pull(M)
        return float(np.mean([_nmi(ph, _apply_yx(fl, myx, ph.shape)) for _, fl, ph in data]))

    scored = sorted(((n, M, score(M)) for n, M in candidates), key=lambda x: x[2], reverse=True)
    best_name, M, best_nmi = scored[0]
    seed_nmi = next(s for n, _, s in scored if n == "seed")
    note = (f"auto channel_reg: best={best_name} meanNMI={best_nmi:.4f} vs seed "
            f"{seed_nmi:.4f} ({len(candidates) - 1} candidates / {len(data)} tiles)")
    if verbose:
        print(f"  {channel}: {note}")
    return M, note, data


def _render_review(channel, M, data, out_png):
    """phase (gray) + registered fluor (magenta), full tile + zoom, for review."""
    t, fluor, phase = data[0]
    reg = _apply_yx(fluor, _yx_pull(M), phase.shape)
    g, m = _norm(phase, 1, 99), _norm(reg, 75, 99.8) * 0.85
    rgb = np.zeros((*g.shape, 3))
    rgb[..., 0] = np.clip(g + m, 0, 1); rgb[..., 1] = g; rgb[..., 2] = np.clip(g + m, 0, 1)
    cy, cx = phase.shape[0] // 2, phase.shape[1] // 2
    z = 256
    sl = (slice(cy - z, cy + z), slice(cx - z, cx + z))
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    ax[0].imshow(rgb, interpolation="nearest")
    ax[0].add_patch(plt.Rectangle((sl[1].start, sl[0].start), 2 * z, 2 * z, ec="yellow", fc="none", lw=1))
    ax[0].set_title(f"{channel}->Phase2D  tile {t}", fontsize=10); ax[0].axis("off")
    ax[1].imshow(rgb[sl[0], sl[1]], interpolation="nearest")
    ax[1].set_title(f"{channel} (zoom) — phase=gray, {channel}=magenta", fontsize=10); ax[1].axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def _backup_affine(out_yml: Path, review_dir: Path):
    """Save a timestamped copy of an existing affine before it is overwritten,
    so a prior (possibly manual) registration is never lost — even in force mode."""
    if not out_yml.exists():
        return
    backup_dir = review_dir / "previous_affines"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    dst = backup_dir / f"{out_yml.stem}_{ts}{out_yml.suffix}"
    shutil.copy2(out_yml, dst)
    print(f"[channel_reg] backed up existing {out_yml.name} -> {dst}")


def _fluor_channels_from_map(channel_map) -> list:
    """Fluor channels from a channel_map. The fluorophore (GFP/mCherry/Cy5) may be
    the key (e.g. {'GFP': 'cis-Golgi...'}) or the value; match either."""
    out = []
    for k, v in (channel_map or {}).items():
        for cand in (k, v):
            info = _FLUOR_LABELS.get(str(cand).strip().lower()) if cand is not None else None
            if info:
                if info not in out:
                    out.append(info)
                break
    return out


def _parse_note(note: str) -> dict:
    """Extract best/NMI/margin from a channel_reg _note string."""
    m = re.search(r"best=(\S+)\s+meanNMI=([0-9.]+)\s+vs seed\s+([0-9.]+)", note or "")
    if not m:
        return {"best": None, "best_nmi": None, "seed_nmi": None, "nmi_margin": None}
    bn, sn = float(m.group(2)), float(m.group(3))
    return {"best": m.group(1), "best_nmi": bn, "seed_nmi": sn, "nmi_margin": bn - sn}


def _confidence_label(parsed: dict) -> str:
    """Heuristic confidence in the auto registration, from NMI improvement over seed."""
    best, margin = parsed.get("best"), parsed.get("nmi_margin")
    if best is None:
        return "unknown"
    if best == "seed":
        return "LOW — fell back to generic seed (no per-experiment improvement); review"
    if margin is None:
        return "unknown"
    if margin >= 0.0020:
        return "HIGH"
    if margin >= 0.0005:
        return "MEDIUM"
    return "LOW — marginal NMI gain; review"


def _qc_dir(dataset):
    return dataset.results / "channel_registration" / "qc"


def _write_qc(dataset, rec):
    """Persist one channel's QC record (race-safe: one file per channel). This is
    the source of truth — consumers load these, never function return values."""
    d = _qc_dir(dataset)
    d.mkdir(parents=True, exist_ok=True)
    rec = {**rec, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / f"{rec['channel']}.json").write_text(json.dumps(rec))


def _grid_resid(Ayx, Byx, n=2048, step=128) -> float:
    """Mean per-pixel displacement (px) between two YX pull affines over a grid."""
    Fa, Fb = np.linalg.inv(Ayx), np.linalg.inv(Byx)
    ys = np.arange(0, n, step)
    gy, gx = np.meshgrid(ys, ys, indexing="ij")
    pts = np.stack([gy.ravel(), gx.ravel(), np.ones(gy.size)])
    return float(np.linalg.norm((Fa @ pts)[:2] - (Fb @ pts)[:2], axis=0).mean())


def _write_affine_yml(path, channel, M, note):
    with open(path, "w") as f:
        yaml.safe_dump({
            "source_channel_names": [channel],
            "target_channel_name": "Phase2D",
            "affine_transform_zyx": M.tolist(),
            "keep_overhang": False,
            "interpolation": "linear",
            "time_indices": "all",
            "_note": note,
        }, f, sort_keys=False)


def _overlay_rgb(phase, reg):
    g, m = _norm(phase, 1, 99), _norm(reg, 75, 99.8) * 0.85
    rgb = np.zeros((*g.shape, 3))
    rgb[..., 0] = np.clip(g + m, 0, 1); rgb[..., 1] = g; rgb[..., 2] = np.clip(g + m, 0, 1)
    return rgb


def _render_compare(channel, named_affines, data, out_png):
    """Side-by-side overlays (e.g. manual | auto), full tile + zoom, for review."""
    t, fluor, phase = data[0]
    n = len(named_affines)
    cy, cx = phase.shape[0] // 2, phase.shape[1] // 2
    z = 256
    sl = (slice(cy - z, cy + z), slice(cx - z, cx + z))
    fig, ax = plt.subplots(2, n, figsize=(5.5 * n, 11), squeeze=False)
    for j, (name, M) in enumerate(named_affines):
        ov = _overlay_rgb(phase, _apply_yx(fluor, _yx_pull(M), phase.shape))
        ax[0, j].imshow(ov, interpolation="nearest")
        ax[0, j].set_title(f"{name}  NMI={_nmi(phase, _apply_yx(fluor, _yx_pull(M), phase.shape)):.4f}",
                           fontsize=10)
        ax[0, j].add_patch(plt.Rectangle((sl[1].start, sl[0].start), 2 * z, 2 * z, ec="yellow", fc="none", lw=1))
        ax[0, j].axis("off")
        ax[1, j].imshow(ov[sl[0], sl[1]], interpolation="nearest")
        ax[1, j].set_title(f"{name} (zoom)", fontsize=9); ax[1, j].axis("off")
    fig.suptitle(f"{channel}->Phase2D  tile {t}  (phase=gray, {channel}=magenta)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# pipeline entry point
# --------------------------------------------------------------------------- #
def _channel_in_store(dataset, channel) -> bool:
    store = dataset.store_paths["lc_20x_fluor_2d_flatfield"]
    try:
        with open_ome_zarr(store, mode="r") as ds:
            pos = next(ds.positions())
            return channel in pos[1].channel_names
    except Exception:
        return False


@versioned_function("v1.0")
def auto_register_channels(
    experiment: str,
    channel_map: dict | None = None,
    method: str = "itk",
    n_tiles: int = 4,
    sigma: float | None = None,
    max_residual_px: float = 60.0,
    scale_band: tuple = (0.95, 0.975),
    ants_env: str = DEFAULT_ANTS_ENV,
    overwrite: bool = False,
    test_mode: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """Auto-register each fluor channel to Phase2D and render review overlays.

    method: "itk" (default) = Mattes mutual-information affine via SimpleITK,
    pure-.venv (no isolated env); "ants" = the same MI approach via antspyx in an
    isolated env (fallback); "pcc" = pure-.venv gradient-magnitude phase
    cross-correlation (translation-only). All seed scale/rotation from the median,
    multi-tile + NMI cross-validate (seed in pool -> no regression).

    Production (test_mode=False): write 3-assembly/lc_<ch>_register.yml (skipped if
    it exists unless overwrite=True; an existing affine is always backed up first)
    and render 3-assembly/channel_registration/<ch>_registration.png.

    Test mode (test_mode=True): NEVER touch lc_<ch>_register.yml. Always refine,
    save lc_<ch>_register_auto.yml, and render a manual-vs-auto side-by-side
    (<ch>_compare.png) before adopting it.

    Returns a per-channel summary for batch validation.
    """
    if sigma is None:
        sigma = 8.0 if method == "pcc" else 4.0   # MI (itk/ants) wants finer detail than PCC
    dataset = OpsDataset(experiment)
    channels = _fluor_channels_from_map(channel_map)
    if not channels:
        print("[channel_reg] no fluorescence channels in channel_map; nothing to do.")
        return []

    review_dir = dataset.results / "channel_registration"
    summary: list[dict] = []
    for ch_name, stem in channels:
        seed_yml = dataset.config_paths.get(f"{stem}_seed")
        real_yml = dataset.config_paths[stem]
        if seed_yml is None or not Path(seed_yml).exists():
            print(f"[channel_reg] {ch_name}: no seed affine at {seed_yml}; skipping.")
            continue
        if not _channel_in_store(dataset, ch_name):
            print(f"[channel_reg] {ch_name}: not present in fluor store; skipping.")
            continue

        rec = {"experiment": experiment, "channel": ch_name}
        try:
            if test_mode:
                # Refine without touching the real affine; save *_auto.yml + compare.
                M, note, data = _refine_channel(
                    dataset, ch_name, seed_yml, n_tiles, sigma, max_residual_px,
                    method, ants_env, scale_band, verbose)
                auto_yml = real_yml.with_name(f"{real_yml.stem}_auto.yml")
                _write_affine_yml(auto_yml, ch_name, M, note)
                rec.update(_parse_note(note)); rec["confidence"] = _confidence_label(rec)
                print(f"[channel_reg] {ch_name}: TEST wrote {auto_yml.name}  "
                      f"confidence={rec['confidence']}")
                if real_yml.exists():
                    Mm = _load_affine_zyx(real_yml)
                    rec["resid_to_manual_px"] = _grid_resid(_yx_pull(Mm), _yx_pull(M))
                    _render_compare(ch_name, [("manual", Mm), ("auto", M)], data,
                                    review_dir / f"{ch_name}_compare.png")
                else:
                    rec["resid_to_manual_px"] = None  # no manual to compare against
                    _render_review(ch_name, M, data, review_dir / f"{ch_name}_registration.png")
                _write_qc(dataset, rec)
                summary.append(rec)
                continue

            # ---- production ----
            if real_yml.exists() and not overwrite:
                print(f"[channel_reg] {ch_name}: {real_yml.name} exists; keeping it "
                      f"(re-rendering review only).")
                M = _load_affine_zyx(real_yml)
                tiles, fstore, pstore = _pick_tiles(dataset, ch_name, 1)
                data = [(tiles[0], _load_tile_yx(fstore, tiles[0], ch_name),
                         _load_tile_yx(pstore, tiles[0], "Phase2D"))] if tiles else []
                rec.update(_parse_note(str(yaml.safe_load(open(real_yml)).get("_note", ""))))
                rec["confidence"] = "kept existing affine (not re-run)"
            else:
                M, note, data = _refine_channel(
                    dataset, ch_name, seed_yml, n_tiles, sigma, max_residual_px,
                    method, ants_env, scale_band, verbose)
                _backup_affine(real_yml, review_dir)   # never lose a prior affine
                _write_affine_yml(real_yml, ch_name, M, note)
                rec.update(_parse_note(note)); rec["confidence"] = _confidence_label(rec)
                print(f"[channel_reg] {ch_name}: wrote {real_yml}  confidence={rec['confidence']}")
            if data:
                _render_review(ch_name, M, data, review_dir / f"{ch_name}_registration.png")
        except Exception as e:
            rec["error"] = str(e)
            print(f"[channel_reg] {ch_name}: FAILED — {e}")
        _write_qc(dataset, rec)
        summary.append(rec)

    print(f"[channel_reg] overlays in {review_dir}")
    return summary


def submit_channel_registration_jobs(
    experiment: str,
    channel_map: dict | None = None,
    method: str = "itk",
    n_tiles: int = 4,
    sigma: float | None = None,
    slurm_params: dict | None = None,
    wait_for_completion: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """Pipeline step: fan out one SLURM job per fluor channel (GFP/mCherry/Cy5 in
    parallel), each running auto_register_channels for that single channel. Mirrors
    submit_registration_jobs / submit_tracking_jobs. Returns submit_parallel_jobs result.

    No-op (returns the same success/jobs=[] shape) when the channel_map declares
    no fluorescent channels OR when none of the declared ones have a seed affine
    and on-disk data — so callers can run this unconditionally and it'll just
    short-circuit on phase-only experiments.
    """
    no_op = {"success": True, "jobs": [], "skipped_reason": None}

    # CLI/Nextflow path: no channel_map dict passed -> load it from the
    # experiment config (OpsDataset auto-reads OPS_EXP_CONFIG_FILE). The
    # orchestrator always passes channel_map explicitly, so this only fires
    # for scalar-only callers like dispatch_cli.
    if channel_map is None:
        channel_map = OpsDataset(experiment).channel_map_data

    # Fast path: empty / None channel_map, or only phase-like labels.
    declared = _fluor_channels_from_map(channel_map)
    if not declared:
        no_op["skipped_reason"] = "no fluorescent channels declared in channel_map"
        print(f"[channel_reg] {no_op['skipped_reason']}; nothing to submit.")
        return no_op

    if slurm_params is None:
        slurm_params = {"timeout_min": 20, "mem": "16G", "cpus_per_task": 4,
                        "slurm_partition": "cpu"}
    dataset = OpsDataset(experiment)
    jobs = []
    missing_reasons = []
    for ch_name, stem in declared:
        seed = dataset.config_paths.get(f"{stem}_seed")
        if seed is None or not Path(seed).exists():
            missing_reasons.append(f"{ch_name}: no seed affine")
            continue
        if not _channel_in_store(dataset, ch_name):
            missing_reasons.append(f"{ch_name}: not present in fluor store")
            continue
        jobs.append({
            "name": f"{experiment}_{ch_name}",
            "func": auto_register_channels,
            "kwargs": {"experiment": experiment, "channel_map": {"x": ch_name},
                       "method": method, "n_tiles": n_tiles, "sigma": sigma},
            "metadata": {"experiment": experiment, "channel": ch_name},
        })
    if not jobs:
        reasons = "; ".join(missing_reasons) or "no usable fluor channels"
        no_op["skipped_reason"] = reasons
        print(f"[channel_reg] {reasons}; nothing to submit.")
        return no_op
    result = submit_parallel_jobs(
        jobs_to_submit=jobs, experiment=experiment, slurm_params=slurm_params,
        log_dir="channel_reg_logs", manifest_prefix="channel_reg",
        step_name='submit_channel_registration_jobs',
        dry_run=dry_run, wait_for_completion=wait_for_completion, verbose=verbose)

    if wait_for_completion and not dry_run:
        _print_registration_summary(dataset, [j["metadata"]["channel"] for j in jobs])
    return result


# ── Nextflow fan-out (setup → per-channel job) ───────────────────────────────
# Replaces the phantom-slurm `submit_channel_registration_jobs` when driven from
# Nextflow: setup lists the eligible fluor channels and each job registers ONE
# channel with no nested slurm (README "Porting Functionality" 3a). The submitit
# wrapper above is kept for the PipelineRunner / direct CLI use.

_CHANNELREG_SENTINEL = "CHANNELREG_CH "


def channel_registration_setup(
    experiment: str,
    channel_map: dict | None = None,
) -> None:
    """Nextflow fan-out setup: print each fluor channel eligible for registration
    — declared in the channel_map AND with a seed affine on disk AND present in
    the fluor store — one per line with the ``CHANNELREG_CH`` sentinel. Prints
    nothing on phase-only / unseeded experiments so the workflow no-ops."""
    if channel_map is None:
        channel_map = OpsDataset(experiment).channel_map_data
    declared = _fluor_channels_from_map(channel_map)
    if not declared:
        print("[channel_reg] no fluorescent channels declared; nothing to register.")
        return
    dataset = OpsDataset(experiment)
    for ch_name, stem in declared:
        seed = dataset.config_paths.get(f"{stem}_seed")
        if seed is None or not Path(seed).exists():
            print(f"[channel_reg] {ch_name}: no seed affine; skipping.")
            continue
        if not _channel_in_store(dataset, ch_name):
            print(f"[channel_reg] {ch_name}: not present in fluor store; skipping.")
            continue
        print(f"{_CHANNELREG_SENTINEL}{ch_name}")


def channel_registration_job(
    experiment: str,
    channel: str,
    method: str = "itk",
    n_tiles: int = 4,
    sigma: float | None = None,
):
    """Nextflow fan-out worker: register ONE fluor channel to Phase2D, no nested
    slurm. Wraps ``auto_register_channels`` for a single channel; Nextflow owns
    the per-channel parallelism."""
    return auto_register_channels(
        experiment=experiment,
        channel_map={"x": channel},
        method=method,
        n_tiles=n_tiles,
        sigma=sigma,
    )


def _print_registration_summary(dataset, channels):
    """After the channel jobs finish, LOAD each channel's persisted QC record
    (written by the worker), print a summary + confidence flag, and write the
    combined QC CSV (channel_registration/channel_registration_qc.csv)."""
    import pandas as pd
    review_dir = dataset.results / "channel_registration"
    qc_dir = _qc_dir(dataset)
    rows = []
    print(f"\n{'='*70}\n  CHANNEL REGISTRATION SUMMARY — {dataset.experiment}\n{'='*70}")
    for ch_name in channels:
        p = qc_dir / f"{ch_name}.json"
        if not p.exists():
            print(f"  {ch_name:8s}  MISSING (no QC record written)")
            rows.append({"experiment": dataset.experiment, "channel": ch_name,
                         "confidence": "MISSING"})
            continue
        rec = json.loads(p.read_text())
        margin = rec.get("nmi_margin")
        margin_s = f"{margin:+.4f}" if isinstance(margin, (int, float)) else "  n/a"
        print(f"  {ch_name:8s}  best={str(rec.get('best')):16s}  "
              f"NMI vs seed {margin_s}  confidence={rec.get('confidence')}")
        rows.append(rec)

    review_dir.mkdir(parents=True, exist_ok=True)
    csv_path = review_dir / "channel_registration_qc.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\n  Review overlays: {review_dir}")
    print(f"  QC CSV:          {csv_path}")
    if any(str(r.get("confidence", "")).startswith(("LOW", "MISSING")) for r in rows):
        print("  ⚠ One or more channels are LOW/MISSING confidence — inspect the "
              "overlays and re-register manually if needed before continuing.")
    print(f"{'='*70}\n")
