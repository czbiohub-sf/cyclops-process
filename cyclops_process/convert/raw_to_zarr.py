"""
Convert raw OPS tiff data from dragonfly to zarr on the fast partition.

Direct equivalent of the Windows convert.ps1 script, plus:
  - Experiment name resolution (accepts shorthand like "146")
  - Writes to fast_ops/<experiment>/0-convert/live_imaging/raw_convert/ instead of dragonfly
  - Skips datasets that are already converted
  - Submits each zarr conversion as a separate SLURM job for max throughput
  - Separate resource profiles for tracking (small, fast) vs pheno (large, slow)
  - Per-job progress monitoring via position counting
  - Copies position list JSONs alongside the zarrs
  - Checks both root and OPS_NUM_1/ subdirectory layouts

Usage:
    python -m cyclops_process.convert.raw_to_zarr 146
    python -m cyclops_process.convert.raw_to_zarr OPS0146 --dry-run
    python -m cyclops_process.convert.raw_to_zarr 146 --no-wait

    # just pheno
    python -m cyclops_process.convert.raw_to_zarr 146 --only pheno
"""
import argparse
import shutil
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())


def _convert_single(src_dir: str, out_zarr: str, recover_partial: bool = False, resume: bool = False) -> str:
    """Convert a single tiff dataset to zarr. Runs as a SLURM job.

    Parallelizes position conversion using a thread pool for I/O overlap.
    Much faster than TIFFConverter's sequential loop for large stores.

    resume: if True and the output store already exists, reuse it (do NOT
    re-init, which would wipe it) and skip positions already written. A
    position is "done" when a corner of its array is non-zero — pre-created
    positions are exactly 0, whereas any written FOV carries at least the
    camera offset. Lets a convert that hit the SLURM wall clock finish on a
    rerun instead of restarting from scratch (a full CP round is 7035 FOVs and
    NDTiff reads don't thread well, so it can exceed the time limit).

    recover_partial: if True and source contains any 0-byte NDTiffStack_*.tif,
    build a read-only-safe symlink farm in a sibling temp dir that omits the
    empty files, and open NDTiff against that farm instead of the original
    source. (We cannot rename in the instrument mount — it's read-only.)
    Positions whose data lived in the omitted stacks fail per-position and
    are skipped, so you get a partial zarr containing every recoverable
    position. Use this when the dragonfly was interrupted and cannot be
    re-imaged.
    """
    import shutil
    import time
    import numpy as np
    from pathlib import Path
    from iohub.convert import TIFFConverter
    from iohub import open_ome_zarr
    from concurrent.futures import ThreadPoolExecutor, as_completed

    src = Path(src_dir)
    out = Path(out_zarr)
    name = src.name

    print(f"[{name}] Starting conversion -> {out.name}", flush=True)
    start = time.perf_counter()

    # Recovery path: build a symlink farm omitting 0-byte files. The farm
    # lives next to the output zarr (writable), not in the source (read-only).
    effective_src = src
    farm_dir = None
    if recover_partial:
        empties = [t for t in sorted(src.glob("*.tif")) if t.stat().st_size == 0]
        if empties:
            farm_dir = out.parent / f".{name}_recover_farm"
            if farm_dir.exists():
                shutil.rmtree(farm_dir)
            farm_dir.mkdir(parents=True)
            empty_names = {t.name for t in empties}
            # Symlink everything except the 0-byte tifs. Include the
            # NDTiff.index, display_settings.json, etc. — any non-tif metadata.
            linked = 0
            for entry in sorted(src.iterdir()):
                if entry.name in empty_names:
                    print(f"[{name}] [recover] Omitting 0-byte file: {entry.name}", flush=True)
                    continue
                (farm_dir / entry.name).symlink_to(entry)
                linked += 1
            print(f"[{name}] [recover] Symlink farm at {farm_dir} ({linked} entries, {len(empties)} omitted)", flush=True)
            effective_src = farm_dir
        else:
            print(f"[{name}] [recover] No 0-byte files found; using source as-is", flush=True)

    try:
        # Use TIFFConverter to initialize the store structure (fast, no data copy)
        converter = TIFFConverter(input_dir=effective_src, output_dir=out)

        resuming = resume and out.exists()
        if resuming:
            # Reuse the existing (partial) store — _init_zarr_arrays uses mode="w-"
            # and would fail/wipe. Rebuild reader-aligned position names from the
            # reader's HCS labels (same order the reader iterates).
            try:
                zarr_names = [f"{r}/{c}/{f}" for r, c, f in converter.reader.hcs_position_labels]
            except Exception:
                converter._init_zarr_arrays()
                zarr_names = converter.zarr_position_names
                resuming = False
            print(f"[{name}] [resume] Reusing existing store; will skip already-written positions", flush=True)
        else:
            converter._init_zarr_arrays()
            zarr_names = converter.zarr_position_names

        # Now parallel-fill positions using the reader
        reader = converter.reader
        n_total = len(zarr_names)

        print(f"[{name}] Store has {n_total} positions. Writing in parallel...", flush=True)

        output_store = open_ome_zarr(out, mode="r+")

        failed_positions = []
        n_skipped = 0

        def _write_position(args):
            nonlocal n_skipped
            zarr_pos_name, (_, fov) = args
            try:
                if resuming:
                    # Cheap corner read: already-written FOVs are non-zero.
                    arr = output_store[zarr_pos_name]["0"]
                    if int(np.asarray(arr[..., :32, :32]).max()) > 0:
                        n_skipped += 1
                        return True
                data = fov.xdata.data.compute()  # Load from NDTiff
                output_store[zarr_pos_name]["0"][:] = data
                return True
            except Exception as e:
                failed_positions.append((zarr_pos_name, type(e).__name__, str(e)[:120]))
                print(f"[{name}] ERROR on {zarr_pos_name}: {e}", flush=True)
                return False

        n_workers = min(16, n_total)
        n_ok = 0
        last_print = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for pair in zip(zarr_names, reader):
                f = executor.submit(_write_position, pair)
                futures[f] = pair[0]

            for f in as_completed(futures):
                if f.result():
                    n_ok += 1
                now = time.perf_counter()
                if now - last_print >= 15 or n_ok == n_total:
                    elapsed = now - start
                    rate = n_ok / elapsed if elapsed > 0 else 0
                    eta = (n_total - n_ok) / rate if rate > 0 else 0
                    print(f"[{name}] {n_ok}/{n_total} ({100*n_ok//n_total}%) | "
                          f"{elapsed:.0f}s elapsed | {rate:.1f} pos/s | ETA {eta:.0f}s",
                          flush=True)
                    last_print = now

        output_store.close()

        elapsed = time.perf_counter() - start
        skip_note = f" ({n_skipped} already done, skipped)" if n_skipped else ""
        print(f"[{name}] Complete: {n_ok}/{n_total} positions{skip_note} in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

        if failed_positions:
            print(f"[{name}] [recover] {len(failed_positions)} positions skipped (data in missing/corrupt stack):", flush=True)
            for pos_name, etype, emsg in failed_positions[:20]:
                print(f"[{name}]   - {pos_name}: {etype}: {emsg}", flush=True)
            if len(failed_positions) > 20:
                print(f"[{name}]   ... (+{len(failed_positions) - 20} more)", flush=True)
            # Write a manifest next to the zarr listing the lost positions
            manifest = out.with_name(out.name + ".missing_positions.txt")
            manifest.write_text("\n".join(p[0] for p in failed_positions) + "\n")
            print(f"[{name}] [recover] Wrote missing-positions manifest: {manifest}", flush=True)

        return f"Done: {out.name} ({n_ok}/{n_total}, {elapsed:.0f}s)"
    finally:
        if farm_dir is not None and farm_dir.exists():
            try:
                shutil.rmtree(farm_dir)
            except Exception as e:
                print(f"[{name}] [recover] WARN: could not remove symlink farm {farm_dir}: {e}", flush=True)


def _precheck_pheno_wells(pheno_dirs):
    """Verify all pheno wells have the same tiff count and no 0-byte files.

    File-size only — does not open or decode any image. Raises RuntimeError
    with a detailed report BEFORE any SLURM job is submitted.
    """
    if len(pheno_dirs) < 2:
        return  # nothing to compare

    print("\n[precheck] Checking pheno well tiff counts and sizes...")
    well_stats = {}
    for d in pheno_dirs:
        tifs = sorted(d.glob("*.tif"))
        sizes = [t.stat().st_size for t in tifs]
        empty = [t.name for t, s in zip(tifs, sizes) if s == 0]
        total_gb = sum(sizes) / 1e9
        well_stats[d.name] = {
            "path": d,
            "n_tifs": len(tifs),
            "empty": empty,
            "total_gb": total_gb,
            "tif_names": [t.name for t in tifs],
        }
        print(f"  {d.name}: {len(tifs)} tifs, {len(empty)} empty, {total_gb:.1f} GB")

    counts = {n: s["n_tifs"] for n, s in well_stats.items()}
    expected = max(counts.values())  # treat the largest well as canonical
    short_wells = {n: c for n, c in counts.items() if c < expected}
    wells_with_empty = {n: s["empty"] for n, s in well_stats.items() if s["empty"]}

    if not short_wells and not wells_with_empty:
        print(f"[precheck] OK: all {len(pheno_dirs)} pheno wells have {expected} tifs, no empty files.")
        return

    # Build a loud, actionable error report
    lines = []
    lines.append("=" * 78)
    lines.append("PRECHECK FAILED — refusing to submit convert jobs")
    lines.append("=" * 78)
    lines.append(f"Found {len(pheno_dirs)} pheno wells. Expected tiff count per well: {expected}")
    lines.append("")
    lines.append("Per-well summary:")
    for name, s in well_stats.items():
        marker = "OK " if s["n_tifs"] == expected and not s["empty"] else "BAD"
        lines.append(
            f"  [{marker}] {name}: {s['n_tifs']} tifs, "
            f"{len(s['empty'])} empty, {s['total_gb']:.1f} GB"
        )
        lines.append(f"        path: {s['path']}")

    if short_wells:
        lines.append("")
        lines.append("Wells with FEWER tiffs than expected:")
        # Compare by the trailing _<N> index in NDTiffStack filenames,
        # since each well's tifs are prefixed with the well name.
        import re
        idx_re = re.compile(r"NDTiffStack(?:_(\d+))?\.tif$")

        def _indices(names):
            out = set()
            for n in names:
                m = idx_re.search(n)
                if m:
                    out.add(int(m.group(1)) if m.group(1) is not None else 0)
            return out

        canonical_name = max(counts, key=counts.get)
        canonical_idx = _indices(well_stats[canonical_name]["tif_names"])
        for name, c in short_wells.items():
            short_idx = _indices(well_stats[name]["tif_names"])
            missing_idx = sorted(canonical_idx - short_idx)
            deficit = expected - c
            lines.append(f"  - {name}: short by {deficit} tifs ({c}/{expected})")
            if missing_idx:
                preview = ", ".join(f"NDTiffStack_{i}" for i in missing_idx[:8])
                more = f" ... (+{len(missing_idx) - 8} more)" if len(missing_idx) > 8 else ""
                lines.append(f"      missing stack indices vs {canonical_name}: {preview}{more}")

    if wells_with_empty:
        lines.append("")
        lines.append("Wells with EMPTY (0-byte) tiffs:")
        for name, empties in wells_with_empty.items():
            preview = ", ".join(empties[:5])
            more = f" ... (+{len(empties) - 5} more)" if len(empties) > 5 else ""
            lines.append(f"  - {name}: {len(empties)} empty file(s): {preview}{more}")

    lines.append("")
    lines.append("Likely cause: imaging on the dragonfly did not finish for these wells,")
    lines.append("or files failed to copy. Re-image / re-copy before converting, OR pass")
    lines.append("--skip-precheck if you intentionally want to convert partial data.")
    lines.append("=" * 78)

    raise RuntimeError("\n" + "\n".join(lines))


def _precheck_track_dirs(track_dirs):
    """Verify every tracking timepoint dir has at least one tiff and no 0-byte files.

    Tracking timepoints legitimately have different counts of stacks per
    timepoint, so we don't compare counts across dirs — only flag empty
    dirs and 0-byte files. File-size only — does not open or decode any image.
    """
    if not track_dirs:
        return

    print("\n[precheck] Checking tracking dirs for empty files...")
    track_stats = {}
    for d in track_dirs:
        tifs = sorted(d.glob("*.tif"))
        sizes = [t.stat().st_size for t in tifs]
        empty = [t.name for t, s in zip(tifs, sizes) if s == 0]
        total_gb = sum(sizes) / 1e9
        track_stats[d.name] = {
            "path": d,
            "n_tifs": len(tifs),
            "empty": empty,
            "total_gb": total_gb,
        }
        print(f"  {d.name}: {len(tifs)} tifs, {len(empty)} empty, {total_gb:.1f} GB")

    no_tifs = {n: s for n, s in track_stats.items() if s["n_tifs"] == 0}
    with_empty = {n: s for n, s in track_stats.items() if s["empty"]}

    if not no_tifs and not with_empty:
        print(f"[precheck] OK: all {len(track_dirs)} tracking dirs have non-empty tifs.")
        return

    lines = []
    lines.append("=" * 78)
    lines.append("PRECHECK FAILED — refusing to submit convert jobs (tracking)")
    lines.append("=" * 78)
    lines.append(f"Found {len(track_dirs)} tracking dirs.")
    lines.append("")
    lines.append("Per-dir summary:")
    for name, s in track_stats.items():
        marker = "OK " if s["n_tifs"] > 0 and not s["empty"] else "BAD"
        lines.append(
            f"  [{marker}] {name}: {s['n_tifs']} tifs, "
            f"{len(s['empty'])} empty, {s['total_gb']:.1f} GB"
        )
        lines.append(f"        path: {s['path']}")

    if no_tifs:
        lines.append("")
        lines.append("Tracking dirs with NO tiffs:")
        for name, s in no_tifs.items():
            lines.append(f"  - {name}: {s['path']}")

    if with_empty:
        lines.append("")
        lines.append("Tracking dirs with EMPTY (0-byte) tiffs:")
        for name, s in with_empty.items():
            preview = ", ".join(s["empty"][:5])
            more = f" ... (+{len(s['empty']) - 5} more)" if len(s["empty"]) > 5 else ""
            lines.append(f"  - {name}: {len(s['empty'])} empty file(s): {preview}{more}")

    lines.append("")
    lines.append("Likely cause: imaging on the dragonfly did not finish for these")
    lines.append("timepoints, or files failed to copy. Re-image / re-copy before")
    lines.append("converting, OR pass --skip-precheck to convert partial data.")
    lines.append("=" * 78)

    raise RuntimeError("\n" + "\n".join(lines))


def _configured_well_tokens(experiment: str) -> set | None:
    """Well tokens (e.g. {'A1', 'A2'}) from the experiment config's
    wells_to_process, or None if it isn't set (→ convert all wells)."""
    import yaml
    from cyclops_utils.data.experiment import OpsDataset

    try:
        cfg_path = OpsDataset(experiment).config_paths.get("exp_config")
        if not (cfg_path and Path(cfg_path).exists()):
            return None
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        wells = cfg.get("wells_to_process")
        if not wells:
            return None
        tokens = set()
        for w in wells:
            parts = [p for p in str(w).split("/") if p]
            if len(parts) >= 2:
                tokens.add(f"{parts[0]}{parts[1]}")
        return tokens or None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw OPS tiffs to zarr on fast partition via SLURM"
    )
    parser.add_argument("experiment", help="Experiment number or name (e.g. 146, OPS0146)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted")
    parser.add_argument("--no-wait", action="store_true", help="Submit jobs and return immediately")
    parser.add_argument("--force", action="store_true", help="Delete existing partial zarrs and reconvert")
    parser.add_argument("--only", nargs="+", default=None, help="Only convert datasets matching these substrings (e.g. A3 tracking_1)")
    parser.add_argument("--skip-precheck", action="store_true", help="Skip pheno-well tiff count/size precheck")
    parser.add_argument(
        "--recover-partial",
        action="store_true",
        help=(
            "Workaround for interrupted dragonfly acquisitions: move 0-byte "
            ".tif files aside before opening NDTiff, and skip positions whose "
            "data lived in those missing stacks (instead of failing the whole "
            "well). Writes a .missing_positions.txt manifest next to the zarr. "
            "Implies --skip-precheck."
        ),
    )
    args = parser.parse_args()

    from cyclops_utils.data.filesystem import resolve_experiment_name

    experiment = resolve_experiment_name(args.experiment, allow_interactive=True)
    from cyclops_process.pipelinerunner.exceptions import PipelineHalted
    try:
        convert_raw(
            experiment,
            only=args.only,
            force=args.force,
            dry_run=args.dry_run,
            no_wait=args.no_wait,
            skip_precheck=args.skip_precheck,
            recover_partial=args.recover_partial,
        )
    except PipelineHalted as e:
        print(f"\nERROR: {e.reason}")
        sys.exit(1)


def convert_raw(
    experiment: str,
    only: list | None = None,
    force: bool = False,
    dry_run: bool = False,
    no_wait: bool = False,
    skip_precheck: bool = False,
    recover_partial: bool = False,
) -> None:
    """Convert raw OPS tiffs (pheno + tracking) to zarr on the fast partition.

    First pipeline step: produces the raw_convert/ zarrs that link_phenotyping
    and link_tracking consume. Submits per-dataset SLURM jobs and (unless
    no_wait) blocks until they finish so the downstream link steps see complete
    inputs.
    """
    from cyclops_utils.data.experiment import OpsDataset
    from cyclops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    # --recover-partial implies --skip-precheck (the partial data is what's being recovered)
    if recover_partial:
        skip_precheck = True

    dataset = OpsDataset(experiment)

    src = dataset.lc_dragonfly_dir
    out = dataset.experiment_path_fast / "0-convert" / "live_imaging" / "raw_convert"

    if not src.exists():
        print(f"ERROR: Source not found: {src}")
        sys.exit(1)

    print(f"Experiment: {experiment}")
    print(f"Source:     {src}")
    print(f"Output:     {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Discover datasets (check root and OPS_NUM_1/ subdirectory)
    ops_subdir = src / f"{experiment.split('_')[0].upper()}_1"
    search_dirs = [src]
    if ops_subdir.exists():
        search_dirs.append(ops_subdir)

    pheno_dirs, track_dirs = [], []
    for d in search_dirs:
        pheno_dirs.extend(sorted(d.glob("phenotyping_well_A*_1")))
        track_dirs.extend(sorted([
            t for t in d.glob("tracking_*")
            if t.is_dir() and "position_list" not in t.name
        ]))

    # Skip pheno wells absent from the experiment config's wells_to_process
    # (when set). Tracking dirs are per-timepoint (all wells), so they're not
    # filtered here — well selection for tracking happens in link_tracking.
    well_tokens = _configured_well_tokens(experiment)
    if well_tokens is not None and pheno_dirs:
        import re as _re
        kept = []
        for d in pheno_dirs:
            m = _re.search(r"phenotyping_well_([A-Za-z]+\d+)_\d+$", d.name)
            if m and m.group(1) in well_tokens:
                kept.append(d)
            else:
                print(f"  [SKIP] {d.name} — not in wells_to_process {sorted(well_tokens)}")
        pheno_dirs = kept

    if not skip_precheck:
        _precheck_pheno_wells(pheno_dirs)
        _precheck_track_dirs(track_dirs)

    if only:
        pheno_dirs = [d for d in pheno_dirs if any(f in d.name for f in only)]
        track_dirs = [d for d in track_dirs if any(f in d.name for f in only)]

    print(f"\nFound {len(pheno_dirs)} phenotyping wells, {len(track_dirs)} tracking timepoints")

    # Build jobs with separate resource profiles
    track_jobs, pheno_jobs = [], []

    def _move_to_trash(path):
        """Rename to .trash_* (instant), delete in background (shared util)."""
        from cyclops_utils.data.filesystem import async_delete_path
        trash = async_delete_path(path)
        if trash is not None:
            print(f"    Renamed to {trash.name}")

    for d in track_dirs:
        zarr_out = out / f"{d.name}.zarr"
        if zarr_out.exists():
            if force:
                print(f"  FORCE: {d.name}.zarr (moving to trash)")
                if not dry_run:
                    _move_to_trash(zarr_out)
            else:
                print(f"  SKIP: {d.name} (exists, use --force to reconvert)")
                continue
        track_jobs.append({
            "name": f"convert_{d.name}",
            "func": _convert_single,
            "kwargs": {"src_dir": str(d), "out_zarr": str(zarr_out), "recover_partial": recover_partial},
            "metadata": {"type": "track", "source": d.name},
        })

    for d in pheno_dirs:
        zarr_out = out / f"{d.name}.zarr"
        if zarr_out.exists():
            if force:
                print(f"  FORCE: {d.name}.zarr (moving to trash)")
                if not dry_run:
                    _move_to_trash(zarr_out)
            else:
                print(f"  SKIP: {d.name} (exists, use --force to reconvert)")
                continue
        pheno_jobs.append({
            "name": f"convert_{d.name}",
            "func": _convert_single,
            "kwargs": {"src_dir": str(d), "out_zarr": str(zarr_out), "recover_partial": recover_partial},
            "metadata": {"type": "pheno", "source": d.name},
        })

    # Copy position lists (always, even if all zarrs exist)
    for search_dir in search_dirs:
        for j in search_dir.glob("*position_list.json"):
            dst = out / j.name
            if not dst.exists():
                print(f"  Copying: {j.name}")
                if not dry_run:
                    shutil.copy(j, dst)

    if not track_jobs and not pheno_jobs:
        print("\nAll datasets already converted.")
    elif dry_run:
        print(f"\nWould submit {len(track_jobs)} tracking + {len(pheno_jobs)} pheno jobs:")
        for j in track_jobs + pheno_jobs:
            print(f"  {j['name']}")
    else:
        from cyclops_utils.hpc.slurm_batch_utils import wait_for_multiple_job_arrays

        job_arrays = []

        # Submit tracking and pheno as separate batches with different timeouts
        track_slurm = {
            "timeout_min": 20, "mem": "32G", "cpus_per_task": 16, "slurm_partition": "cpu,gpu",
        }
        pheno_slurm = {
            "timeout_min": 400, "mem": "200G", "cpus_per_task": 64, "slurm_partition": "cpu,gpu",
        }

        if track_jobs:
            print(f"\nSubmitting {len(track_jobs)} tracking jobs (20 min timeout)...")
            track_result = submit_parallel_jobs(
                jobs_to_submit=track_jobs,
                experiment=experiment,
                slurm_params=track_slurm,
                log_dir=f"slurm_convert_raw_logs/{experiment}",
                manifest_prefix="convert_raw_track",
                dry_run=False,
                wait_for_completion=False,
            )
            if track_result.get("success") and "submitted_jobs" in track_result:
                job_arrays.append({
                    "submitted_jobs": track_result["submitted_jobs"],
                    "base_job_id": track_result["base_job_id"],
                    "label": f"Tracking ({track_result['base_job_id']})",
                    "slurm_params": track_slurm,
                })

        if pheno_jobs:
            print(f"\nSubmitting {len(pheno_jobs)} pheno jobs (400 min timeout)...")
            pheno_result = submit_parallel_jobs(
                jobs_to_submit=pheno_jobs,
                experiment=experiment,
                slurm_params=pheno_slurm,
                log_dir=f"slurm_convert_raw_logs/{experiment}",
                manifest_prefix="convert_raw_pheno",
                dry_run=False,
                wait_for_completion=False,
            )
            if pheno_result.get("success") and "submitted_jobs" in pheno_result:
                job_arrays.append({
                    "submitted_jobs": pheno_result["submitted_jobs"],
                    "base_job_id": pheno_result["base_job_id"],
                    "label": f"Pheno ({pheno_result['base_job_id']})",
                    "slurm_params": pheno_slurm,
                })

        # Wait for both arrays in parallel
        if job_arrays and not no_wait:
            wait_result = wait_for_multiple_job_arrays(
                job_arrays=job_arrays,
                experiment=experiment,
                verbose=True,
            )
            # Hard-halt on any non-clean outcome so link steps never start on
            # incomplete raw_convert/ (PipelineHalted stops the DAG cleanly; a
            # plain Exception is skip-on-120s-timeout in unattended runs).
            from cyclops_process.pipelinerunner.exceptions import PipelineHalted
            if wait_result.get("interrupted"):
                raise PipelineHalted(
                    "raw conversion wait was interrupted before completion — "
                    "halting before link steps"
                )
            failed_jobs = [
                name
                for res in wait_result.get("array_results", {}).values()
                for name in res.get("failed", [])
            ]
            if failed_jobs or not wait_result.get("all_completed", False):
                joined = ", ".join(failed_jobs) if failed_jobs else "unknown"
                raise PipelineHalted(
                    f"{len(failed_jobs)} raw-convert job(s) failed ({joined}); "
                    f"raw_convert/ is incomplete — halting before link steps"
                )
        elif not job_arrays:
            print("\nNo jobs were submitted.")

    print(f"\nDone. Output: {out}")


if __name__ == "__main__":
    main()
