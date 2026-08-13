#!/usr/bin/env python3
"""Orchestrate the full fixed-cell pipeline (cell painting + 4i) for an experiment.

One command runs every step in dependency order instead of invoking each by hand.
The two modalities share this orchestrator; per-experiment settings (orientation,
stitch channel/overlap) live in PIPELINE_PARAMS and modality naming in
modality_config.

    convert -> project(+flatfield, orientation baked in) -> stitch ->
    segment_5x (registration input) -> register -> convert_v3 ->
    segment_20x (CP{N}/R{N}_nuclear_seg + cell_seg, all concurrent) -> link

Orientation (flip/rot90) is applied ONCE in project, so stitch/segment run on
already-oriented data (identity). Discover a new experiment's orientation +
channel/overlap with sweep_stitch_params, then record them in PIPELINE_PARAMS.

Each step is a SLURM-submitting CLI that blocks until its jobs finish; this
sequences them and halts on the first failure. A step with multiple independent
commands (segment_20x) runs them concurrently.

Usage:
    python -m cyclops_process.fixed_cp_4i.run_pipeline -e ops0174           # cp (default)
    python -m cyclops_process.fixed_cp_4i.run_pipeline -e ops0144 --modality 4i
    python -m cyclops_process.fixed_cp_4i.run_pipeline -e ops0174 --from convert_v3
    python -m cyclops_process.fixed_cp_4i.run_pipeline -e ops0174 --steps stitch segment_5x --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from cyclops_process.fixed_cp_4i.configs.modality_config import get_modality

STEP_ORDER = ["convert", "project", "stitch", "segment_5x",
              "register", "convert_v3", "segment_20x", "link"]

# ── Per-experiment settings (orientation + stitch) ──────────────────────────
# orient: applied in project (flip/rot90) — the single orientation point.
# stitch: registration channel index + tile overlap (px) + CLAHE.
# Find these with `sweep_stitch_params`; record the winner here so they're durable.
PIPELINE_PARAMS = {
    "ops0174": dict(modality="cp", orient=dict(rot90=1),
                    stitch=dict(channel=3, overlap=100, use_clahe=True)),
    "ops0094": dict(modality="cp", orient=dict(fliplr=True),
                    stitch=dict(channel=1, overlap=75, use_clahe=True)),
    "ops0144": dict(modality="4i", orient=dict(),
                    stitch=dict(channel=1, overlap=75, use_clahe=True)),
}


def _params_for(experiment: str, modality_override: str | None) -> dict:
    import re
    m = re.search(r"ops\d{4}", experiment, re.IGNORECASE)
    key = m.group(0).lower() if m else None
    p = dict(PIPELINE_PARAMS.get(key, {}))
    if modality_override:
        p["modality"] = modality_override
    p.setdefault("modality", "cp")
    p.setdefault("orient", {})
    p.setdefault("stitch", dict(channel=0, overlap=100, use_clahe=True))
    return p


def _mod(name: str) -> list[str]:
    return [sys.executable, "-m", f"cyclops_process.fixed_cp_4i.{name}"]


def _pymod(dotted: str) -> list[str]:
    return [sys.executable, "-m", dotted]


def build_commands(experiment: str, p: dict, force_convert: bool) -> list[tuple[str, list[list[str]]]]:
    mod = p["modality"]
    m = get_modality(mod)
    units = [str(u) for u in m.default_units]
    orient, st = p["orient"], p["stitch"]
    E = ["--experiment", experiment, "--modality", mod]
    cmds: list[tuple[str, list[list[str]]]] = []

    convert = _mod("00_convert") + E + ["--parts", *units]
    if force_convert:
        convert.append("--force")
    cmds.append(("convert", [convert]))

    # project (+ flatfield); orientation baked in here.
    project = _mod("01_project") + E + ["--parts", *units]
    if orient.get("flipud"):
        project.append("--flipud")
    if orient.get("fliplr"):
        project.append("--fliplr")
    if orient.get("rot90"):
        project += ["--rot90", str(orient["rot90"])]
    cmds.append(("project", [project]))

    # stitch — data already oriented, so no flip/rot90 here.
    stitch = _mod("04_stitch") + E + ["--parts", *units,
             "--channel", str(st["channel"]), "--overlap", str(st["overlap"])]
    if st.get("use_clahe"):
        stitch.append("--use-clahe")
    cmds.append(("stitch", [stitch]))

    # 5x nuclear seg = registration input (identity orientation).
    cmds.append(("segment_5x", [_mod("05_segment") + ["segment"] + E + ["--parts", *units]]))

    cmds.append(("register", [_mod("06_register") + E]))

    cmds.append(("convert_v3", [_pymod("cyclops_process.convert.v3_fixed_cli") + E + ["--parts", *units]]))

    # 20x seg on the v3 store: per-unit nuclear seg + cell seg, all concurrent.
    seg = [
        _pymod("cyclops_process.processes.cell_seg.nuclei_segmentation_slurm")
        + ["--experiment", experiment, "--nuclei-channel", m.nuclei_channel(u),
           "--output-label", m.nuclear_seg_label(u), "--force"]
        for u in m.default_units
    ]
    cell_paint_flag = ["--cell-paint"] if mod == "cp" else ["--4i"]
    seg.append(_pymod("cyclops_process.processes.cell_seg.cell_segmentation_slurm")
               + ["--experiment", experiment] + cell_paint_flag)
    cmds.append(("segment_20x", seg))

    # link — canonical module still lives in cell_painting/ until cutover.
    link = _pymod("cyclops_process.fixed_cp_4i.link_slurm") + ["--experiment", experiment]
    if mod == "4i":
        link.append("--4i")
    cmds.append(("link", [link]))

    return cmds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", "-e", required=True)
    ap.add_argument("--modality", choices=["cp", "4i"], default=None, help="Override the modality from PIPELINE_PARAMS")
    ap.add_argument("--steps", nargs="+", choices=STEP_ORDER, default=None)
    ap.add_argument("--from", dest="from_step", choices=STEP_ORDER, default=None)
    ap.add_argument("--to", dest="to_step", choices=STEP_ORDER, default=None)
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from ops_utils.data.filesystem import resolve_experiment_name
    experiment = resolve_experiment_name(args.experiment, autoselect=True)
    p = _params_for(experiment, args.modality)

    steps = args.steps or STEP_ORDER
    if args.from_step:
        steps = STEP_ORDER[STEP_ORDER.index(args.from_step):]
    if args.to_step:
        steps = [s for s in steps if STEP_ORDER.index(s) <= STEP_ORDER.index(args.to_step)]

    all_cmds = dict(build_commands(experiment, p, args.force_convert))
    plan = [(s, all_cmds[s]) for s in STEP_ORDER if s in steps]

    print(f"\n{'='*72}\nFixed-cell pipeline: {experiment}  (modality={p['modality']})")
    print(f"orient={p['orient']}  stitch={p['stitch']}")
    print(f"steps: {[s for s, _ in plan]}\n{'='*72}")
    for step, cmd_list in plan:
        for cmd in cmd_list:
            print(f"\n>>> {step}: {' '.join(cmd[2:])}")
    if args.dry_run:
        print("\n[dry-run] nothing executed.")
        return 0

    for step, cmd_list in plan:
        print(f"\n{'='*72}\n=== STEP: {step} ({len(cmd_list)} job(s){' — parallel' if len(cmd_list) > 1 else ''}) ===\n{'='*72}", flush=True)
        if len(cmd_list) == 1:
            if subprocess.run(cmd_list[0]).returncode != 0:
                print(f"\n❌ Step '{step}' failed; halting pipeline.")
                return 1
        else:
            procs = [(c, subprocess.Popen(c)) for c in cmd_list]
            failed = [(" ".join(c[2:]), pr.wait()) for c, pr in procs]
            failed = [(lbl, rc) for lbl, rc in failed if rc != 0]
            if failed:
                for lbl, rc in failed:
                    print(f"  ❌ failed (exit {rc}): {lbl}")
                print(f"\n❌ Step '{step}' failed ({len(failed)}/{len(cmd_list)}); halting pipeline.")
                return 1
        print(f"✅ Step '{step}' complete.")

    print(f"\n🎉 Fixed-cell pipeline complete for {experiment} ({p['modality']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
