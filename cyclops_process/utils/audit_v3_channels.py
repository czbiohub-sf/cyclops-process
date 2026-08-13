#!/usr/bin/env python3
"""Audit v3 pheno stores: compare channels against ops_channel_maps.yaml.

Flags experiments whose v3 store is missing expected fluorescent channels.

Usage:
    python -m cyclops_process.utils.audit_v3_channels
    python -m cyclops_process.utils.audit_v3_channels --fix  # print fix commands
"""
import json
import yaml
from pathlib import Path

from cyclops_utils.data.experiment import OpsDataset
from cyclops_process.paths import BASE_PATH


def load_channel_map() -> dict[str, list[str]]:
    """Load ops_channel_maps.yaml, return {experiment_prefix: [channel_names]}."""
    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    channel_map_path = configs_dir / "ops_channel_maps.yaml"
    with open(channel_map_path) as f:
        raw = yaml.safe_load(f)
    result = {}
    for exp, channels in raw.items():
        if not isinstance(channels, list):
            continue
        names = []
        for ch in channels:
            if isinstance(ch, dict) and "channel_name" in ch:
                names.append(ch["channel_name"])
        if names:
            result[exp] = names
    return result


def get_v3_channel_names(v3_path: Path) -> list[str] | None:
    """Read channel names from the first position's zarr.json in a v3 store."""
    for zj in sorted(v3_path.glob("A/*/*/zarr.json")):
        meta = json.loads(zj.read_text())
        omero = meta.get("attributes", {}).get("ome", {}).get("omero", {})
        return [ch["label"] for ch in omero.get("channels", [])]
    return None


def get_v2_channel_names(v2_path: Path) -> list[str] | None:
    """Read channel names from the first position's .zattrs in a v2 store."""
    for za in sorted(v2_path.glob("A/*/*/.zattrs")):
        attrs = json.loads(za.read_text())
        return [ch["label"] for ch in attrs.get("omero", {}).get("channels", [])]
    return None


def get_v2_array_shape(v2_path: Path) -> tuple | None:
    """Read the actual array shape from the v2 store to get true channel count."""
    import zarr
    for pos_dir in sorted(v2_path.glob("A/*/*")):
        arr_path = pos_dir / "0"
        if arr_path.exists():
            try:
                arr = zarr.open(str(arr_path), mode="r")
                return tuple(arr.shape)
            except Exception:
                pass
    return None


# Map from raw channel_name -> expected labels in assembled store
FLUOR_CHANNEL_NAMES = {"GFP", "mCherry"}


def check_missing_fluor_channels(experiment: str, v3_pheno_path: Path) -> list[str] | None:
    """Check if a v3 pheno store is missing expected fluorescent channels.

    Returns list of missing channel names, empty list if all present,
    or None if no fluorescence is expected for this experiment.
    """
    channel_map = load_channel_map()
    exp_prefix = experiment.split("_")[0]
    expected_raw = channel_map.get(exp_prefix, [])
    expected_fluor = [ch for ch in expected_raw if ch in FLUOR_CHANNEL_NAMES]
    if not expected_fluor:
        return None
    v3_channels = get_v3_channel_names(v3_pheno_path)
    if v3_channels is None:
        return expected_fluor
    return [ch for ch in expected_fluor if ch not in v3_channels]


def audit():
    import argparse
    parser = argparse.ArgumentParser(description="Audit v3 pheno store channels")
    parser.add_argument("--fix", action="store_true", help="Print fix commands")
    args = parser.parse_args()

    channel_map = load_channel_map()
    base_dir = Path(f"{BASE_PATH}")

    ok = []
    mismatched = []
    no_store = []
    no_fluor_expected = []

    for exp_prefix, expected_raw_channels in sorted(channel_map.items()):
        expected_fluor = [ch for ch in expected_raw_channels if ch in FLUOR_CHANNEL_NAMES]
        if not expected_fluor:
            no_fluor_expected.append(exp_prefix)
            continue

        # Find experiment directory
        matches = sorted(base_dir.glob(f"{exp_prefix}_*"))
        if not matches:
            continue

        for exp_dir in matches:
            experiment = exp_dir.name
            try:
                dataset = OpsDataset(experiment)
            except Exception:
                continue

            v3_path = dataset.store_paths.get("pheno_assembled_v3")
            v2_path = dataset.store_paths.get("pheno_assembled")

            if not v3_path or not Path(v3_path).exists():
                no_store.append(experiment)
                continue

            v3_channels = get_v3_channel_names(Path(v3_path))
            if v3_channels is None:
                no_store.append(experiment)
                continue

            # Check if expected fluor channels are present in v3
            missing_fluor = [ch for ch in expected_fluor if ch not in v3_channels]

            # Also check v2 metadata vs actual array shape
            v2_meta_channels = get_v2_channel_names(Path(v2_path)) if v2_path else None
            v2_shape = get_v2_array_shape(Path(v2_path)) if v2_path else None
            v2_n_meta = len(v2_meta_channels) if v2_meta_channels else "?"
            v2_n_array = v2_shape[1] if v2_shape and len(v2_shape) >= 2 else "?"

            if missing_fluor:
                mismatched.append({
                    "experiment": experiment,
                    "expected_fluor": expected_fluor,
                    "missing_fluor": missing_fluor,
                    "v3_channels": v3_channels,
                    "v2_meta_channels": v2_n_meta,
                    "v2_array_channels": v2_n_array,
                })
            else:
                ok.append(experiment)

    # Summary
    print("=" * 70)
    print("  V3 PHENO STORE FLUORESCENCE CHANNEL AUDIT")
    print("=" * 70)
    print(f"\n  OK: {len(ok)} experiments have correct fluor channels")
    print(f"  No fluor expected: {len(no_fluor_expected)} experiments (BF-only)")
    print(f"  No v3 store: {len(no_store)} experiments")
    print(f"  MISMATCHED: {len(mismatched)} experiments")

    if mismatched:
        print(f"\n{'=' * 70}")
        print("  EXPERIMENTS WITH MISSING FLUORESCENT CHANNELS")
        print(f"{'=' * 70}\n")
        for m in mismatched:
            print(f"  {m['experiment']}")
            print(f"    expected fluor: {m['expected_fluor']}")
            print(f"    missing from v3: {m['missing_fluor']}")
            print(f"    v3 channels: {m['v3_channels']}")
            print(f"    v2 metadata channels: {m['v2_meta_channels']}, v2 array C dim: {m['v2_array_channels']}")
            print()

    if args.fix and mismatched:
        print(f"{'=' * 70}")
        print("  FIX COMMANDS")
        print(f"{'=' * 70}\n")
        for m in mismatched:
            exp = m["experiment"]
            print(f"  # {exp} — missing {m['missing_fluor']}")
            print(f"  # 1. Rebuild unified tiles (with fluor)")
            print(f"  python -m cyclops_process.processes.run {exp} -ss --rerun prepare_unified_pheno_tiles")
            print(f"  # 2. Re-stitch pheno assembled")
            print(f"  python -m cyclops_process.processes.run {exp} -ss --rerun estimate_and_stitch_pheno")
            print(f"  # 3. Re-convert v3 base images")
            print(f"  python -m cyclops_process.convert.v3_livecell --experiment {exp} --mode pheno --force-base")
            print()

    print(f"{'=' * 70}")


if __name__ == "__main__":
    audit()
