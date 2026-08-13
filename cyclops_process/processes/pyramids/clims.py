"""Contrast-limit (clims) computation written in place to pyramid stores.

Store-writing driver on top of the channel_clims math library. Re-exported
by build_dask so existing `from ...build_dask import build_clims_in_place` works."""
import json
import logging
import zarr
from pathlib import Path
from typing import Optional, Sequence
from tqdm import tqdm
from iohub import open_ome_zarr
from cyclops_process.napari.dask.channel_clims import match_profile, compute_position_clims
from ops_utils.io.zarr_utils import _iter_position_paths, write_component_attrs
from ops_utils.data.filesystem import vprintf

# Print only one clims table per run (module-global)
CLIMS_REPORT_PRINTED: bool = False


def build_clims_in_place(
    source_store: str | Path,
    positions: Optional[Sequence[str]] = None,
    scale_factor: int = 2,
) -> Path:
    """
    Compute and write per-level contrast limits (contrast_limits in .zattrs) for existing pyramid levels.

    Channel-type detection and all clim parameters are centralised in
    ``cyclops_process.napari.dask.channel_clims`` (``CHANNEL_PROFILES``).
    """
    source_store = Path(source_store)

    vprintf(
        "Starting per-level clims build (no pyramid rebuild): store=%s",
        str(source_store),
    )

    def process_single_position(pos_path: str) -> None:
        try:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
        except Exception:
            pass

        with open_ome_zarr(source_store, mode="r+") as store_local:
            fov = store_local[pos_path]
            level_names = [
                k for k in getattr(fov, "array_keys", lambda: [])() if str(k).isdigit()
            ]
            if not level_names:
                vprintf("No numeric levels under %s; skipping clims build", pos_path)
                return

            # Determine channels
            try:
                c_dim = fov.data.shape[1] if fov.data.ndim >= 2 else 1
            except Exception:
                c_dim = 1
            levels_sorted = sorted([int(l) for l in level_names])

            # Resolve channel names
            try:
                with open_ome_zarr(source_store, mode="r") as _store_names:
                    _fov_names = _store_names[pos_path]
                    ch_names = list(
                        getattr(_store_names, "channel_names", None)
                        or getattr(_fov_names, "channel_names", None)
                        or []
                    )
            except Exception:
                ch_names = []

            # Pad channel names to c_dim so indexing is safe
            while len(ch_names) < int(c_dim):
                ch_names.append("")

            # Delegate all sampling, computation, and per-level scaling to channel_clims
            per_level_per_channel = compute_position_clims(
                fov=fov,
                pos_path=pos_path,
                source_store=source_store,
                channel_names=ch_names[:int(c_dim)],
                levels_sorted=levels_sorted,
                scale_factor=float(scale_factor if scale_factor and scale_factor > 0 else 2.0),
            )

            # Build method description from matched profiles
            profile_names = [match_profile(ch_names[c]).name for c in range(int(c_dim))]
            method_str = "channel_clims:" + "+".join(sorted(set(profile_names)))

            # Detect zarr format to write clims in the correct location
            zarr_fmt = detect_zarr_format(source_store)

            if zarr_fmt == 3:
                # V3: write clims_per_level dict to position-level zarr.json attributes
                # This is where read_per_level_clims expects to find them
                clims_per_level_dict = {}
                for lvl in levels_sorted:
                    per_ch_lvl = per_level_per_channel.get(int(lvl), [])
                    per_lvl = per_ch_lvl[0] if per_ch_lvl and per_ch_lvl[0] is not None else None
                    lvl_entry = {
                        "contrast_limits_method": method_str,
                    }
                    if per_lvl is not None:
                        lvl_entry["contrast_limits"] = [float(per_lvl[0]), float(per_lvl[1])]
                    lvl_entry["contrast_limits_per_channel"] = [
                        ([float(v[0]), float(v[1])] if v is not None else None)
                        for v in per_ch_lvl
                    ]
                    clims_per_level_dict[str(lvl)] = lvl_entry

                try:
                    pos_dir = source_store / Path(pos_path)
                    zarr_json_path = pos_dir / "zarr.json"
                    if zarr_json_path.exists():
                        with open(zarr_json_path, "r") as f:
                            pos_meta = json.load(f)
                    else:
                        pos_meta = {"zarr_format": 3, "node_type": "group", "attributes": {}}
                    pos_meta.setdefault("attributes", {})
                    pos_meta["attributes"]["clims_per_level"] = clims_per_level_dict
                    with open(zarr_json_path, "w") as f:
                        json.dump(pos_meta, f, indent=2)
                except Exception:
                    pass
            else:
                # V2: write to each pyramid level's .zattrs
                for lvl in levels_sorted:
                    per_ch_lvl = per_level_per_channel.get(int(lvl), [])
                    per_lvl = per_ch_lvl[0] if per_ch_lvl and per_ch_lvl[0] is not None else None
                    try:
                        lvl_dir = source_store / Path(pos_path) / str(lvl)
                        updates = {
                            "contrast_limits_method": method_str,
                        }
                        if per_lvl is not None:
                            updates["contrast_limits"] = [float(per_lvl[0]), float(per_lvl[1])]
                        updates["contrast_limits_per_channel"] = [
                            ([float(v[0]), float(v[1])] if v is not None else None)
                            for v in per_ch_lvl
                        ]
                        write_component_attrs(lvl_dir, updates)
                    except Exception:
                        pass

            vprintf(
                "build-clim: %s levels=%s",
                pos_path,
                levels_sorted,
            )

            # PrettyTable report (first position only)
            global CLIMS_REPORT_PRINTED
            if not CLIMS_REPORT_PRINTED:
                try:
                    from prettytable import PrettyTable  # type: ignore

                    tbl = PrettyTable(
                        ["position", "level", "ch", "channel_name", "profile", "lo", "hi"]
                    )
                    for lvl in levels_sorted:
                        ch_list = per_level_per_channel.get(int(lvl), [])
                        for c in range(int(c_dim)):
                            v = ch_list[c] if c < len(ch_list) else None
                            lo_str = f"{v[0]:.6g}" if v is not None else "-"
                            hi_str = f"{v[1]:.6g}" if v is not None else "-"
                            prof = match_profile(ch_names[c])
                            tbl.add_row([
                                str(pos_path), int(lvl), int(c),
                                str(ch_names[c]), prof.name,
                                lo_str, hi_str,
                            ])
                    print(tbl)
                except Exception:
                    print(f"CLIMS report for {pos_path}:")
                    for lvl in levels_sorted:
                        ch_list = per_level_per_channel.get(int(lvl), [])
                        rows = []
                        for c in range(int(c_dim)):
                            v = ch_list[c] if c < len(ch_list) else None
                            prof = match_profile(ch_names[c])
                            rows.append(
                                f"C{c} {ch_names[c]}({prof.name})=[{v[0]:.6g},{v[1]:.6g}]"
                                if v is not None
                                else f"C{c} {ch_names[c]}=[-,-]"
                            )
                        print(f"  L{int(lvl)}: " + "  ".join(rows))
                CLIMS_REPORT_PRINTED = True

    # Discover positions
    pos_paths = positions or _iter_position_paths(source_store)
    vprintf("Found %d positions for clims update", len(pos_paths))

    # Use threads to share Zarr cache and reduce process startup overhead; cap workers modestly
    num_workers = max(1, min(4, get_optimal_workers(use_gpu=False)))
    print(f"Building clims with {num_workers} thread worker(s)")
    vprintf("Building clims with %d thread worker(s)", num_workers)
    Parallel(n_jobs=num_workers, prefer="threads")(
        delayed(process_single_position)(pos)
        for pos in tqdm(pos_paths, desc="Positions (clims)")
    )

    return source_store
