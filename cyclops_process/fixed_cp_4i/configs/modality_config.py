"""Modality configuration for the unified fixed-cell pipeline (cell painting + 4i).

Single source of truth for everything that differs between cell painting and 4i.
The pipeline logic is identical; only these settings change:

  - unit vocabulary: CP has "parts" (2), 4i has "rounds" (5)
  - channel naming:  CP{n}_<name>[_<marker>]   vs   4i_R{n}_<name>[_<marker>]
  - panel channel definitions (per unit, per channel index)
  - nuclei channel name (segmented per unit -> {unit}_nuclear_seg)
  - register-YAML names (chained: unit1 -> pheno, unitN -> unit1)
  - cell-seg label

4i's per-experiment specifics (instrument dirs, antibody swaps) stay in
four_i_config.py; this module wraps them so callers use one interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ── Cell-painting panel channels (per part) ──────────────────────────────────
# Mirrors convert_v3_slurm_cp.CELL_PAINTING_CHANNELS; kept here so this config is
# the one place the CP panel is defined for the unified pipeline.
CELL_PAINTING_CHANNELS: dict[int, dict[int, dict]] = {
    1: {
        0: {"name": "nuclei", "marker": "Hoechst", "structure": "Nucleus"},
        1: {"name": "mitochondria", "marker": "TOMM20", "structure": "Mitochondria"},
        2: {"name": "plasma_membrane", "marker": "WGA", "structure": "Plasma Membrane"},
        3: {"name": "f_actin", "marker": "Phalloidin", "structure": "F-actin"},
    },
    2: {
        0: {"name": "nuclei", "marker": "Hoechst", "structure": "Nucleus"},
        1: {"name": "nucleoli", "marker": "NPM1", "structure": "Nucleoli"},
        2: {"name": "microtubules", "marker": "Tubulin", "structure": "Microtubules"},
        3: {"name": "ER", "marker": "ConA", "structure": "Endoplasmic Reticulum"},
    },
}


@dataclass(frozen=True)
class Modality:
    """Everything that differs between CP and 4i for one modality."""
    name: str                       # "cp" | "4i"
    unit_word: str                  # "part" | "round"
    unit_prefix: str                # "CP" | "4i_R"  (channel/label name prefix)
    default_units: list[int]        # [1, 2] | [1, 2, 3, 4, 5]
    cell_seg_label: str             # "cp_cell_seg" | "4i_cell_seg"
    register_kind: str              # token used in the register YAML name
    convert_subdir: str             # 0-convert/<subdir>: "cell_painting" | "4i"
    store_stem_word: str            # per-unit zarr stem word: "part" | "round"
    register_dir_rel: str           # register-YAML dir, relative to experiment_path
    full_res_tile_size: int         # camera tile size (px) for seg tile geometry
    # channel definitions keyed by unit -> {channel_idx: {name, marker, structure}}
    channels: dict[int, dict[int, dict]]

    def unit_stem(self, unit: int) -> str:
        """Raw per-unit convert store stem, e.g. 'part1' / 'round1' (-> <stem>.zarr)."""
        return f"{self.store_stem_word}{unit}"

    def convert_dir(self, experiment: str):
        """<experiment>/0-convert/<convert_subdir> — where per-unit zarrs live."""
        from cyclops_utils.data.experiment import OpsDataset
        return OpsDataset(experiment).experiment_path / "0-convert" / self.convert_subdir

    def seg_store_path(self, experiment: str, unit: int):
        """5x nuclear-seg store used as the registration source for this unit:
        <convert_dir>/<unit_stem>_max_proj_flatfield_segmentation.zarr."""
        return self.convert_dir(experiment) / f"{self.unit_stem(unit)}_max_proj_flatfield_segmentation.zarr"

    def register_dir(self, experiment: str):
        """Directory where this modality's chained register YAMLs are written."""
        from cyclops_utils.data.experiment import OpsDataset
        return OpsDataset(experiment).experiment_path / self.register_dir_rel

    # ---- naming helpers (identical logic, prefix differs) ----
    def channel_name(self, unit: int, ch: int, fmt: str = "short") -> str:
        info = self.channels.get(unit, {}).get(ch, {})
        nm = info.get("name", f"ch{ch}")
        marker = info.get("marker", "")
        base = f"{self.unit_prefix}{unit}_{nm}"
        return base if (fmt == "short" or not marker) else f"{base}_{marker}"

    def nuclei_channel(self, unit: int) -> str:
        """Full name of the nuclei channel for a unit (channel 0), e.g.
        CP1_nuclei_Hoechst / 4i_R1_nuclei_DAPI."""
        return self.channel_name(unit, 0, fmt="full")

    def nuclear_seg_label(self, unit: int) -> str:
        """Per-unit nuclear-seg label written into the v3 store."""
        return f"{self.unit_prefix}{unit}_nuclear_seg"

    def register_yaml_name(self, well, unit: int) -> str:
        """Chained register YAML: {row}{col}_{register_kind}{unit}_register.yml.
        CP: A1_cell_painting{n}_register.yml ; 4i: B2_4i_round{n}_register.yml."""
        from cyclops_utils.data.filesystem import parse_well
        row, col = parse_well(well)
        return f"{row}{col}_{self.register_kind}{unit}_register.yml"


def _four_i_channels() -> dict[int, dict[int, dict]]:
    """Load 4i channel defs from the experiment-specific four_i_config (lazy)."""
    from cyclops_process.fixed_cp_4i.configs.four_i_config import FOUR_I_CHANNELS
    return FOUR_I_CHANNELS


MODALITIES: dict[str, Modality] = {
    "cp": Modality(
        name="cp",
        unit_word="part",
        unit_prefix="CP",
        default_units=[1, 2],
        cell_seg_label="cp_cell_seg",
        register_kind="cell_painting",
        convert_subdir="cell_painting",
        store_stem_word="part",
        register_dir_rel="2-tracking",
        full_res_tile_size=2048,
        channels=CELL_PAINTING_CHANNELS,
    ),
    "4i": Modality(
        name="4i",
        unit_word="round",
        unit_prefix="4i_R",
        default_units=[1, 2, 3, 4, 5],
        cell_seg_label="4i_cell_seg",
        register_kind="4i_round",
        convert_subdir="4i",
        store_stem_word="round",
        register_dir_rel="0-convert/4i/registration",
        full_res_tile_size=2304,
        channels=_four_i_channels(),
    ),
}


def get_modality(name: str) -> Modality:
    if name not in MODALITIES:
        raise ValueError(f"Unknown modality {name!r}; expected one of {sorted(MODALITIES)}")
    return MODALITIES[name]
