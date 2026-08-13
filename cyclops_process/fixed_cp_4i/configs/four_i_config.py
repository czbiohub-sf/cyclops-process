"""4i experiment configuration.

Central config for the 4i immunofluorescence pipeline. Defines round-to-directory
mappings, channel definitions, and antibody metadata for the 20260318_4i_re-run
acquisition.

Each round has 3 channels:
  - Channel 0: DAPI (nuclei)
  - Channel 1: mouse-488 antibody
  - Channel 2: rabbit-647 antibody
"""

from __future__ import annotations

from pathlib import Path

from cyclops_process import paths

# ============================================================================
# Associated OPS experiment
# ============================================================================
EXPERIMENT = "ops0144_20260406"

# ============================================================================
# Instrument / data root
# ============================================================================
# Acquisition subdirectory for this 4i run. The mount it sits under is supplied by
# OPS_INSTRUMENT_ROOT and resolved on demand — see cyclops_process/paths.py.
ACQUISITION_DIR = "20260318_4i_re-run"


def default_instrument_root() -> Path:
    """Acquisition root for this run, under the configured instrument mount."""
    return Path(paths.instrument_root()) / ACQUISITION_DIR

# ============================================================================
# Round definitions
# ============================================================================
# Each round maps to: (subdirectory name, final acquisition subfolder, antibody info)
# The "final" subfolder is the one with the complete dataset (45 NDTiffStack files).

ROUNDS: dict[int, dict] = {
    # NOTE: For R1, R3, R4, R5 the 488 and 647 channels were acquired in reverse
    # order, so in the zarr the physical slot-1 holds the 647-captured pixels
    # (rabbit antibody) and slot-2 holds the 488-captured pixels (mouse antibody).
    # mouse_488 / rabbit_647 still name which antibody went on which secondary;
    # swap_488_647=True flags the acquisition-order swap so FOUR_I_CHANNELS
    # assigns the 647 metadata to slot-1 and the 488 metadata to slot-2.
    # R2 was acquired in the expected [DAPI, 488, 647] order.
    1: {
        "dir": "20260318_round1_p53_mse488_gH2AX_rab647",
        "final": "all_wells_final_2",
        "mouse_488": "p53",
        "rabbit_647": "gH2AX",
        "swap_488_647": True,
    },
    2: {
        "dir": "20260320_round2_c-Myc_mse488_RSP6_rab647",
        "final": "final_2",
        "mouse_488": "c-Myc",
        "rabbit_647": "RPS6",
        "swap_488_647": False,
    },
    3: {
        "dir": "20260323_round3_Rb_mse488_pRb_rab647",
        "final": "final_1",
        "mouse_488": "Rb",
        "rabbit_647": "pRb",
        "swap_488_647": True,
    },
    4: {
        "dir": "20260325_round4_b-catenin_mse488_p21_rab647",
        "final": "final_1",
        "mouse_488": "b-catenin",
        "rabbit_647": "p21",
        "swap_488_647": True,
    },
    5: {
        "dir": "20260327_round5_NFkB_mse488_pS6_rab647",
        "final": "final_3",
        "mouse_488": "NFkB",
        "rabbit_647": "pS6",
        "swap_488_647": True,
    },
}

NUM_ROUNDS = len(ROUNDS)

# ============================================================================
# Channel definitions per round
# ============================================================================
# Channel index → name within each round's zarr store
# (actual names depend on Micro-Manager config; these are semantic labels)
CHANNEL_MAP = {
    0: "DAPI",
    1: "mouse_488",
    2: "rabbit_647",
}

# ============================================================================
# 4i channel metadata for v3 aggregation (analogous to CELL_PAINTING_CHANNELS)
# ============================================================================

# Match CP format: name = short identifier, marker = antibody/dye, structure = full biological feature
# full_marker = full conjugate name (for antibodies, includes species+fluorophore)
FOUR_I_CHANNELS: dict[int, dict[int, dict]] = {}
for rnd, info in ROUNDS.items():
    _488 = {
        "name": info["mouse_488"],
        "marker": info["mouse_488"],
        "structure": info["mouse_488"],
        "full_marker": f"{info['mouse_488']} (mouse-488)",
    }
    _647 = {
        "name": info["rabbit_647"],
        "marker": info["rabbit_647"],
        "structure": info["rabbit_647"],
        "full_marker": f"{info['rabbit_647']} (rabbit-647)",
    }
    slot1, slot2 = (_647, _488) if info.get("swap_488_647", False) else (_488, _647)
    FOUR_I_CHANNELS[rnd] = {
        0: {"name": "nuclei", "marker": "DAPI", "structure": "Nucleus", "full_marker": "DAPI"},
        1: slot1,
        2: slot2,
    }

# Color map keyed by marker name (matches CP_COLORS pattern)
FOUR_I_COLORS = {
    "DAPI": "0000FF",  # Blue
}
# Dynamically assign colors for antibody markers (green for 488, red for 647)
for rnd, info in ROUNDS.items():
    FOUR_I_COLORS[info["mouse_488"]] = "00FF00"   # Green
    FOUR_I_COLORS[info["rabbit_647"]] = "FF0000"  # Red

# ============================================================================
# Helper functions
# ============================================================================


def get_default_output_dir(experiment: str | None = None) -> Path:
    """Return the default 0-convert/four_i/ output directory for the experiment."""
    from ops_utils.data.experiment import OpsDataset
    dataset = OpsDataset(experiment or EXPERIMENT)
    return dataset.experiment_path / "0-convert" / "4i"


def get_round_input_dir(round_num: int, instrument_root: Path | None = None) -> Path:
    """Return the path to the final acquisition directory for a given round."""
    root = instrument_root or default_instrument_root()
    info = ROUNDS[round_num]
    return root / info["dir"] / info["final"]


def get_all_round_input_dirs(instrument_root: Path | None = None) -> dict[int, Path]:
    """Return {round_num: input_dir} for all rounds."""
    return {r: get_round_input_dir(r, instrument_root) for r in ROUNDS}


def get_channel_name(round_num: int, channel: int, format: str = "short") -> str:
    """Get channel name for a given round and channel index.

    Matches the CP naming pattern:
      short: 4i_R<n>_<name>   (e.g., 4i_R1_nuclei, 4i_R1_p53)
      full:  4i_R<n>_<name>_<marker>  (e.g., 4i_R1_nuclei_DAPI)
             Falls back to short form when name == marker to avoid redundancy.
    """
    info = FOUR_I_CHANNELS.get(round_num, {}).get(channel, {})
    name = info.get("name", f"ch{channel}")
    marker = info.get("marker", "")
    if format == "short":
        return f"4i_R{round_num}_{name}"
    elif format == "full":
        if name == marker:
            return f"4i_R{round_num}_{name}"
        return f"4i_R{round_num}_{name}_{marker}"
    return f"4i_R{round_num}_{name}"
