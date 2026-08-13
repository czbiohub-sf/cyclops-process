import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from typing import Dict
from datetime import datetime
from tqdm import tqdm
import dask.array as da
import yaml

from ops_utils.data.experiment import OpsDataset
from iohub.ngff import open_ome_zarr
from cyclops_process.metrics.plate_stats.match_reads import (
    _get_effective_iss_rounds,
    _parse_failed_rounds_spec,
    match_reads,
)






def _save_rounds_manifest(dataset, iss_rounds, failed_rounds_by_well, method):
    """Save ISS rounds manifest to YAML file."""
    try:
        seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
        wells = [a[0] for a in seg_store.positions()]
        manifest = {
            "experiment": dataset.experiment,
            "method": method,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_iss_rounds": list(iss_rounds),
            "failed_rounds_config": {k: list(v) if isinstance(v, list) else v for k, v in (failed_rounds_by_well or {}).items()},
            "wells": {}
        }
        for well in wells:
            dropout, offset = _parse_failed_rounds_spec(well, failed_rounds_by_well)
            effective = _get_effective_iss_rounds(iss_rounds, well, failed_rounds_by_well)
            manifest["wells"][well] = {"effective_rounds": list(effective), "dropout_rounds": list(dropout), "offset_rounds": list(offset), "num_rounds_used": len(effective)}
        with open(dataset.results_iss / "iss_rounds_manifest.yaml", "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        print(f"Could not save ISS rounds manifest: {e}")


def count_bases(experiment, iss_rounds=None, method=None) -> Dict:
    """
    measure the relative amounts of each base in reads
    - ideally should be 25/25/25/25 because the sgRNA sequence is close to random
    """
    dataset = OpsDataset(experiment, method=method)
    seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
    position_list = [a[0] for a in seg_store.positions()]

    fig, ax = plt.subplots(len(position_list), 1, figsize=(9, 9))
    for i, pos in enumerate(position_list):
        reads_df = pd.read_csv(dataset.append_well("reads", pos))
        reads = reads_df["barcode"].values

        counts = {}
        # Use actual barcode length (may be shorter than iss_rounds if rounds were excluded)
        if len(reads) == 0:
            continue
        barcode_len = len(reads[0]) if len(reads) > 0 else 0
        num_rounds = min(len(iss_rounds), barcode_len)
        for round in range(num_rounds):
            counts[round] = Counter([a[round] for a in reads if round < len(a)])

        count_df = pd.DataFrame.from_dict(counts, orient="index")

        count_frac = count_df / count_df.loc[0].sum()
        if len(position_list) == 1:
            ax = [ax]
        ax[i].plot(count_frac)
        if i == len(position_list) - 1:
            ax[i].set_xlabel("ISS round")
        ax[i].set_ylabel("Read Fraction")
        ax[i].legend(["G", "T", "A", "C"], ncol=4)
        ax[i].set_xlim(0, 9)
        ax[i].set_ylim(0.0, 1.0)
        ax[i].grid(True, linestyle="--", alpha=0.5)
        ax[i].set_title(pos)
    fig.suptitle(f"Base Fraction by Round\n{experiment}")
    plt.savefig(dataset.metrics_paths["base_frac_by_round"], dpi=300)

    return count_df


def frequency_table(reads: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a table of the number of times each barcode appears in the well.
    """
    temp = reads["barcode"].value_counts().reset_index()
    temp.columns = ["barcode", "count"]
    return temp



def count_doublets(
    experiment: str,
    iss_rounds: list[int] | None = None,
    method=None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> None:
    """
    Fraction of cells with 2 different barcodes
    - importantly assumes that doublets must have different barcodes
    - ignores cells that have 2 copies of the same barcode, this is often an artifact of stitching, and is not real

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        method: Base calling method.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    dataset = OpsDataset(experiment, method=None)
    code_db = dataset.load_codebook()
    seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
    position_list = [a[0] for a in seg_store.positions()]

    num_doublets = 0
    num_total_cells = 0
    for pos in position_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        read_db = pd.read_csv(dataset.append_well("reads", pos))

        matched_reads = match_reads(read_db, code_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
        spots_per_cell = matched_reads["cell"].value_counts()[1:]
        counts, bins = np.histogram(spots_per_cell, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 6.5])

        grouped = matched_reads.groupby("cell")
        filtered = grouped.filter(lambda x: len(x) == 2)
        result = (
            filtered.groupby("cell")["barcode"]
            .nunique()
            .reset_index(name="unique_B_count")
        )

        # Values where B is the same in both rows
        cells_with_2_diff_barcodes = result[result["unique_B_count"] == 2]
        cells_with_2_same_barcodes = result[result["unique_B_count"] == 1]

        num_doublets += len(cells_with_2_diff_barcodes)


def create_freq_tables(
    experiment: str,
    iss_rounds: list[int] | None = None,
    method: str = None,
    confidence_threshold: float = 0.95,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    """
    Calculate cell numbers for each step of the process, adapting to the base calling method.

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        method: Base calling method ('mine' or 'probabilistic').
        confidence_threshold: Confidence threshold for probabilistic method.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    dataset = OpsDataset(experiment, method=method)
    codebook_db = dataset.load_codebook()

    seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"], mode="r")
    position_list = [a[0] for a in seg_store.positions()]

    freq_dict = {}

    for pos in position_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        read_db = pd.read_csv(dataset.append_well("reads", pos))

        # Define "good" reads based on the method
        if method == "probabilistic":
            good_reads = read_db[read_db["confidence"] >= confidence_threshold].copy()
            good_reads_key = "high_confidence"
        else:  # 'mine'
            good_reads = match_reads(read_db, codebook_db, iss_rounds=well_iss_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
            if good_reads is not None:
                good_reads = good_reads.copy()
            good_reads_key = "matched"

        # Replace barcode with effective_barcode (filtered by well-specific ISS rounds)
        if good_reads is not None and not good_reads.empty:
            good_reads["barcode"] = good_reads["barcode"].apply(
                lambda barcode: "".join([barcode[i] for i in well_iss_rounds if i < len(barcode)])
            )

        freq_table = frequency_table(good_reads)
        freq_dict[pos] = freq_table

        # Save per-well frequency table
        freq_pos = freq_table.groupby("barcode", as_index=False)["count"].sum()
        freq_pos.to_csv(dataset.append_well("frequency_table", pos))

    # --- Create and save the POOLED frequency table ---
    if freq_dict:
        merged_freq = pd.concat(freq_dict.values())
        exp_freq = merged_freq.groupby("barcode", as_index=False)["count"].sum()
        exp_freq.to_csv(dataset.metrics_paths["frequency_table"])




def snr_by_round(experiment: str, num_samples: int = 50) -> Dict:
    """
    Estimate the average SNR of spots in each round of ISS for each channel
    """
    print(f"Estimating SNR by round for {experiment}")
    dataset = OpsDataset(experiment, method=None)
    from cyclops_process.metrics.plate_stats.metrics_iss_utils import resolve_iss_registered_store
    stitched_path = resolve_iss_registered_store(dataset)
    stitch_store = open_ome_zarr(stitched_path, mode="r")
    pos_list = [a[0] for a in stitch_store.positions()]

    snr_dict = {}
    for pos in tqdm(pos_list):
        fov = da.from_array(stitch_store[pos].data)
        spots = np.load(dataset.append_well("spots", pos))
        indxs = np.random.randint(0, spots.shape[0], num_samples)
        points = spots[indxs]
        ind = points.astype("int16")
        signals = []
        noises = []
        for p in ind[:]:
            value = fov[:, 1:, 0, p[0] - 5 : p[0] + 5, p[1] - 5 : p[1] + 5].compute()
            noise_crop = fov[
                :, 1:, 0, p[0] - 50 : p[0] + 50, p[1] - 50 : p[1] + 50
            ].compute()
            background = np.percentile(noise_crop, 80, axis=(2, 3))
            sig = np.max(value, axis=(2, 3))
            signals.append(sig)
            mask = noise_crop < np.expand_dims(background, axis=(2, 3))
            noises.append(np.std(noise_crop, where=mask, axis=(2, 3)))

        signal_array = np.asarray(signals)
        noise_array = np.asarray(noises)

        snr_array = signal_array / noise_array
        mean_snr = np.mean(snr_array, axis=0)
        snr_dict[pos] = mean_snr

    fig, ax = plt.subplots(1, len(snr_dict), figsize=(12, 3))
    if len(snr_dict) == 1:
        ax = [ax]
    for i, key in enumerate(snr_dict):
        mean_snr = snr_dict[key]
        ax[i].plot(mean_snr)
        ax[i].set_xlim(0, 9)
        ax[i].set_ylim(0.0, np.nanmax(mean_snr) * 1.05)
        ax[i].grid(True, linestyle="--", alpha=0.5)
        ax[i].set_title(key)
        if i == 0:
            ax[i].set_ylabel("Average SNR")
        if i == len(snr_dict) - 1:
            ax[i].legend(["G", "T", "A", "C"])
        ax[i].set_xlabel("ISS round")
    fig.suptitle(f"Average SNR by round\n{experiment}")

    plt.savefig(dataset.metrics_paths["snr_by_round"], dpi=300)

    return snr_dict




def read_accuracy_by_round_mine(
    experiment: str,
    iss_rounds: list[int] | None = None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
    force: bool = False,
) -> None:
    """
    For each round, what fraction of reads match back to the codebook?
    - for the first few rounds will be ~100%, but then expect a decrease after round 4ish

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
        force: If True, regenerate even if output file exists. Default False.
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    dataset = OpsDataset(experiment, method="mine")

    # Check if output already exists
    output_path = dataset.metrics_paths["read_acc_by_round"]
    if output_path.exists() and not force:
        print(f"Read accuracy by round plot already exists at: {output_path}")
        print("Skipping. Use --force or force=True to regenerate.")
        return None
    # get positions
    iss_seg_store = open_ome_zarr(dataset.store_paths["iss_segmentation"])
    pos_list = [a[0] for a in iss_seg_store.positions()]
    codebook_db = dataset.load_codebook()
    plate_results = {}
    max_round = max(iss_rounds)
    for pos in pos_list:
        # Filter out failed rounds for this well
        well_iss_rounds = _get_effective_iss_rounds(
            iss_rounds, pos, failed_rounds_by_well
        )

        # print(f"\n--- read_accuracy_by_round_mine: Processing well {pos} ---")
        # print(f"Original iss_rounds: {iss_rounds}")
        # print(f"Well-specific iss_rounds: {well_iss_rounds}")
        if failed_rounds_by_well and pos in failed_rounds_by_well:
            print(f"Failed rounds config for {pos}: {failed_rounds_by_well[pos]}")

        reads = pd.read_csv(dataset.append_well("reads", pos))
        round_acc = []
        # For each round 0 to max, calculate accuracy using only the available rounds up to that point
        for round_idx in range(max_round + 1):
            # Get all well-specific rounds up to and including this round
            current_rounds = [r for r in well_iss_rounds if r <= round_idx]
            if len(current_rounds) == 0:
                # This round and all before it are dropped out
                round_acc.append(np.nan)
            else:
                matched_reads = match_reads(reads, codebook_db, iss_rounds=current_rounds, well_name=pos, failed_rounds_by_well=failed_rounds_by_well, debug=False)
                round_acc.append(len(matched_reads) / len(reads))
        plate_results[pos] = round_acc

    plate_df = pd.DataFrame.from_dict(plate_results)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(plate_df)
    ax.set_xlim(0, max_round)
    ax.set_ylim(0.0, 1.05)
    ax.legend(list(plate_df.columns))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("ISS round")
    ax.set_ylabel("Read Accuracy")
    fig.suptitle(f"Read Accuracy by Round\n{experiment}")
    plt.savefig(dataset.metrics_paths["read_acc_by_round"], dpi=300)

    return plate_df

def read_accuracy_by_round(
    experiment,
    iss_rounds: list[int] = None,
    method: str = "mine",
    failed_rounds_by_well: dict[str, list[int]] | None = None,
    force: bool = False,
) -> None:
    if method == "probabilistic":
        # read_accuracy_by_round_prob(experiment, iss_rounds=iss_rounds)
        print(
            "skipping read accuracy by round prob, replaced with confidence score approach."
        )
    else:
        read_accuracy_by_round_mine(
            experiment, iss_rounds=iss_rounds, failed_rounds_by_well=failed_rounds_by_well, force=force
        )
