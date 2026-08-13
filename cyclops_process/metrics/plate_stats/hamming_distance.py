from ops_utils.data.experiment import OpsDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cyclops_process.metrics.plate_stats.match_reads import _get_effective_iss_rounds




def hamming_distance(
    experiment: str,
    well: str,
    iss_rounds: list[int] | None = None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> None:
    """
    Report the minimum hamming distance from each read to the codebook

    Args:
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        failed_rounds_by_well: Dictionary mapping wells to lists of failed round indices to exclude
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    # Filter out failed rounds for this well
    iss_rounds = _get_effective_iss_rounds(iss_rounds, well, failed_rounds_by_well)

    dataset = OpsDataset(experiment, method="mine")
    results_path = dataset.append_well("reads", well)

    # Read inputs with validation
    try:
        codebook_db = dataset.load_codebook()
    except Exception as e:
        print(
            f"Could not read codebook at {dataset.codebook}: {e}. Skipping hamming distance."
        )
        return

    if "sgRNA" not in codebook_db.columns:
        print("'sgRNA' column missing in codebook. Skipping hamming distance.")
        return

    codes = ["".join([a[i] for i in iss_rounds]) for a in codebook_db["sgRNA"]]
    if len(codes) == 0:
        print("Codebook has no sgRNA entries. Skipping hamming distance.")
        return

    try:
        results_df = pd.read_csv(results_path)
    except Exception as e:
        print(
            f"Could not read reads file at {results_path}: {e}. Skipping hamming distance."
        )
        return

    if results_df.empty:
        print("Reads dataframe is empty. Skipping hamming distance.")
        return
    if "cell" not in results_df.columns or "barcode" not in results_df.columns:
        print(
            "Required columns 'cell' or 'barcode' missing in reads. Skipping hamming distance."
        )
        return

    results_in_cells = results_df[results_df["cell"] != 0]
    if len(results_in_cells) == 0:
        print("No reads within cells found. Skipping hamming distance.")
        return

    # Size-aware, fail-safe sampling: try up to 10,000 then 1,000, else use all rows
    try:
        n1 = min(len(results_in_cells), 10000)
        results_sample = (
            results_in_cells.sample(n1, random_state=42)
            if n1 < len(results_in_cells)
            else results_in_cells
        )
    except Exception as e:
        print(f"Error sampling reads: {e}, retrying with 1,000 reads")
        try:
            n2 = min(len(results_in_cells), 1000)
            results_sample = (
                results_in_cells.sample(n2, random_state=42)
                if n2 < len(results_in_cells)
                else results_in_cells
            )
        except Exception as e2:
            print(f"Fallback sampling failed: {e2}. Using all reads in cells.")
            results_sample = results_in_cells

    if len(results_sample) == 0:
        print("Sampling produced zero rows. Skipping hamming distance.")
        return

    ref_array = np.array([list(seq) for seq in codes])
    # Filter barcodes using the same ISS round positions as the codebook
    seq_array = np.array(
        [list("".join([barcode[i] for i in iss_rounds if i < len(barcode)]))
         for barcode in results_sample["barcode"]]
    )

    # Handle round count mismatch (barcodes may already be truncated from failed rounds)
    ref_len = ref_array.shape[1] if ref_array.ndim > 1 else len(ref_array[0]) if len(ref_array) > 0 else 0
    seq_len = seq_array.shape[1] if seq_array.ndim > 1 else len(seq_array[0]) if len(seq_array) > 0 else 0

    if ref_len != seq_len:
        # Truncate to the shorter length (common rounds)
        common_len = min(ref_len, seq_len)
        if common_len == 0:
            print(f"Cannot compute hamming distance: ref has {ref_len} rounds, barcodes have {seq_len} rounds. Skipping.")
            return
        print(f"[hamming_distance] Round mismatch: codebook has {ref_len} rounds, barcodes have {seq_len} rounds. Using first {common_len} rounds.")
        ref_array = ref_array[:, :common_len]
        seq_array = seq_array[:, :common_len]

    # Expand dimensions for broadcasting: (R, 1, L) vs (1, N, L)
    ref_array_exp = ref_array[:, np.newaxis, :]
    seq_array_exp = seq_array[np.newaxis, :, :]

    hamming_distances = (ref_array_exp != seq_array_exp).sum(axis=2)
    min_distances = np.min(hamming_distances, axis=0)
    counts, bins = np.histogram(min_distances, bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    counts = np.insert(counts, 0, len(results_sample))
    counts = counts / np.sum(counts)
    bins = np.insert(bins, 0, -0.5)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(bins[:-1] + 0.5, counts, width=0.9)
    ax.set_xlabel("Hamming distance to codebook")
    ax.set_ylabel("Fraction of reads")
    fig.suptitle("Hamming distance to codebook")
    ax.set_title(f"{experiment} - {well}", fontsize=10)
    plt.savefig(dataset.metrics_paths["hamming_distance"], dpi=300)

    return
