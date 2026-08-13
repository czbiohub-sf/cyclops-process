import numpy as np
import pandas as pd


def extract_reads(fov, points, nuclei):
    """
    extract intensities of each detected point
    """
    indecies = points.T.astype(np.uint16)

    values = fov[:, 1:, indecies[0], indecies[1]].transpose([2, 0, 1])
    # print(values.shape)
    cell_label = nuclei[indecies[0], indecies[1]]

    cycles = list(range(1, fov.shape[0] + 1))
    bases = list("GTAC")

    df = _format_bases(values, cell_label, indecies, cycles, bases)

    return df


def _format_bases(values, labels, positions, cycles, bases):
    """Arrange ?x?x? arrays of base intensity information into a dataframe in "long" format (one
    row per observation).
    """
    index = ("cycle", cycles), ("channel", bases)
    try:
        df = _ndarray_to_dataframe(values, index)
    except ValueError:
        print(
            "failed to reshape extracted pixels to sequencing bases, writing empty table"
        )
        return pd.DataFrame()

    df_positions = pd.DataFrame(positions.T, columns=["i", "j"])
    df = (
        df.stack(["cycle", "channel"], future_stack=True)
        .reset_index()
        .rename(columns={0: "intensity", "level_0": "read"})
        .join(pd.Series(labels, name="cell"), on="read")
        .join(df_positions, on="read")
        .sort_values(["cell", "read", "cycle"])
    )

    return df


def _ndarray_to_dataframe(values, index):
    names, levels = zip(*index)
    columns = pd.MultiIndex.from_product(levels, names=names)
    df = pd.DataFrame(values.reshape(values.shape[0], -1), columns=columns)
    return df


def call_reads(df_bases, peaks=None, correction_only_in_cells=True):
    """Call reads by compensating for channel cross-talk and calling the base
    with highest corrected intensity for each cycle. This "median correction"
    is performed independently for each tile.
    """
    if df_bases is None:
        return
    if correction_only_in_cells:
        if len(df_bases.query("cell > 0")) == 0:
            return

    cycles = len(set(df_bases["cycle"]))
    channels = len(set(df_bases["channel"]))

    df_reads = df_bases.pipe(_clean_up_bases).pipe(
        _do_median_call,
        cycles,
        channels=channels,
        correction_only_in_cells=correction_only_in_cells,
    )

    if peaks is not None:
        i, j = df_reads[["i", "j"]].values.T
        df_reads["peak"] = peaks[i, j]

    return df_reads  # .loc[df_reads["cell"] != 0] # remove calls that map to background


def _clean_up_bases(df_bases):
    """Sort. Pre-processing for `dataframe_to_values`."""
    return df_bases.sort_values(["cell", "read", "cycle", "channel"])


def _do_median_call(df_bases, cycles=12, channels=4, correction_only_in_cells=False):
    """Call reads from raw base signal using median correction. Use the
    `correction_within_cells` flag to specify if correction is based on reads within
    cells, or all reads.
    """
    if correction_only_in_cells:
        # first obtain transformation matrix W
        X_ = _dataframe_to_values(df_bases.query("cell > 0"))
        _, W = _transform_medians(X_.reshape(-1, channels))

        # then apply to all data
        X = _dataframe_to_values(df_bases)
        Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
    else:
        X = _dataframe_to_values(df_bases)
        Y, W = _transform_medians(X.reshape(-1, channels))

    df_reads = _call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads


def _dataframe_to_values(df, value="intensity"):
    """Dataframe must be sorted on [cycle, channel].
    Returns N x cycles x channels.
    """
    cycles = df["cycle"].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)
    n_channels = len(df["channel"].value_counts())
    x = np.array(df[value]).reshape(-1, n_cycles, n_channels)
    return x


def _transform_medians(X):
    """Estimate and correct differences in channel intensity and spectral overlap among sequencing
    channels. For each channel, find points where the largest signal is from that channel. Use the
    median of these points to define new basis vectors. Describe with linear transformation W
    so that W * X = Y, where Y is the corrected data.
    """

    def get_medians(X):
        arr = []
        for i in range(X.shape[1]):
            max_spots = X[X.argmax(axis=1) == i]
            # Handle the edge case where a channel is never the brightest
            if max_spots.shape[0] == 0:
                # If no spots have this channel as max, its signal is always low.
                # We create a pseudo-median vector representing a "pure" signal for this
                # channel, which is a robust fallback for the correction matrix.
                pseudo_median = np.zeros(X.shape[1])
                pseudo_median[i] = 1.0  # Use a float to prevent type issues later
                arr += [pseudo_median]
            else:
                arr += [np.median(max_spots, axis=0)]
        M = np.array(arr)
        return M

    M = get_medians(X).T
    # Normalize each column to sum to 1, with a check for zero-sum columns.
    column_sums = M.sum(axis=0)

    # Use np.isclose for robust floating-point comparison
    zero_sum_mask = np.isclose(column_sums, 0)

    if np.any(zero_sum_mask):
        print(
            "WARNING: One or more correction matrix columns summed to zero. Adding epsilon to prevent division by zero."
        )
        column_sums[zero_sum_mask] += 1e-9  # Add epsilon only where needed

    M = M / column_sums

    try:
        W = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        # If the matrix is singular, fall back to an identity matrix
        print(
            "WARNING: Median correction matrix is singular. Falling back to identity (no correction)."
        )
        W = np.eye(X.shape[1])

    Y = W.dot(X.T).T.astype(int)
    return Y, W


def _call_barcodes(df_bases, Y, cycles=12, channels=4):
    """Use the argmax over channels in corrected data `Y` to identify bases in each cycle. Result
    dataframe has one row per read and preserves any extra information from `df_bases`.
    """
    bases = sorted(set(df_bases["channel"]))
    if any(len(x) != 1 for x in bases):
        raise ValueError("supplied weird bases: {0}".format(bases))
    df_reads = df_bases.drop_duplicates(["cell", "read"]).copy()
    df_reads["barcode"] = _call_bases_fast(Y.reshape(-1, cycles, channels), bases)
    Q = _quality(Y.reshape(-1, cycles, channels))
    # needed for performance later
    for i in range(len(Q[0])):
        df_reads["Q_%d" % i] = Q[:, i]

    return df_reads.assign(Q_min=lambda x: x.filter(regex=r"Q_\d+").min(axis=1)).drop(
        ["cycle", "channel", "intensity"], axis=1
    )


def _call_bases_fast(values, bases):
    """Apply argmax to `values` and form strings by indexing `bases` (e.g., "ACGT" if channel
    dimension in `values` is pre-sorted).
    """
    assert values.ndim == 3
    assert values.shape[2] == len(bases)
    calls = values.argmax(axis=2)
    calls = np.array(list(bases))[calls]
    return ["".join(x) for x in calls]


def _quality(X):
    """Define an ad-hoc quality score per sequencing call based on the highest and
    second-highest channels. Adjusted empirically to give reasonable results with 4-color
    sequencing.
    """
    X = np.abs(np.sort(X, axis=-1).astype(float))
    Q = 1 - np.log(2 + X[..., -2]) / np.log(2 + X[..., -1])
    Q = (Q * 2).clip(0, 1)
    return Q
