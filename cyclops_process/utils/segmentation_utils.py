from __future__ import annotations  # defer torch.Tensor annotations (torch optional below)
from typing import Tuple
try:
    import torch
except ModuleNotFoundError:  # optional GPU dep; used only in GPU runtime paths
    torch = None

"""
Util Functions adapted from Instanseg Library, originally written by Thibaut Goldsborough
https://github.com/instanseg/instanseg/tree/main/instanseg/utils
"""


def remap_values(remapping: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    This remaps the values in x according to the pairs in the remapping tensor.
    remapping: 2,N      Make sure the remapping is 1 to 1, and there are no loops (i.e. 1->2, 2->3, 3->1). Loops can be removed using graph based connected components algorithms (see instanseg postprocessing for an example)
    x: any shape
    """
    sorted_remapping = remapping[:, remapping[0].argsort()]
    index = torch.bucketize(x.ravel(), sorted_remapping[0])
    return sorted_remapping[1][index].reshape(x.shape)


def torch_fastremap(x: torch.Tensor) -> torch.Tensor:
    if x.max() == 0:
        return x
    unique_values = torch.unique(x, sorted=True)
    new_values = torch.arange(len(unique_values), dtype=x.dtype, device=x.device)
    remapping = torch.stack((unique_values, new_values))
    return remap_values(remapping, x)


def torch_sparse_onehot(
    x: torch.Tensor, flatten: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # x is a labeled image of shape _,_,H,W returns a sparse tensor of shape C,H,W
    unique_values = torch.unique(x, sorted=True)
    x = torch_fastremap(x)

    H, W = x.shape[-2], x.shape[-1]

    if flatten:

        if x.max() == 0:
            return torch.zeros_like(x).reshape(1, 1, H * W)[:, :0], unique_values

        x = x.reshape(H * W)
        xxyy = torch.nonzero(x > 0).squeeze(1)
        zz = x[xxyy] - 1
        C = x.max().int().item()

        # print(C, H, W, type(C), type(H), type(W))
        sparse_onehot = torch.sparse_coo_tensor(
            torch.stack((zz, xxyy)).long(),
            (torch.ones_like(xxyy).float()),
            size=(int(C), int(H * W)),
            dtype=torch.float32,
        )

    else:
        if x.max() == 0:
            return torch.zeros_like(x).reshape(1, 0, H, W), unique_values

        x = x.squeeze().view(H, W)
        x_temp = torch.nonzero(x > 0).T
        zz = x[x_temp[0], x_temp[1]] - 1
        C = x.max().int().item()
        sparse_onehot = torch.sparse_coo_tensor(
            torch.stack((zz, x_temp[0], x_temp[1])).long(),
            (torch.ones_like(x_temp[0]).float()),
            size=(int(C), int(H), int(W)),
            dtype=torch.float32,
        )

    return sparse_onehot, unique_values


def fast_sparse_dual_iou(onehot1: torch.Tensor, onehot2: torch.Tensor) -> torch.Tensor:
    """
    Returns the (dense) intersection over union between two sparse onehot encoded tensors
    """
    # onehot1 and onehot2 are C1,H*W and C2,H*W

    intersection = torch.sparse.mm(onehot1, onehot2.T).to_dense()
    sparse_sum1 = torch.sparse.sum(onehot1, dim=(1,))[None].to_dense()
    sparse_sum2 = torch.sparse.sum(onehot2, dim=(1,))[None].to_dense()
    union = sparse_sum1.T + sparse_sum2 - intersection

    return intersection / union


def match_labels(
    tile_1: torch.Tensor, tile_2: torch.Tensor, threshold: float = 0.5, strict=False
):
    """This function takes two labeled tiles, and matches the overlapping labels of tile_2 to the labels of tile_1.
    If strict is set to True, the function will discard non matching objects.
    """

    if tile_1.max() == 0 or tile_2.max() == 0:
        if not strict:
            return tile_1, tile_2
        else:
            return torch.zeros_like(tile_1), torch.zeros_like(tile_2)

    old_problematic_onehot, old_unique_values = torch_sparse_onehot(
        tile_1, flatten=True
    )
    new_problematic_onehot, new_unique_values = torch_sparse_onehot(
        tile_2, flatten=True
    )

    iou = fast_sparse_dual_iou(old_problematic_onehot, new_problematic_onehot)

    onehot_remapping = torch.nonzero(iou > threshold).T  # + 1

    if old_unique_values.min() == 0:
        old_unique_values = old_unique_values[old_unique_values > 0]
    if new_unique_values.min() == 0:
        new_unique_values = new_unique_values[new_unique_values > 0]

    if onehot_remapping.shape[1] > 0:

        onehot_remapping = torch.stack(
            (
                new_unique_values[onehot_remapping[1]],
                old_unique_values[onehot_remapping[0]],
            )
        )

        if not strict:
            mask = torch.isin(tile_2, onehot_remapping[0])
            tile_2[mask] = remap_values(onehot_remapping, tile_2[mask])

            return tile_1, tile_2
        else:
            tile_1 = tile_1 * torch.isin(tile_1, onehot_remapping[1]).int()
            tile_2 = tile_2 * torch.isin(tile_2, onehot_remapping[0]).int()

            tile_2[tile_2 > 0] = remap_values(onehot_remapping, tile_2[tile_2 > 0])

            return tile_1, tile_2

    else:
        if not strict:
            return tile_1, tile_2
        else:
            return torch.zeros_like(tile_1), torch.zeros_like(tile_2)


def _edge_mask(labels, ignore=[None]):
    labels = labels.squeeze()
    first_row = labels[0, :]
    last_row = labels[-1, :]
    first_column = labels[:, 0]
    last_column = labels[:, -1]

    edges = []
    if "top" not in ignore:
        edges.append(first_row)
    if "bottom" not in ignore:
        edges.append(last_row)
    if "left" not in ignore:
        edges.append(first_column)
    if "right" not in ignore:
        edges.append(last_column)

    if len(edges) == 0:
        return torch.zeros_like(labels).bool()

    edges = torch.cat(edges, dim=0)
    return torch.isin(labels, edges[edges > 0])


def _remove_edge_labels(labels, ignore=[None]):
    return labels * ~_edge_mask(labels, ignore=ignore)


def match_labels_fast(
    tile_1: torch.Tensor, tile_2: torch.Tensor, threshold: float = 0.5
) -> tuple:
    """
    Fast label matching that only computes IoU for spatially overlapping labels.

    This is much faster than match_labels for tiles with many small objects
    because it avoids computing IoU between labels that don't share any pixels.

    Args:
        tile_1: Reference labeled tile (labels to match against)
        tile_2: Tile to be remapped
        threshold: IoU threshold for matching (default: 0.5)

    Returns:
        Tuple of (tile_1, remapped_tile_2)
    """
    import numpy as np

    if tile_1.max() == 0 or tile_2.max() == 0:
        return tile_1, tile_2

    # Convert to numpy for faster processing
    t1 = tile_1.numpy() if isinstance(tile_1, torch.Tensor) else tile_1
    t2 = tile_2.numpy() if isinstance(tile_2, torch.Tensor) else tile_2

    # Find pixels where both tiles have labels
    both_labeled = (t1 > 0) & (t2 > 0)
    if not both_labeled.any():
        return tile_1, tile_2

    # Get unique label pairs at overlapping pixels
    t1_at_overlap = t1[both_labeled]
    t2_at_overlap = t2[both_labeled]

    # Create a mapping of (t1_label, t2_label) -> count
    # Using a structured array for efficient grouping
    pairs = np.stack([t1_at_overlap, t2_at_overlap], axis=1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

    if len(unique_pairs) == 0:
        return tile_1, tile_2

    # For each unique pair, compute IoU
    # IoU = intersection / union = intersection / (area1 + area2 - intersection)
    t1_labels_in_overlap = unique_pairs[:, 0]
    t2_labels_in_overlap = unique_pairs[:, 1]

    # Get areas of each label
    t1_unique, t1_counts = np.unique(t1[t1 > 0], return_counts=True)
    t2_unique, t2_counts = np.unique(t2[t2 > 0], return_counts=True)

    t1_area = dict(zip(t1_unique, t1_counts))
    t2_area = dict(zip(t2_unique, t2_counts))

    # Build remapping for tile_2 labels
    remap = {}
    for i, (l1, l2) in enumerate(unique_pairs):
        intersection = counts[i]
        area1 = t1_area.get(l1, 0)
        area2 = t2_area.get(l2, 0)
        union = area1 + area2 - intersection

        if union > 0:
            iou = intersection / union
            if iou > threshold:
                # Only remap if this is the best match for l2
                if l2 not in remap or iou > remap[l2][1]:
                    remap[l2] = (l1, iou)

    # Apply remapping to tile_2
    result = t2.copy()
    for old_label, (new_label, _) in remap.items():
        result[t2 == old_label] = new_label

    # Convert back to torch if input was torch
    if isinstance(tile_2, torch.Tensor):
        result = torch.tensor(result, dtype=tile_2.dtype)

    return tile_1, result
