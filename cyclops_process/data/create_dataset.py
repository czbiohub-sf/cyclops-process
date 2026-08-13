import pandas as pd
import matplotlib.pyplot as plt
import zarr
from tqdm import tqdm
import numpy as np

from iohub import open_ome_zarr
from cyclops_utils.data.experiment import OpsDataset


def check_edges(array: np.array) -> None:
    """
    Assumes that array is in OME format with shape (T, C, Z, Y, X)
    returns a bool:
        - True if any of the edges contain
    """
    top = array[0, :, 0, 0, :]
    bottom = array[0, :, 0, -1, :]
    left = array[0, :, 0, :, 0]
    right = array[0, :, 0, :, -1]

    edges = np.concatenate((top, bottom, left, right), axis=0)

    return np.any(edges == 0)


def create_dataset(experiment: str, well: str) -> None:
    """
    Creates a dataset of single-cell crops
        - crops are taken from individual tiles

    TODO:
        - Look at fraction of cells that are removed by edge detector
            - Could pull from stitched image, but worried about model being able
            detect cells that span a stitched region
        - Look into why shape checker is necessary
    """
    print(f"Creating dataset for well {well} in experiment {experiment}")
    dataset = OpsDataset(experiment)
    output_path = dataset.append_well("sc_dataset", well)
    w = dataset.store_props["sc_crop_size"]

    linked_path = dataset.append_well("linked_results", well)
    linked_df = pd.read_csv(linked_path, index_col=0)
    # Prefer stitched global coordinates if available
    sort_key = "tile_pheno" if "tile_pheno" in linked_df.columns else None
    if sort_key:
        linked_df = linked_df.sort_values(by=sort_key)
    pheno_assembled_path = dataset.store_paths["pheno_assembled_v3"]
    pheno_store = open_ome_zarr(pheno_assembled_path)
    channel_names = pheno_store.channel_names

    output_zarr = zarr.DirectoryStore(output_path)
    store = zarr.group(
        store=output_zarr, overwrite=True
    )  # will not ammend, will create a new zarr

    gene_set = set()
    gene_count_dict = {}
    zattrs_dict = {}
    # Access the stitched well image once
    well_data = pheno_store[well].data  # (T,C,Z,Y,X)
    for i in tqdm(range(linked_df.shape[0]), desc="Creating dataset"):

        tile = linked_df.iloc[i].get("tile_pheno", None)
        gene_name = linked_df.iloc[i].gene_name
        if str(gene_name) == "nan":
            gene_name = "NTC_" + str(linked_df.iloc[i].barcode)
        barcode = linked_df.iloc[i].barcode
        # Use stitched global coordinates if present; else fallback to local tile coords
        if {"y_global_pheno", "x_global_pheno"}.issubset(linked_df.columns):
            cy = int(linked_df.iloc[i].y_global_pheno)
            cx = int(linked_df.iloc[i].x_global_pheno)
        else:
            # Fallback: use local tile coords if global not available
            cy = int(linked_df.iloc[i].y_local)
            cx = int(linked_df.iloc[i].x_local)

        y0 = cy - w
        y1 = cy + w
        x0 = cx - w
        x1 = cx + w
        # Bounds check; skip crops that would go out of bounds
        H, W = int(well_data.shape[-2]), int(well_data.shape[-1])
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue

        sc_crop = well_data[:, :, :, y0:y1, x0:x1]

        if sc_crop.shape[-2:] != (2 * w, 2 * w):
            # print(sc_crop.shape)
            continue

        if check_edges(sc_crop):
            continue

        # TODO: do we want this to include all z-slices?
        norm_df = pd.DataFrame(np.mean(sc_crop, axis=(0, 2, 3, 4)), columns=["mean"])
        norm_df["channel"] = channel_names
        norm_df["var"] = np.var(sc_crop, axis=(0, 2, 3, 4))
        norm_df["min"] = np.min(sc_crop, axis=(0, 2, 3, 4))
        norm_df["max"] = np.max(sc_crop, axis=(0, 2, 3, 4))

        # if gene is not in gene set, create a new position and add to set
        if gene_name not in gene_set:
            store.create_group(gene_name)
            gene_set.add(gene_name)
            gene_count_dict[gene_name] = 0

        store[gene_name][gene_count_dict[gene_name]] = sc_crop

        sc_position = f"{gene_name}/{gene_count_dict[gene_name]}"
        zattrs_dict[sc_position] = {
            "experiment": experiment,
            "well": well,
            "tile": tile,
            "gene_name": gene_name,
            "barcode": barcode,
            "x_global": cx,
            "y_global": cy,
            "normalization": norm_df.to_dict(),
        }

        gene_count_dict[gene_name] += 1
    store.attrs.update(zattrs_dict)

    return


if __name__ == "__main__":
    experiment = "ops0031_20250424"
    well = "A/1/0"
    create_dataset(experiment, well)
