from pathlib import Path
from tqdm import tqdm
import warnings
import yaml
import subprocess
import json

import numpy as np
from iohub import open_ome_zarr
from iohub.ngff import TransformationMeta
from ops_utils.data.image_utils import augment_tile
from stitch.registration.register import read_transform_biahub

from ops_utils.profiling.decorators import versioned_function
from ops_utils.data.experiment import OpsDataset
from cyclops_process.convert.tiff_to_zarr import convert
from cyclops_process.processes.reconstruct import reconstruct
from ops_utils.data.filesystem import decide_overwrite_resume_skip
from cyclops_process.utils.project import create_max_projection as _create_max_projection

try:
    import cupy as xp
    from cupyx.scipy import ndimage as cundi

except (ModuleNotFoundError, ImportError):
    import numpy as xp
    from scipy import ndimage as cundi

warnings.filterwarnings("ignore")

# def get_augmentations(json_path):

#     with open(json_path, "r") as f:
#         image_plane_meta = json.load(f)

#     # within image_plane_meta need to index into specific channels with
#     # 3 diget index T/C/Z

#     gfp_aug_dict = image_plane_meta['0/0/0']['UserData']
#     mcherry_aug_dict = image_plane_meta['0/1/0']['UserData']
#     phase_aug_dict = image_plane_meta['0/2/0']['UserData']

#     channels = ['gfp', 'mcherry', 'phase']

#     flip_dict = {
#         'gfp': {'flipud': False, 'fliplr': False, 'rot90': 0},
#         'mcherry': {'flipud': False, 'fliplr': False, 'rot90': 0},
#         'phase': {'flipud': False, 'fliplr': False, 'rot90': 0},
#     }

#     for c, d in zip(channels, [gfp_aug_dict, mcherry_aug_dict, phase_aug_dict]):
#         if not d:
#             print(f"No augmentation metadata for channel {c}, assuming no augmentations")
#             continue
#         # if d['ImageFlipper-Mirror']['scalar'] == 'On':
#         #     flip_dict[c]['fliplr'] = True
#         flip_dict[c]['rot90'] = d['ImageFlipper-Rotation']['scalar'] // 90

#     return flip_dict


def prepare_beads(experiment, flip_dict):

    dataset = OpsDataset(experiment)

    # convert beads
    if not dataset.store_paths["lc_20x_beads"].exists():
        convert(experiment, process="20x_beads")
    if not dataset.store_paths["lc_20x_beads_phase"].exists():
        reconstruct(experiment, process="20x_beads", local=True)

    # restack zarr with augmentations applied
    original_orientation_ds = open_ome_zarr(
        dataset.store_paths["lc_20x_beads"], mode="r"
    )

    # json_path = dataset.store_paths["lc_20x_beads"] / "0/0/0/0/image_plane_metadata.json"
    # flip_dict = get_augmentations(json_path)

    original_orientation_ds_phase = open_ome_zarr(
        dataset.store_paths["lc_20x_beads_phase"], mode="r"
    )
    channel_list = original_orientation_ds.channel_names
    channel_list = ["Phase3D" if s == "BF" else s for s in channel_list]
    if not dataset.store_paths["lc_20x_beads_assembled"].exists():
        beads_out = open_ome_zarr(
            dataset.store_paths["lc_20x_beads_assembled"],
            layout="hcs",
            mode="w-",
            channel_names=channel_list,
        )

    pos = "0/0/0"
    original_orientation_data = original_orientation_ds[pos].data
    original_orientation_data_phase = original_orientation_ds_phase[pos].data

    oo_gfp = original_orientation_data[0, channel_list.index("GFP"), :, :, :]
    oo_mcherry = original_orientation_data[0, channel_list.index("mCherry"), :, :, :]
    oo_phase = original_orientation_data_phase[0, 0, :, :, :]

    aug_gfp = np.max(
        augment_tile(
            oo_gfp,
            flipud=flip_dict["gfp"]["flipud"],
            fliplr=flip_dict["gfp"]["fliplr"],
            rot90=flip_dict["gfp"]["rot90"],
        ),
        axis=0,
    )
    aug_mcherry = np.max(
        augment_tile(
            oo_mcherry,
            flipud=flip_dict["mcherry"]["flipud"],
            fliplr=flip_dict["mcherry"]["fliplr"],
            rot90=flip_dict["mcherry"]["rot90"],
        ),
        axis=0,
    )
    aug_phase = np.max(
        augment_tile(
            oo_phase,
            flipud=flip_dict["phase"]["flipud"],
            fliplr=flip_dict["phase"]["fliplr"],
            rot90=flip_dict["phase"]["rot90"],
        ),
        axis=0,
    )

    out = np.expand_dims(
        np.stack([aug_gfp, aug_mcherry, aug_phase], axis=0), axis=(0, 2)
    )

    out_pos = beads_out.create_position(*Path(pos).parts)
    out_fov = out_pos.create_image(
        name="0",
        data=out,
        chunks=dataset.store_props["chunk_size"],
        transform=[
            TransformationMeta(type="scale", scale=dataset.store_props["20x_scale"])
        ],
    )

    return


def viscy_normalization(
    dataset: OpsDataset,
    channel_names: list[str],
):
    """
    Run viscy - preprocess on all channels after dataset it assembled

    See code in virtual-staining line 48
    """
    from ops_utils.hpc.resource_manager import get_optimal_workers

    input_path = dataset.store_paths["pheno_assembled_v3"]
    num_workers = get_optimal_workers(use_gpu=False, model_ram_gb=1.0, data_ram_gb=2.0)
    normalization_config = {
        "data_path": f"{input_path}",
        "channel_names": channel_names,
        "num_workers": num_workers,
        "block_size": 32,
    }
    norm_config_path = dataset.config_paths["pheno_assembled_norm"]

    # generate new normalization config yml (record of what was run)
    with open(norm_config_path, "w") as f:
        yaml.dump(normalization_config, f)

    # Compute normalization statistics with bounded I/O. The assembled canvas is
    # ~100k*100k per well, and viscy's preprocess reads each full Y*X slice into
    # RAM before subsampling (tens of GB/channel) — it times out. fast_normalize
    # samples a bounded grid of chunk-aligned blocks and writes identical
    # `normalization` metadata. See cyclops_process.utils.fast_normalize.
    from cyclops_process.utils.fast_normalize import generate_normalization_metadata_fast
    from cyclops_process.processes.virtual_staining import _mirror_normalization_to_custom_metadata

    generate_normalization_metadata_fast(str(input_path), channel_ids=-1, num_workers=num_workers)
    _mirror_normalization_to_custom_metadata(input_path)
    return


@versioned_function("v1.0")
def viscy_normalize(experiment: str) -> None:
    """Run ViSCy preprocess on the stitched unified phenotyping store."""
    dataset = OpsDataset(experiment)
    pheno_path = dataset.store_paths["pheno_assembled_v3"]
    try:
        store = open_ome_zarr(pheno_path, mode="r")
        ch_names = list(store.channel_names)
    except Exception:
        ch_names = []
    viscy_normalization(dataset, ch_names)
    return


@versioned_function("v1.0")
def create_max_projection(
    experiment: str,
    process: str,
    slices: list | str = "all",
    projection: str = "max",
) -> None:
    """Create max projection from 3D data.

    Wrapper function that calls the implementation in utils.project.
    """
    return _create_max_projection(experiment, process, slices, projection)
