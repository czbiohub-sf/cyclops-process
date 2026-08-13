import pytest
import os
import shutil
import yaml
import json
from pathlib import Path
import shutil

from ops_utils.data.experiment import OpsDataset
from cyclops_process.processes.assemble_link import convert


def create_stitching_configs(phenotyping_positions, iss_positions, tracking_positions):
    source_dataset = OpsDataset("ops0033_20250429")
    dest_dataset = OpsDataset("ops_testset")

    def filter_config(config, positions):
        temp_dict = {k: config["total_translation"][k] for k in positions}

        i_values = [temp_dict[k][0] for k in temp_dict.keys()]
        j_values = [temp_dict[k][1] for k in temp_dict.keys()]

        i_shifted = [i - min(i_values) for i in i_values]
        j_shifted = [j - min(j_values) for j in j_values]

        test_config = {}
        test_config["total_translation"] = {
            k: [i_shifted[idx], j_shifted[idx]]
            for idx, k in enumerate(temp_dict.keys())
        }

        return test_config

    # ISS stitching config
    iss_source_path = source_dataset.config_paths["iss_stitch"]
    with open(iss_source_path, "r") as f:
        iss_config = yaml.safe_load(f)
    iss_config = filter_config(iss_config, iss_positions)
    iss_dest_path = dest_dataset.config_paths["iss_stitch"]
    with open(iss_dest_path, "w") as f:
        yaml.safe_dump(iss_config, f)

    # Phenotyping stitching config
    phenotyping_source_path = source_dataset.config_paths["lc_20x_stitch"]
    with open(phenotyping_source_path, "r") as f:
        phenotyping_config = yaml.safe_load(f)
    phenotyping_config = filter_config(phenotyping_config, phenotyping_positions)
    phenotyping_dest_path = dest_dataset.config_paths["lc_20x_stitch"]
    with open(phenotyping_dest_path, "w") as f:
        yaml.safe_dump(phenotyping_config, f)

    # Tracking stitching config
    tracking_source_path = source_dataset.config_paths["lc_5x_stitch"]
    with open(tracking_source_path, "r") as f:
        tracking_config = yaml.safe_load(f)
    tracking_config = filter_config(tracking_config, tracking_positions)
    tracking_dest_path = dest_dataset.config_paths["lc_5x_stitch"]
    with open(tracking_dest_path, "w") as f:
        yaml.safe_dump(tracking_config, f)

    # Fluorescence registration configs
    gfp_source_path = source_dataset.config_paths["lc_GFP_register"]
    gfp_dest_path = dest_dataset.config_paths["lc_GFP_register"]
    mcherry_source_path = source_dataset.config_paths["lc_mCherry_register"]
    mcherry_dest_path = dest_dataset.config_paths["lc_mCherry_register"]
    if not gfp_dest_path.exists():
        os.makedirs(gfp_dest_path.parent, exist_ok=True)
    shutil.copy(gfp_source_path, gfp_dest_path)
    shutil.copy(mcherry_source_path, mcherry_dest_path)

    return


def create_images_testset(phenotyping_positions, iss_positions, tracking_positions):

    def filter_positions(source_path, dest_path, positions):
        dest_path.mkdir(exist_ok=True)

        # copy fov level metadata
        for fname in [".zattrs", ".zgroup"]:
            src = source_path / fname
            dest = dest_path / fname
            if src.exists():
                if not dest.exists():
                    os.symlink(src.resolve(), dest)

        # copy fovs
        for pos in positions:
            pos = Path(pos)
            src_pos = source_path / pos
            dst_pos = dest_path / pos
            dst_pos.parent.mkdir(parents=True, exist_ok=True)
            if src_pos.exists():
                if not dst_pos.exists():
                    os.symlink(src_pos.resolve(), dst_pos)

        # copy well level zattrs
        src_zattrs = source_path / pos.parent / ".zattrs"
        dst_zattrs = dest_path / pos.parent / ".zattrs"
        if src_zattrs.exists() and not dst_zattrs.exists():
            with open(src_zattrs, "r") as f:
                attrs = json.load(f)
            filtered_attrs = attrs.copy()
            fov_positions = [Path(p).name for p in positions]
            filtered_attrs["well"]["images"] = [
                attrs["well"]["images"][i]
                for i in range(len(attrs["well"]["images"]))
                if attrs["well"]["images"][i]["path"] in fov_positions
            ]

            with open(dst_zattrs, "w") as f:
                json.dump(filtered_attrs, f, indent=2)

        # copy well level zgroup
        source_zgroup = source_path / pos.parent / ".zgroup"
        dest_zgroup = dest_path / pos.parent / ".zgroup"
        if source_zgroup.exists() and not dest_zgroup.exists():
            os.symlink(source_zgroup.resolve(), dest_zgroup)

        # copy row level zgroup
        source_zgroup = source_path / pos.parent.parent / ".zgroup"
        dest_zgroup = dest_path / pos.parent.parent / ".zgroup"
        if source_zgroup.exists() and not dest_zgroup.exists():
            os.symlink(source_zgroup.resolve(), dest_zgroup)
        return

    source_dataset = OpsDataset("ops0033_20250429")
    dest_dataset = OpsDataset("ops_testset")

    # ISS raw data
    iss_source_path = source_dataset.store_paths["iss"]
    iss_dest_path = dest_dataset.store_paths["iss"]
    filter_positions(iss_source_path, iss_dest_path, iss_positions)

    # ISS drift corrected
    iss_drift_corrected_source_path = source_dataset.store_paths["iss_drift_corrected"]
    iss_drift_corrected_dest_path = dest_dataset.store_paths["iss_drift_corrected"]
    filter_positions(
        iss_drift_corrected_source_path, iss_drift_corrected_dest_path, iss_positions
    )

    # Phenotyping BF
    phenotyping_bf_source_path = source_dataset.store_paths["lc_20x"]
    phenotyping_bf_dest_path = dest_dataset.store_paths["lc_20x"]
    filter_positions(
        phenotyping_bf_source_path, phenotyping_bf_dest_path, phenotyping_positions
    )

    # Phenotyping Phase
    phenotyping_phase_source_path = source_dataset.store_paths["lc_20x_phase"]
    phenotyping_phase_dest_path = dest_dataset.store_paths["lc_20x_phase"]
    filter_positions(
        phenotyping_phase_source_path,
        phenotyping_phase_dest_path,
        phenotyping_positions,
    )

    # Phenotyping Phase 2D
    pp2d_source_path = source_dataset.store_paths["lc_20x_phase_2d"]
    pp2d_dest_path = dest_dataset.store_paths["lc_20x_phase_2d"]
    filter_positions(pp2d_source_path, pp2d_dest_path, phenotyping_positions)

    # Cell segmentation
    cell_seg_source_path = source_dataset.store_paths["lc_20x_segmentation_cells"]
    cell_seg_dest_path = dest_dataset.store_paths["lc_20x_segmentation_cells"]
    filter_positions(cell_seg_source_path, cell_seg_dest_path, phenotyping_positions)

    # Phenotyping VS
    phenotyping_vs_source_path = source_dataset.store_paths["lc_20x_vs_max_proj"]
    phenotyping_vs_dest_path = dest_dataset.store_paths["lc_20x_vs_max_proj"]
    filter_positions(
        phenotyping_vs_source_path, phenotyping_vs_dest_path, phenotyping_positions
    )

    # Tracking phase
    tracking_phase_source_path = source_dataset.store_paths["lc_5x_phase"]
    tracking_phase_dest_path = dest_dataset.store_paths["lc_5x_phase"]
    filter_positions(
        tracking_phase_source_path, tracking_phase_dest_path, tracking_positions
    )

    # Fluorescence registration beads
    experiment = "ops0033_20250429"
    if not source_dataset.store_paths["lc_20x_beads"].exists():
        convert(experiment, process="20x_beads")

    beads_source_path = source_dataset.store_paths["lc_20x_beads"]
    beads_dest_path = dest_dataset.store_paths["lc_20x_beads"]
    filter_positions(beads_source_path, beads_dest_path, ["0/0/0"])

    # Phase registration beads
    if not source_dataset.store_paths["lc_20x_beads_phase"].exists():
        from cyclops_process.processes.reconstruct import reconstruct

        reconstruct(experiment=experiment, process="20x_beads", local=True)
    beads_phase_source_path = source_dataset.store_paths["lc_20x_beads_phase"]
    beads_phase_dest_path = dest_dataset.store_paths["lc_20x_beads_phase"]
    filter_positions(beads_phase_source_path, beads_phase_dest_path, ["0/0/0"])

    return

def create_for_iss():
    """
    Need: 
    - Stitched
    - Segmented
    """
    from cyclops_process.processes import ops_stitch, segment, spots

    experiment = "ops_testset"
    dataset_in = OpsDataset(experiment)

    output_store_path = dataset_in.store_paths["iss_stitch"]
    if not output_store_path.exists():
        ops_stitch.estimate_and_stitch(
            experiment=None,
            input_store_path=dataset_in.store_paths["iss_drift_corrected"],
            output_config_path=dataset_in.config_paths["iss_stitch"],
            output_store_path=output_store_path,
            flipud=True,
            fliplr=False,
            rot90=0,
        )

    output_store_path_seg = dataset_in.store_paths["iss_segmentation"]
    if not output_store_path_seg.exists():

        segment.segment_and_stitch(
            experiment=None,
            process="iss",  # need to define process to chose cellpose model
            input_store_path=dataset_in.store_paths["iss"],
            input_config_path=dataset_in.config_paths["iss_stitch"],
            output_store_path=output_store_path_seg,
            flipud=True,
            fliplr=False,
            rot90=0,
            tile_size=(2048, 2048),
            num_workers=None,
        )

    if not dataset_in.append_well("spots", "A/1/0").exists():
        spots.detect_spots(experiment)

    return



if __name__ == "__main__":
    iss_positions = [
        "A/1/005005",
        "A/1/005006",
    ]
    phenotyping_positions = [
        "A/1/030030",
        "A/1/031030",
    ]
    tracking_positions = [
        "A/1/005005",
        "A/1/005006",
    ]

    create_stitching_configs(phenotyping_positions, iss_positions, tracking_positions)
    create_images_testset(phenotyping_positions, iss_positions, tracking_positions)
    create_for_iss()
