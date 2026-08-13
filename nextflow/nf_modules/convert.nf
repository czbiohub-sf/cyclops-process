def kwargsToArgs(Map kwargs) {
    def merged = [experiment: params.experiment] + (kwargs ?: [:])
    merged.collect { k, v -> "--${k} ${v}" }.join(' ')
}

process convert_tiff_to_zarrv3 {
    // Keep the skeleton process from first nextflow iteration
    // label 'cpu_large'

    input:
    val experiment
    
    output:
    tuple val(experiment), path("${experiment}_convert_iss.done"), emit: done
    // TODO: see if python level arguments can be extracted / if they need to be
    script:
    """ 
    echo "=== SLURM Environment ==="
    echo "Node: \$(hostname)"
    echo "Partition: \$SLURM_JOB_PARTITION"
    echo "CPUs allocated: \$SLURM_CPUS_ON_NODE"
    echo "Memory: \$SLURM_MEM_PER_NODE"

    uv run python ${projectDir}/bin/dispatch_cli.py convert_tiff_to_zarrv3 ${kwargsToArgs(task.ext.kwargs)}
    
    touch ${experiment}_convert_iss.done
    """
}

process stack_symlinks {
    input:
    tuple val(experiment), val(convert_done) // TODO: think about what inputs we actually need

    output:
    tuple val(experiment), path("${experiment}_stack_symlinks_iss.done"), emit: done

    script:
    """
    echo "Stacking symlinks"
    uv run python ${projectDir}/bin/dispatch_cli.py stack_symlinks ${kwargsToArgs(task.ext.kwargs)}
    
    touch ${experiment}_stack_symlinks_iss.done 
    """
}


process correct_cycle_drift {
    input:
    tuple val(experiment), val(symlink_done) // TODO: think about what inputs we actually need

    output:
    tuple val(experiment), path("${experiment}_correct_cycle_drift.done"), emit: done

    script:
    """
    # Ensure 3-assembly/ISS directory exists for drift correction output
    experiment_output_dir="\${OPS_OUTPUT_BASE_DIR:?OPS_OUTPUT_BASE_DIR is not set}/${experiment}/3-assembly"
    mkdir -p "\$experiment_output_dir/ISS"
    echo "Created ISS assembly directory: \$experiment_output_dir/ISS"

    uv run python ${projectDir}/bin/dispatch_cli.py correct_cycle_drift ${kwargsToArgs(task.ext.kwargs)}
    touch ${experiment}_correct_cycle_drift.done
    """
}


process estimate_stitch_parameters {
    input:
    tuple val(experiment), val(cycle_done) // TODO: think about what inputs we actually need

    output:
    tuple val(experiment), path("${experiment}_estimate_stitch_parameters.done"), emit: done

    script:
    """
    uv run python ${projectDir}/bin/dispatch_cli.py estimate_stitch_parameters ${kwargsToArgs(task.ext.kwargs)}

    touch ${experiment}_estimate_stitch_parameters.done
    """
}


process segment_and_stitch {
    input:
    tuple val(experiment), val(estimate_stitch_done) // TODO: think about what inputs we actually need

    output:
    tuple val(experiment), path("${experiment}_segment_and_stitch.done"), emit: done

    script:
    """
    uv run python ${projectDir}/bin/dispatch_cli.py segment_and_stitch ${kwargsToArgs(task.ext.kwargs)}

    touch ${experiment}_segment_and_stitch.done
    """
}

process estimate_and_stitch {
    input:
    tuple val(experiment), val(segment_stitch_done)

    output:
    tuple val(experiment), path("${experiment}_estimate_and_stitch.done"), emit: done

    script:
    """
    uv run python ${projectDir}/bin/dispatch_cli.py estimate_and_stitch ${kwargsToArgs(task.ext.kwargs)}

    touch ${experiment}_estimate_and_stitch.done
    """
}

process iss_snr_bimodal {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("iss_snr_bimodal.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py iss_snr_bimodal ${kwargsToArgs(task.ext.kwargs)}
        touch iss_snr_bimodal.done
        """
}

process merge_spots_base_calling {
    tag "w${well}"
    input:
        tuple val(experiment), val(well), path(_done)
    output:
        tuple val(experiment), val(well), path("merge_spots_base_calling_w${well}.done")
    script:
        """
        mkdir -p "\${OPS_OUTPUT_BASE_DIR}/${experiment}/1-preprocess/in_situ_sequencing/base_calling"
        uv run python ${projectDir}/bin/dispatch_cli.py merge_spots_base_calling ${kwargsToArgs(task.ext.kwargs)} --well ${well}
        touch merge_spots_base_calling_w${well}.done
        """
}

// Convert the ISS registered store to v3 right after merge (ISS is independent
// of the track/pheno chains) and async-delete the v2 source. Submits its own
// per-position convert jobs and waits.
process convert_iss_to_v3 {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("convert_iss_to_v3.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py convert_iss_to_v3 ${kwargsToArgs(task.ext.kwargs)}
        touch convert_iss_to_v3.done
        """
}

process optimize_failed_rounds {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("optimize_failed_rounds.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py optimize_failed_rounds ${kwargsToArgs(task.ext.kwargs)}
        touch optimize_failed_rounds.done
        """
}

process get_metrics {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("get_metrics.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py get_metrics ${kwargsToArgs(task.ext.kwargs)} --n_rounds ${params.n_rounds}
        touch get_metrics.done
        """
}


// Fan-out processes below are tagged to distinguish parallel instances (per well / per round pair)

process register_iss_seg_to_nucleus {
    tag "w${well}"
    input:
        tuple val(experiment), path(_done), val(well)
    output:
        tuple val(experiment), val(well), path("register_iss_seg_to_nuc_w${well}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py register_iss_seg_to_nucleus ${kwargsToArgs(task.ext.kwargs)} --well ${well}
        touch register_iss_seg_to_nuc_w${well}.done
        """
}

process register_iss_nucleus_to_round0 {
    tag "w${well}"
    input:
        tuple val(experiment), val(well), path(_done)
    output:
        tuple val(experiment), val(well), path("register_iss_nuc_to_r0_w${well}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py register_iss_nucleus_to_round0 ${kwargsToArgs(task.ext.kwargs)} --well ${well}
        touch register_iss_nuc_to_r0_w${well}.done
        """
}

process register_iss_round_pair {
    tag "w${well}_r${round_source}"
    input:
        tuple val(experiment), val(well), path(_done), val(round_source)
    output:
        tuple val(experiment), val(well), path("register_iss_round_pair_w${well}_r${round_source}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py register_iss_round_pair ${kwargsToArgs(task.ext.kwargs)} \
            --well ${well} --round_source ${round_source} --round_target ${round_source - 1}
        touch register_iss_round_pair_w${well}_r${round_source}.done
        """
}

process precreate_iss_registered {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("precreate_iss_registered.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py precreate_iss_registered \
            ${kwargsToArgs(task.ext.kwargs)}
        touch precreate_iss_registered.done
        """
}

process finalize_iss_registration {
    tag "w${well}"
    input:
        tuple val(experiment), val(well), path(_dones)
    output:
        tuple val(experiment), val(well), path("finalize_iss_registration_w${well}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py finalize_iss_registration ${kwargsToArgs(task.ext.kwargs)} --well ${well} --n_rounds ${params.n_rounds}
        touch finalize_iss_registration_w${well}.done
        """
}


// ─── Pheno / tracking processes ───────────────────────────────────────────────

// First live-cell step: dragonfly raw tiffs -> zarr on the fast partition.
// Submits its own per-dataset SLURM jobs and waits (run_locally-style).
// Feeds both link_tracking and link_phenotyping.
process convert_raw {
    input:
        val experiment
    output:
        tuple val(experiment), path("convert_raw.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py convert_raw ${kwargsToArgs(task.ext.kwargs)}
        touch convert_raw.done
        """
}

process link_phenotyping {
    input:
        tuple val(experiment), path(_convert_raw_done)
    output:
        tuple val(experiment), path("link_phenotyping.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py link_phenotyping ${kwargsToArgs(task.ext.kwargs)}
        touch link_phenotyping.done
        """
}

process link_tracking {
    input:
        tuple val(experiment), path(_convert_raw_done)
    output:
        tuple val(experiment), path("link_tracking.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py link_tracking ${kwargsToArgs(task.ext.kwargs)}
        touch link_tracking.done
        """
}

process correct_distortion_bf {
    input:
        tuple val(experiment), path(_tracking_done)
    output:
        tuple val(experiment), path("correct_distortion_bf.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py correct_distortion ${kwargsToArgs(task.ext.kwargs)}
        touch correct_distortion_bf.done
        """
}

process reconstruct_track {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("reconstruct_track.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py reconstruct ${kwargsToArgs(task.ext.kwargs)}
        touch reconstruct_track.done
        """
}

process calibrate_tilt_track {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("calibrate_tilt_track.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py calibrate_tilt ${kwargsToArgs(task.ext.kwargs)}
        touch calibrate_tilt_track.done
        """
}

// NOTE: reconstruct_tilt_corrected_track superseded by setup/job fan-out below.
// Kept for standalone recovery: dispatch_cli.py reconstruct_tilt_corrected
// process reconstruct_tilt_corrected_track { ... }

// NOTE: reconstruct_tilt_corrected_setup_track superseded by get_n_positions_track below.
// process reconstruct_tilt_corrected_setup_track { ... }

process get_n_positions_track {
    input:
        tuple val(experiment), val(well), path(_reconstruct_done), path(_calibrate_done)
    output:
        tuple val(experiment), val(well), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py get_n_positions \
            --well ${well} ${kwargsToArgs(task.ext.kwargs)}
        """
}

process reconstruct_tilt_corrected_job_track {
    tag "${well}:${position_start}-${position_end}"
    input:
        tuple val(experiment), val(well), val(position_start), val(position_end)
    output:
        tuple val(experiment), path("reconstruct_tilt_corrected_job_track.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py reconstruct_tilt_corrected_job \
            --well ${well} --position_start ${position_start} --position_end ${position_end} \
            ${kwargsToArgs(task.ext.kwargs)}
        touch reconstruct_tilt_corrected_job_track.done
        """
}

process vs_preprocess_track {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("vs_preprocess_track.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_preprocess ${kwargsToArgs(task.ext.kwargs)}
        touch vs_preprocess_track.done
        """
}

// NOTE: vs_inference_track superseded by vs_inference_setup/job_track below.
// Kept for standalone recovery invocations via dispatch_cli.py virtual_staining_inference.
// process vs_inference_track {
//     input:  tuple val(experiment), path(_done)
//     output: tuple val(experiment), path("vs_inference_track.done")
//     script: """
//         uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_inference ${kwargsToArgs(task.ext.kwargs)}
//         touch vs_inference_track.done
//     """
// }

process vs_inference_setup_track {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_inference_setup ${kwargsToArgs(task.ext.kwargs)}
        """
}

process vs_inference_job_track {
    tag "job${job_index}"
    input:
        tuple val(experiment), val(job_index), val(num_jobs), val(num_positions)
    output:
        tuple val(experiment), path("vs_inference_job_track_${job_index}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_inference_job \
            ${kwargsToArgs(task.ext.kwargs)} \
            --job_index ${job_index} --num_jobs ${num_jobs} --num_positions ${num_positions}
        touch vs_inference_job_track_${job_index}.done
        """
}

// NOTE: vs_combine_track is superseded by vs_combine_setup/stream/validate_track below.
// Kept for standalone recovery invocations via dispatch_cli.py virtual_staining_combine_only.
// process vs_combine_track {
//     input:  tuple val(experiment), path(_done)
//     output: tuple val(experiment), path("vs_combine_track.done")
//     script: """
//         uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_combine_only ${kwargsToArgs(task.ext.kwargs)}
//         touch vs_combine_track.done
//     """
// }

process vs_combine_setup_track {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_combine_setup ${kwargsToArgs(task.ext.kwargs)}
        """
}

process vs_combine_stream_track {
    tag "store${store_index}"
    input:
        tuple val(experiment), val(store_index)
    output:
        tuple val(experiment), path("vs_combine_stream_track_${store_index}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_combine_stream ${kwargsToArgs(task.ext.kwargs)} --store_index ${store_index}
        touch vs_combine_stream_track_${store_index}.done
        """
}

process vs_combine_validate_track {
    input:
        tuple val(experiment), val(_dones)
    output:
        tuple val(experiment), path("vs_combine_track.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_combine_validate ${kwargsToArgs(task.ext.kwargs)}
        touch vs_combine_track.done
        """
}

process estimate_stitch_parameters_track {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("estimate_stitch_parameters_track.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py estimate_stitch_parameters ${kwargsToArgs(task.ext.kwargs)}
        touch estimate_stitch_parameters_track.done
        """
}

process segment_and_stitch_track {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("segment_and_stitch_track.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py segment_and_stitch ${kwargsToArgs(task.ext.kwargs)}
        touch segment_and_stitch_track.done
        """
}

process estimate_and_stitch_track {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("estimate_and_stitch_track.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py estimate_and_stitch ${kwargsToArgs(task.ext.kwargs)}
        touch estimate_and_stitch_track.done
        """
}

process calibrate_tilt_pheno {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("calibrate_tilt_pheno.done")
    script:
        """
        export OMP_NUM_THREADS=1
        uv run python ${projectDir}/bin/dispatch_cli.py calibrate_tilt ${kwargsToArgs(task.ext.kwargs)}
        touch calibrate_tilt_pheno.done
        """
}

// NOTE: reconstruct_tilt_corrected_pheno superseded by setup/job fan-out below.
// process reconstruct_tilt_corrected_pheno { ... }

// NOTE: reconstruct_tilt_corrected_setup_pheno superseded by get_n_positions_pheno below.
// process reconstruct_tilt_corrected_setup_pheno { ... }

process get_n_positions_pheno {
    input:
        tuple val(experiment), val(well), path(_reconstruct_done), path(_calibrate_done)
    output:
        tuple val(experiment), val(well), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py get_n_positions \
            --well ${well} ${kwargsToArgs(task.ext.kwargs)}
        """
}

process reconstruct_tilt_corrected_job_pheno {
    tag "${well}:${position_start}-${position_end}"
    input:
        tuple val(experiment), val(well), val(position_start), val(position_end)
    output:
        tuple val(experiment), path("reconstruct_tilt_corrected_job_pheno.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py reconstruct_tilt_corrected_job \
            --well ${well} --position_start ${position_start} --position_end ${position_end} \
            ${kwargsToArgs(task.ext.kwargs)}
        touch reconstruct_tilt_corrected_job_pheno.done
        """
}

process reconstruct_pheno {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("reconstruct_pheno.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py reconstruct ${kwargsToArgs(task.ext.kwargs)}
        touch reconstruct_pheno.done
        """
}

process create_max_projection_fluor {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("create_max_projection_fluor.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py create_max_projection ${kwargsToArgs(task.ext.kwargs)}
        touch create_max_projection_fluor.done
        """
}

// Flatfield-correct the fluorescence channels (no-op internally on phase-only
// experiments). Runs after the fluor max projection, before the VS chain.
process correct_flatfield_fluor {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("correct_flatfield_fluor.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py correct_flatfield ${kwargsToArgs(task.ext.kwargs)}
        touch correct_flatfield_fluor.done
        """
}

process vs_preprocess_pheno {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("vs_preprocess_pheno.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_preprocess ${kwargsToArgs(task.ext.kwargs)}
        touch vs_preprocess_pheno.done
        """
}

// NOTE: vs_inference_pheno superseded by vs_inference_setup/job_pheno below.
// Kept for standalone recovery invocations via dispatch_cli.py virtual_staining_inference.
// process vs_inference_pheno {
//     input:  tuple val(experiment), path(_done)
//     output: tuple val(experiment), path("vs_inference_pheno.done")
//     script: """
//         uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_inference ${kwargsToArgs(task.ext.kwargs)}
//         touch vs_inference_pheno.done
//     """
// }

process vs_inference_setup_pheno {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_inference_setup ${kwargsToArgs(task.ext.kwargs)}
        """
}

process vs_inference_job_pheno {
    tag "job${job_index}"
    input:
        tuple val(experiment), val(job_index), val(num_jobs), val(num_positions)
    output:
        tuple val(experiment), path("vs_inference_job_pheno_${job_index}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_inference_job \
            ${kwargsToArgs(task.ext.kwargs)} \
            --job_index ${job_index} --num_jobs ${num_jobs} --num_positions ${num_positions}
        touch vs_inference_job_pheno_${job_index}.done
        """
}

// NOTE: vs_combine_pheno is superseded by vs_combine_setup/stream/validate_pheno below.
// Kept for standalone recovery invocations via dispatch_cli.py virtual_staining_combine_only.
// process vs_combine_pheno {
//     input:  tuple val(experiment), path(_done)
//     output: tuple val(experiment), path("vs_combine_pheno.done")
//     script: """
//         uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_combine_only ${kwargsToArgs(task.ext.kwargs)}
//         touch vs_combine_pheno.done
//     """
// }

process vs_combine_setup_pheno {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_combine_setup ${kwargsToArgs(task.ext.kwargs)}
        """
}

process vs_combine_stream_pheno {
    tag "store${store_index}"
    input:
        tuple val(experiment), val(store_index)
    output:
        tuple val(experiment), path("vs_combine_stream_pheno_${store_index}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_combine_stream ${kwargsToArgs(task.ext.kwargs)} --store_index ${store_index}
        touch vs_combine_stream_pheno_${store_index}.done
        """
}

process vs_combine_validate_pheno {
    input:
        tuple val(experiment), val(_dones)
    output:
        tuple val(experiment), path("vs_combine_pheno.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py virtual_staining_combine_validate ${kwargsToArgs(task.ext.kwargs)}
        touch vs_combine_pheno.done
        """
}

process create_max_projection_lc20x {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("create_max_projection_lc20x.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py create_max_projection ${kwargsToArgs(task.ext.kwargs)}
        touch create_max_projection_lc20x.done
        """
}

process estimate_stitch_parameters_pheno {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("estimate_stitch_parameters_pheno.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py estimate_stitch_parameters ${kwargsToArgs(task.ext.kwargs)}
        touch estimate_stitch_parameters_pheno.done
        """
}

// Auto fluor->Phase2D channel registration. NF-native fan-out (README 3a):
// setup lists the eligible fluor channels (auto-loaded from the experiment
// config; empty on phase-only experiments), each job registers ONE channel.
process channel_registration_setup {
    input:
        tuple val(experiment), path(_done)
    // pass the upstream done through for the zero-channel (phase-only) case
    output:
        tuple val(experiment), path(_done), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py channel_registration_setup ${kwargsToArgs(task.ext.kwargs)}
        """
}

process channel_registration_job {
    tag "${channel}"
    input:
        tuple val(experiment), val(channel)
    output:
        tuple val(experiment), path("channel_reg_${channel}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py channel_registration_job ${kwargsToArgs(task.ext.kwargs)} --channel '${channel}'
        touch channel_reg_${channel}.done
        """
}

process prepare_unified_pheno_tiles {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("prepare_unified_pheno_tiles.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py prepare_unified_pheno_tiles ${kwargsToArgs(task.ext.kwargs)}
        touch prepare_unified_pheno_tiles.done
        """
}

process estimate_and_stitch_pheno {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("estimate_and_stitch_pheno.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py estimate_and_stitch ${kwargsToArgs(task.ext.kwargs)}
        touch estimate_and_stitch_pheno.done
        """
}

process viscy_normalize {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("viscy_normalize.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py viscy_normalize ${kwargsToArgs(task.ext.kwargs)}
        touch viscy_normalize.done
        """
}

process build_pyramids_setup {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py build_pyramids_setup \
            ${kwargsToArgs(task.ext.kwargs)}
        """
}

process build_pyramids_position_job {
    tag "${store_key}:${position}"
    input:
        tuple val(experiment), val(store_key), val(position)
    output:
        tuple val(experiment), path("napari_pos_${store_key}_${position.replaceAll('/', '_')}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py build_pyramids_position_job \
            ${kwargsToArgs(task.ext.kwargs)} \
            --store_key '${store_key}' --position '${position}'
        touch napari_pos_${store_key}_${position.replaceAll('/', '_')}.done
        """
}

process auto_register {
    input:
        tuple val(experiment), path(_pheno_done), path(_track_done)
    output:
        tuple val(experiment), path("auto_register.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py submit_registration_jobs ${kwargsToArgs(task.ext.kwargs)}
        touch auto_register.done
        """
}

process submit_tracking_jobs {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("submit_tracking_jobs.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py submit_tracking_jobs ${kwargsToArgs(task.ext.kwargs)}
        touch submit_tracking_jobs.done
        """
}

process cell_segmentation {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("cell_segmentation.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py submit_cell_segmentation_jobs ${kwargsToArgs(task.ext.kwargs)}
        touch cell_segmentation.done
        """
}

// Native-20x nuclei segmentation: writes the `nuclear_seg` label into
// phenotyping_v3.zarr; must precede auto_register/tracking which read it.
// NF-native fan-out (README 3a): setup enumerates positions, each job segments
// ONE position (GPU, no nested slurm), a groupTuple downstream collects them.
process nuclei_segmentation_setup {
    input:
        tuple val(experiment), path(_done)
    // pass the upstream done through so the zero-position case still has a token
    output:
        tuple val(experiment), path(_done), stdout
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py nuclei_segmentation_setup ${kwargsToArgs(task.ext.kwargs)}
        """
}

process nuclei_segmentation_job {
    tag "${position}"
    input:
        tuple val(experiment), val(position)
    output:
        tuple val(experiment), path("nuclei_seg_${position.replaceAll('/', '_')}.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py nuclei_segmentation_job ${kwargsToArgs(task.ext.kwargs)} --position '${position}'
        touch nuclei_seg_${position.replaceAll('/', '_')}.done
        """
}

process link_calls_tracks {
    input:
        tuple val(experiment), path(_iss_done), path(_pheno_done)
    output:
        tuple val(experiment), path("link_calls_tracks.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py link_calls_tracks ${kwargsToArgs(task.ext.kwargs)}
        touch link_calls_tracks.done
        """
}

// Audit + fix all v3 stores (missing pyramids, overlays, labels, clims).
// Submits its own fix jobs internally and waits. Gates recompute_metrics.
process fix_v3_stores {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("fix_v3_stores.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py fix_v3_stores ${kwargsToArgs(task.ext.kwargs)}
        touch fix_v3_stores.done
        """
}

process recompute_metrics {
    input:
        tuple val(experiment), path(_done)
    output:
        tuple val(experiment), path("recompute_metrics.done")
    script:
        """
        uv run python ${projectDir}/bin/dispatch_cli.py recompute_metrics ${kwargsToArgs(task.ext.kwargs)}
        touch recompute_metrics.done
        """
}