include {
    convert_tiff_to_zarrv3
    stack_symlinks
    correct_cycle_drift
    estimate_stitch_parameters
    segment_and_stitch
    estimate_and_stitch
    register_iss_seg_to_nucleus
    register_iss_nucleus_to_round0
    register_iss_round_pair
    finalize_iss_registration
    precreate_iss_registered
    iss_snr_bimodal
    merge_spots_base_calling
    convert_iss_to_v3
    optimize_failed_rounds
    get_metrics
    convert_raw
    link_phenotyping
    link_tracking
    correct_distortion_bf
    reconstruct_track
    calibrate_tilt_track
    // reconstruct_tilt_corrected_track  // superseded by setup/job fan-out
    // reconstruct_tilt_corrected_setup_track  // superseded by get_n_positions_track
    get_n_positions_track
    reconstruct_tilt_corrected_job_track
    vs_preprocess_track
    vs_inference_setup_track
    vs_inference_job_track
    vs_combine_setup_track
    vs_combine_stream_track
    vs_combine_validate_track
    estimate_stitch_parameters_track
    segment_and_stitch_track
    estimate_and_stitch_track
    calibrate_tilt_pheno
    // reconstruct_tilt_corrected_pheno  // superseded by setup/job fan-out
    // reconstruct_tilt_corrected_setup_pheno  // superseded by get_n_positions_pheno
    get_n_positions_pheno
    reconstruct_tilt_corrected_job_pheno
    reconstruct_pheno
    create_max_projection_fluor
    correct_flatfield_fluor
    vs_preprocess_pheno
    vs_inference_setup_pheno
    vs_inference_job_pheno
    vs_combine_setup_pheno
    vs_combine_stream_pheno
    vs_combine_validate_pheno
    create_max_projection_lc20x
    estimate_stitch_parameters_pheno
    channel_registration_setup
    channel_registration_job
    prepare_unified_pheno_tiles
    estimate_and_stitch_pheno
    viscy_normalize
    build_pyramids_setup
    build_pyramids_position_job
    auto_register
    submit_tracking_jobs
    cell_segmentation
    nuclei_segmentation_setup
    nuclei_segmentation_job
    link_calls_tracks
    fix_v3_stores
    recompute_metrics
} from "${projectDir}/nf_modules/convert.nf"

if (params.processes.size() == 0) {
    log.warn "No executor configuration provided, using defaults"
}


workflow {
    if (params.processes.size() == 0) {
        log.warn "No executor configuration provided"
        log.warn "Using default executors. Provide -params-file executors.yaml"
    }
    
    convert_done = convert_tiff_to_zarrv3(params.experiment ?: "ops0094_20251217_mark")

    stack_symlinks_done = stack_symlinks(convert_done)
    snr_done            = iss_snr_bimodal(stack_symlinks_done)

    cc_done = correct_cycle_drift(stack_symlinks_done)

    estimate_stitch_done = estimate_stitch_parameters(cc_done)

    // segment_and_stitch runs first; estimate_and_stitch then attaches/materializes
    // nuclear_seg into the stitched store inline (main's flow — the discrete
    // attach_seg_symlink step was retired, see royerlab/ops_process#108), so it
    // must run after segment_and_stitch.
    segment_and_stitch_done  = segment_and_stitch(estimate_stitch_done)
    estimate_and_stitch_done = estimate_and_stitch(segment_and_stitch_done)

    wells_ch  = Channel.fromList(params.wells ?: ['A/1', 'A/2', 'A/3']).map { it.split('/')[1].toInteger() }
    rounds_ch = Channel.of(1..(params.n_rounds ?: 9))

    precreate_done  = precreate_iss_registered(estimate_and_stitch_done)
    seg_nuc_done    = register_iss_seg_to_nucleus(precreate_done.combine(wells_ch))
    nuc_r0_done     = register_iss_nucleus_to_round0(seg_nuc_done)
    round_pair_done = register_iss_round_pair(seg_nuc_done.combine(rounds_ch))

    // finalize (per well) composes transforms only; merge owns the warp.
    finalize_done = finalize_iss_registration(
        round_pair_done.mix(nuc_r0_done).groupTuple(by: [0, 1])
    )

    // shm merge (per well): warp + detect_spots + base_calling fused.
    merge_done = merge_spots_base_calling(finalize_done)

    // Convert the registered ISS store to v3 once all wells' base-calling is done.
    convert_iss_v3_done = convert_iss_to_v3(
        merge_done.groupTuple(by: [0])
                  .map { exp, wells, dones -> tuple(exp, dones[0]) }
    )

    // Optimize per-well failed rounds (writes ops_failed_rounds.yaml + the
    // experiment config) before metrics, which reads that config.
    optimize_failed_rounds_done = optimize_failed_rounds(convert_iss_v3_done)

    get_metrics_done = get_metrics(optimize_failed_rounds_done)

    // ─── Pheno / tracking branch ──────────────────────────────────────────────
    // TODO: extract into named workflow pheno { } in pheno.nf
    if (!params.iss_only) {
    // Live-cell raw conversion feeds both link chains.
    convert_raw_done      = convert_raw(params.experiment ?: "ops0094_20251217_mark")
    link_phenotyping_done = link_phenotyping(convert_raw_done)
    link_tracking_done    = link_tracking(convert_raw_done)

    correct_distortion_bf_done = correct_distortion_bf(
        link_tracking_done
    )

    // calibrate_tilt_track and reconstruct_track both depend only on correct_distortion_bf — run in parallel
    calibrate_tilt_track_done             = calibrate_tilt_track(correct_distortion_bf_done)
    reconstruct_track_done                = reconstruct_track(correct_distortion_bf_done)
    reconstruct_tilt_corrected_track_done = get_n_positions_track(
        reconstruct_track_done.join(calibrate_tilt_track_done)
            .combine(Channel.fromList(params.wells ?: ['A/1', 'A/2', 'A/3']))
            .map { exp, r, c, well -> tuple(exp, well, r, c) }
    ).map { exp, well, n ->
        tuple(exp, well, n.trim().readLines().last().toInteger())
    }.flatMap { exp, well, n ->
        (0..<n).step(25).collect { start ->
            tuple(exp, well, start, Math.min(start + 25, n))
        }
    } | reconstruct_tilt_corrected_job_track
     | groupTuple(by: [0])
     | map { exp, dones -> tuple(exp, dones[0]) }
    vs_preprocess_track_done = vs_preprocess_track(reconstruct_tilt_corrected_track_done)
    vs_inference_setup_track_ch = vs_inference_setup_track(vs_preprocess_track_done)
        .map { exp, s -> def p = s.trim().readLines().last().trim().split(' '); tuple(exp, p[0].toInteger(), p[1].toInteger()) }
    vs_inference_track_done     = vs_inference_job_track(
        vs_inference_setup_track_ch.flatMap { exp, nj, np -> (0..<nj).collect { i -> tuple(exp, i, nj, np) } }
    ).groupTuple(by: [0]).map { exp, dones -> tuple(exp, dones[0]) }
    vs_combine_setup_track_ch = vs_combine_setup_track(vs_inference_track_done)
        .map { exp, n -> tuple(exp, n.trim().readLines().last().toInteger()) }
    vs_stream_track_items     = vs_combine_setup_track_ch
        .flatMap { exp, n -> n > 0 ? (0..<n).collect { i -> tuple(exp, i) } : [] }
    vs_stream_track_done      = vs_combine_stream_track(vs_stream_track_items).groupTuple(by: [0])
    hcs_track_ch              = vs_combine_setup_track_ch.filter { exp, n -> n == 0 }.map { exp, n -> tuple(exp, []) }
    vs_combine_track_done     = vs_combine_validate_track(vs_stream_track_done.mix(hcs_track_ch))

    estimate_stitch_track_done = estimate_stitch_parameters_track(vs_combine_track_done)
    // segment_and_stitch_track runs first; estimate_and_stitch_track materializes nuclear_seg inline
    segment_and_stitch_track_done  = segment_and_stitch_track(estimate_stitch_track_done)
    estimate_and_stitch_track_done = estimate_and_stitch_track(segment_and_stitch_track_done)

    // Pheno sub-branch — runs in parallel with tracking from link_phenotyping_done
    // calibrate_tilt_pheno and reconstruct_pheno both depend only on link_phenotyping — run in parallel
    calibrate_tilt_pheno_done             = calibrate_tilt_pheno(link_phenotyping_done)
    reconstruct_pheno_done                = reconstruct_pheno(link_phenotyping_done)
    reconstruct_tilt_corrected_pheno_done = get_n_positions_pheno(
        reconstruct_pheno_done.join(calibrate_tilt_pheno_done)
            .combine(Channel.fromList(params.wells ?: ['A/1', 'A/2', 'A/3']))
            .map { exp, r, c, well -> tuple(exp, well, r, c) }
    ).map { exp, well, n ->
        tuple(exp, well, n.trim().readLines().last().toInteger())
    }.flatMap { exp, well, n ->
        (0..<n).step(25).collect { start ->
            tuple(exp, well, start, Math.min(start + 25, n))
        }
    } | reconstruct_tilt_corrected_job_pheno
     | groupTuple(by: [0])
     | map { exp, dones -> tuple(exp, dones[0]) }
    // TODO: conditional on fluorescence channels — Python handles the no-fluor case internally
    create_max_projection_fluor_done = create_max_projection_fluor(reconstruct_tilt_corrected_pheno_done)
    correct_flatfield_fluor_done     = correct_flatfield_fluor(create_max_projection_fluor_done)

    vs_preprocess_pheno_done = vs_preprocess_pheno(correct_flatfield_fluor_done)
    vs_inference_setup_pheno_ch = vs_inference_setup_pheno(vs_preprocess_pheno_done)
        .map { exp, s -> def p = s.trim().readLines().last().trim().split(' '); tuple(exp, p[0].toInteger(), p[1].toInteger()) }
    vs_inference_pheno_done     = vs_inference_job_pheno(
        vs_inference_setup_pheno_ch.flatMap { exp, nj, np -> (0..<nj).collect { i -> tuple(exp, i, nj, np) } }
    ).groupTuple(by: [0]).map { exp, dones -> tuple(exp, dones[0]) }
    vs_combine_setup_pheno_ch = vs_combine_setup_pheno(vs_inference_pheno_done)
        .map { exp, n -> tuple(exp, n.trim().readLines().last().toInteger()) }
    vs_stream_pheno_items     = vs_combine_setup_pheno_ch
        .flatMap { exp, n -> n > 0 ? (0..<n).collect { i -> tuple(exp, i) } : [] }
    vs_stream_pheno_done      = vs_combine_stream_pheno(vs_stream_pheno_items).groupTuple(by: [0])
    hcs_pheno_ch              = vs_combine_setup_pheno_ch.filter { exp, n -> n == 0 }.map { exp, n -> tuple(exp, []) }
    vs_combine_pheno_done     = vs_combine_validate_pheno(vs_stream_pheno_done.mix(hcs_pheno_ch))

    create_max_projection_lc20x_done = create_max_projection_lc20x(vs_combine_pheno_done)

    estimate_stitch_pheno_done    = estimate_stitch_parameters_pheno(create_max_projection_lc20x_done)
    // Fluor->Phase2D channel registration — fan out one job per fluor channel,
    // collect, then gate the unified tile prep. Phase-only experiments no-op.
    // (segment_and_stitch_pheno retired: nuclear_seg comes from the native-20x
    // nuclei fan-out, matching the orchestrator.)
    channel_reg_setup_ch = channel_registration_setup(estimate_stitch_pheno_done)
        .map { exp, done, s ->
            tuple(exp, done, s.split('\n').findAll { it.startsWith('CHANNELREG_CH ') }
                              .collect { it.substring('CHANNELREG_CH '.length()).trim() })
        }
    channel_reg_jobs_done = channel_registration_job(
            channel_reg_setup_ch.flatMap { exp, done, chans -> chans.collect { tuple(exp, it) } }
        ).groupTuple(by: [0]).map { exp, dones -> tuple(exp, dones[0]) }
    channel_reg_noop = channel_reg_setup_ch.filter { exp, done, chans -> chans.isEmpty() }
        .map { exp, done, chans -> tuple(exp, done) }
    channel_reg_done               = channel_reg_jobs_done.mix(channel_reg_noop)
    prepare_unified_done           = prepare_unified_pheno_tiles(channel_reg_done)
    estimate_and_stitch_pheno_done = estimate_and_stitch_pheno(prepare_unified_done)

    viscy_normalize_done   = viscy_normalize(estimate_and_stitch_pheno_done)
    build_pyramids_setup_ch = build_pyramids_setup(viscy_normalize_done)
        .flatMap { exp, s ->
            // build_pyramids_setup emits real specs as `PYRAMID_UNIT {store_key}:{position}`.
            // The sentinel prefix makes the fan-out immune to ANY stdout noise -- status
            // lines ("Using store: ...", "Selected N/M ... wells: ...") and
            // import logs -- which would otherwise be mis-parsed into garbage tuples and
            // fail build_pyramids_position_job.
            s.split('\n').findAll { it.startsWith('PYRAMID_UNIT ') }
                .collect { line ->
                    def parts = line.substring('PYRAMID_UNIT '.length()).split(':', 2)
                    tuple(exp, parts[0], parts[1])
                }
        }
    build_pyramids_done = build_pyramids_position_job(build_pyramids_setup_ch)
        .groupTuple(by: [0])
        .map { exp, dones -> tuple(exp, dones[0]) }
    // Native-20x nuclei seg writes nuclear_seg (read by auto_register/tracking).
    // Fan out one job per position; groupTuple collects, then auto_register.
    nuclei_setup_ch = nuclei_segmentation_setup(build_pyramids_done)
        .map { exp, done, s ->
            tuple(exp, done, s.split('\n').findAll { it.startsWith('NUCSEG_POS ') }
                              .collect { it.substring('NUCSEG_POS '.length()).trim() })
        }
    nuclei_seg_jobs_done = nuclei_segmentation_job(
            nuclei_setup_ch.flatMap { exp, done, positions -> positions.collect { tuple(exp, it) } }
        ).groupTuple(by: [0]).map { exp, dones -> tuple(exp, dones[0]) }
    // Zero-position passthrough: reuse the upstream done so auto_register still fires.
    nuclei_seg_noop = nuclei_setup_ch.filter { exp, done, positions -> positions.isEmpty() }
        .map { exp, done, positions -> tuple(exp, done) }
    nuclei_seg_done        = nuclei_seg_jobs_done.mix(nuclei_seg_noop)
    auto_register_done     = auto_register(nuclei_seg_done.join(estimate_and_stitch_track_done))
    submit_tracking_done   = submit_tracking_jobs(auto_register_done)
    cell_segmentation_done = cell_segmentation(submit_tracking_done)

    // Join ISS and pheno branches
    link_calls_tracks_done = link_calls_tracks(
        get_metrics_done.join(cell_segmentation_done)
    )
    // build_iss_overlay removed (not in orchestrator; fix_v3_stores audits+fixes overlays).
    fix_v3_stores_done     = fix_v3_stores(link_calls_tracks_done)
    recompute_metrics(fix_v3_stores_done)
    } // end if (!params.iss_only)
}