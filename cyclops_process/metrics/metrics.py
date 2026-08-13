"""
OPS Metrics Generation Module

This module generates comprehensive metrics and visualizations for OPS experiments,
including ISS quality metrics, SNR analysis, growth effects, and more.

CLI Usage
---------
Run all metrics for a single experiment (supports shorthand names):
    python -m cyclops_process.metrics.metrics 33 --method mine
    python -m cyclops_process.metrics.metrics 33 -m mine
    python -m cyclops_process.metrics.metrics -e 33 --method mine

Run with force regeneration (ignore cached results):
    python -m cyclops_process.metrics.metrics 33 --method probabilistic --force

Run a specific function only:
    python -m cyclops_process.metrics.metrics -e ops0033_20250429 -m probabilistic \\
        --function plot_cells_per_gene_histogram

For batch processing across all experiments, use:
    python -m cyclops_process.processes.run --all --rerun get_metrics

Arguments
---------
experiment (positional) or -e/--exp : str
    Experiment name or shorthand (e.g., "33", "ops33", "ops0033_20250429").
    Shorthand names are auto-resolved to full experiment names.
-m/--method : str
    Base calling method: 'probabilistic', 'mine', or 'both'. Default: 'probabilistic'.
--confidence-threshold : float
    Confidence threshold for probabilistic method. Default: 0.95.
--function : str
    Specific function to run. Options: 'all', 'get_metrics', 'plot_cells_per_gene_histogram',
    'plot_top_genes_by_cell_count', 'plot_top_guides_by_cell_count',
    'plot_guide_entropy_vs_cell_count', 'cell_count_vs_growth_effect',
    'plot_link_cells_per_gene_histogram', 'plot_link_top_genes_by_cell_count',
    'plot_link_top_guides_by_cell_count'. Default: 'all'.
-f/--force : flag
    Force regeneration of all outputs even if they already exist (skips cache).

Outputs
-------
Results are saved to the experiment's results_iss directory, including:
- plate_stats.csv: Summary statistics for each well
- Various PNG plots for visualization
- SNR stats cache (snr_stats_cache_{method}.csv) for faster re-runs
"""

from ops_utils.data.experiment import OpsDataset
import pandas as pd
from prettytable import PrettyTable


import os
import sys

sys.path.insert(0, os.getcwd())

from ops_utils.profiling.decorators import versioned_function
from cyclops_process.metrics.plate_stats.metrics_over_time import plot_metrics_over_time
from cyclops_process.metrics.plate_stats.metrics_probs_ISS import main_for_metrics

# Re-export from iss_metrics for backward compatibility
from cyclops_process.metrics.plate_stats.plate_stats_stitch_metrics import (
    generate_all_stitch_confidence_heatmaps,
)
from cyclops_process.metrics.plate_stats.iss_confluency import (
    confluency,
)
from cyclops_process.metrics.plate_stats.iss_metrics import (
    count_bases,
    create_freq_tables,
    _save_rounds_manifest,
    read_accuracy_by_round,
)
from cyclops_process.metrics.plate_stats.iss_histrogram import (
    plot_cells_per_gene_histogram,
    plot_top_guides_by_cell_count,
    plot_top_genes_by_cell_count,
    plot_guide_entropy_vs_cell_count,
)
from cyclops_process.metrics.plate_stats.plate_stats_track_metrics import (
    plot_link_cells_per_gene_histogram,
    plot_link_top_genes_by_cell_count,
    plot_link_top_guides_by_cell_count,
)

from cyclops_process.metrics.plate_stats.iss_timing import (
    timing_plot,
)

from cyclops_process.metrics.plate_stats.hamming_distance import (
    hamming_distance,
)

from cyclops_process.metrics.plate_stats.iss_heatmaps import (
    read_accuracy,
    cells_with_reads_heatmaps,
)

from cyclops_process.metrics.plate_stats.iss_stats import (
    statistics,
)

from cyclops_process.metrics.plate_stats.iss_growth_effect import (
    cell_count_vs_growth_effect,
)

from cyclops_process.metrics.plate_stats.iss_spatial_coherence import (
    get_spatial_coherence_stats_for_well,
)

from cyclops_process.metrics.plate_stats.iss_cell_size import (
    cell_size_metrics,
)















@versioned_function("1.0")
def get_metrics(
    experiment,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    iss_rounds: list[int] = None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
    force: bool = False,
    n_rounds: int = 9,
) -> None:

    print(f"Metrics using base calling METHOD: {method}")
    print(f"Failed rounds by well: {failed_rounds_by_well}")
    if force:
        print("Force mode: regenerating all outputs")

    if iss_rounds is None:
        iss_rounds = list(range(n_rounds + 1))
    
    # create the results directory
    dataset = OpsDataset(experiment, method=method)
    dataset.results_iss.mkdir(parents=True, exist_ok=True)
    _save_rounds_manifest(dataset, iss_rounds, failed_rounds_by_well, method)

    # generate freq tables
    create_freq_tables(
        experiment,
        iss_rounds=iss_rounds,
        method=method,
        failed_rounds_by_well=failed_rounds_by_well,
    )

    # check that results exist
    print(f"ISS ROUNDS: {iss_rounds}")
    try:
        timing_plot(experiment)
    except:
        print("Timing plot failed")
    read_accuracy(
        experiment,
        iss_rounds=iss_rounds,
        method=method,
        failed_rounds_by_well=failed_rounds_by_well,
        force=force,
    )
    read_accuracy_by_round(
        experiment,
        iss_rounds=iss_rounds,
        method=method,
        force=force,
    )
    confluency(experiment, force=force)

    cell_size_stats = {}
    try:
        cell_size_stats = cell_size_metrics(experiment, method=method, force=force)
    except Exception as e:
        print(f"Cell size metrics failed: {e}")

    try:
        cells_with_reads_heatmaps(
            experiment,
            iss_rounds=iss_rounds,
            method=method,
            confidence_threshold=confidence_threshold,
            failed_rounds_by_well=failed_rounds_by_well,
            force=force,
        )
    except Exception as e:
        print(f"Cells-with-reads heatmaps failed: {e}")

    if method == "mine":
        hamming_distance(
            experiment,
            "A/1/0",
            iss_rounds=iss_rounds,
            failed_rounds_by_well=failed_rounds_by_well,
        )

    # snr_by_round(experiment)  # Deprecated: SNR analysis is now handled by main_for_metrics -> snr_analysis_ISS
    try:
        generate_all_stitch_confidence_heatmaps(dataset, experiment, force=force)
    except Exception as e:
        print(f"Stitching confidence heatmaps failed with error: {e}")
        import traceback

        traceback.print_exc()

    count_bases(experiment, iss_rounds=iss_rounds, method=method)

    # --- Generate General and Probabilistic ISS Metrics ---
    prob_stats = {}
    try:
        print("\n--- Generating ISS Signal, Noise, and Crosstalk Metrics ---")
        # This function will now run for any method, but only generates
        # probabilistic-specific plots/stats if method == 'probabilistic'.
        prob_stats = main_for_metrics(
            experiment,
            method=method,
            iss_rounds=iss_rounds,
            failed_rounds_by_well=failed_rounds_by_well,
            force=force,
        )
    except Exception as e:
        print(f"Failed to generate probabilistic ISS metrics: {e}")
        import traceback
        traceback.print_exc()

    # Now run growth effect, which reads the frequency tables and appends to the stats file
    growth_effect_stats = {}
    try:
        growth_effect_stats = cell_count_vs_growth_effect(
            experiment, debug=False, method=method,
            iss_rounds=iss_rounds, failed_rounds_by_well=failed_rounds_by_well,
            force=force,
        )
    except Exception as e:
        print(f"Failed cell count vs growth effect: {e}")

    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    # Run statistics first to generate the frequency tables
    stats_df = statistics(
        experiment,
        iss_rounds=iss_rounds,
        prob_stats=prob_stats,
        growth_effect_stats=growth_effect_stats,
        method=method,
        confidence_threshold=confidence_threshold,
        failed_rounds_by_well=failed_rounds_by_well,
        force=force,
    )

    try:
        plot_cells_per_gene_histogram(
            experiment,
            iss_rounds=iss_rounds,
            method=method,
            confidence_threshold=confidence_threshold,
            failed_rounds_by_well=failed_rounds_by_well,
        )
    except Exception as e:
        print(f"Failed to generate cells per gene histogram: {e}")

    try:
        plot_top_genes_by_cell_count(
            experiment,
            iss_rounds=iss_rounds,
            method=method,
            confidence_threshold=confidence_threshold,
            top_n=50,
            failed_rounds_by_well=failed_rounds_by_well,
        )
    except Exception as e:
        print(f"Failed to generate top genes by cell count plot: {e}")

    try:
        plot_top_guides_by_cell_count(
            experiment,
            iss_rounds=iss_rounds,
            method=method,
            failed_rounds_by_well=failed_rounds_by_well,
            confidence_threshold=confidence_threshold,
            top_n=50,
        )
    except Exception as e:
        print(f"Failed to generate top guides by cell count plot: {e}")

    try:
        plot_guide_entropy_vs_cell_count(
            experiment,
            iss_rounds=iss_rounds,
            method=method,
            failed_rounds_by_well=failed_rounds_by_well,
            confidence_threshold=confidence_threshold,
        )
    except Exception as e:
        print(f"Failed to generate guide entropy vs cell count plot: {e}")

    # --- Link-level (post-tracking) histogram plots ---
    try:
        plot_link_cells_per_gene_histogram(experiment, method=method)
    except Exception as e:
        print(f"Failed to generate linked cells per gene histogram: {e}")

    try:
        plot_link_top_genes_by_cell_count(experiment, method=method, top_n=50)
    except Exception as e:
        print(f"Failed to generate linked top genes by cell count plot: {e}")

    try:
        plot_link_top_guides_by_cell_count(experiment, method=method, top_n=50)
    except Exception as e:
        print(f"Failed to generate linked top guides by cell count plot: {e}")

    # Print statistics as a pretty table
    try:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT STATISTICS SUMMARY - {experiment} ({method})")
        print(f"{'='*80}")
        table = PrettyTable()
        table.field_names = ["Metric"] + list(stats_df.columns)
        table.align = "l"

        def format_stat_value(v):
            """Format a statistic value: integers without decimals, floats with 2."""
            import math
            if isinstance(v, float):
                if math.isnan(v):
                    return "nan"
                if v == int(v):
                    return f"{int(v):,}"
                return f"{v:,.2f}"
            return str(v)

        for idx, row in stats_df.iterrows():
            table.add_row([idx] + [format_stat_value(v) for v in row.values])
        print(table)
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"Could not print statistics table: {e}")

    try:
        print("\n--- Generating Metrics Over Time ---")
        plot_metrics_over_time(experiment, method=method)
    except Exception as e:
        print(f"Failed to generate metrics over time plots: {e}")

    return


@versioned_function("1.0")
def recompute_metrics(
    experiment,
    method: str = "mine",
    confidence_threshold: float = 0.95,
    iss_rounds: list[int] = None,
    failed_rounds_by_well: dict[str, list[int]] | None = None,
) -> None:
    """Re-run get_metrics at end of pipeline to recompute QC with all steps complete."""
    get_metrics(
        experiment,
        method=method,
        confidence_threshold=confidence_threshold,
        iss_rounds=iss_rounds,
        failed_rounds_by_well=failed_rounds_by_well,
        force=True,
    )


if __name__ == "__main__":
    import argparse
    from ops_utils.data.filesystem import resolve_experiment_name

    parser = argparse.ArgumentParser(
        description="Generate metrics and plots for OPS experiments"
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        type=str,
        help="Experiment name or shorthand (e.g., '33', 'ops33', 'ops0033_20250429')",
    )
    parser.add_argument(
        "-e", "--exp",
        type=str,
        dest="experiment_flag",
        help="Experiment name (alternative to positional argument)",
    )
    parser.add_argument(
        "-m", "--method",
        type=str,
        choices=["probabilistic", "mine", "both"],
        default="mine",
        help="Base calling method (default: mine). Use 'both' to run for both methods.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for probabilistic method (default: 0.95)",
    )
    parser.add_argument(
        "--function",
        type=str,
        choices=[
            "all",
            "get_metrics",
            "plot_cells_per_gene_histogram",
            "plot_top_genes_by_cell_count",
            "plot_top_guides_by_cell_count",
            "plot_guide_entropy_vs_cell_count",
            "cell_count_vs_growth_effect",
            "plot_link_histograms",
            "plot_link_cells_per_gene_histogram",
            "plot_link_top_genes_by_cell_count",
            "plot_link_top_guides_by_cell_count",
        ],
        default="all",
        help="Specific function to run (default: all)",
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force regeneration of all outputs even if they already exist",
    )

    args = parser.parse_args()

    # Get experiment from positional arg or -e flag
    exp_input = args.experiment or args.experiment_flag
    if not exp_input:
        parser.error("Experiment required: provide as first argument or use -e/--exp")

    # Resolve shorthand experiment names (e.g., "33" -> "ops0033_20250429")
    exp = resolve_experiment_name(exp_input, autoselect=True)

    # Determine which methods to run
    methods = ["probabilistic", "mine"] if args.method == "both" else [args.method]

    print(f"\n{'#'*80}")
    print(f"Processing experiment: {exp}")
    print(f"{'#'*80}\n")

    for method in methods:
        if len(methods) > 1:
            print(f"\n{'='*60}")
            print(f"Running with method: {method}")
            print(f"{'='*60}\n")

        try:
            if args.function == "all" or args.function == "get_metrics":
                get_metrics(
                    exp,
                    method=method,
                    confidence_threshold=args.confidence_threshold,
                    force=args.force,
                )
            elif args.function == "plot_cells_per_gene_histogram":
                plot_cells_per_gene_histogram(
                    exp,
                    method=method,
                    confidence_threshold=args.confidence_threshold,
                )
            elif args.function == "plot_top_genes_by_cell_count":
                plot_top_genes_by_cell_count(
                    exp,
                    method=method,
                    confidence_threshold=args.confidence_threshold,
                    top_n=50,
                )
            elif args.function == "plot_top_guides_by_cell_count":
                plot_top_guides_by_cell_count(
                    exp,
                    method=method,
                    confidence_threshold=args.confidence_threshold,
                    top_n=50,
                )
            elif args.function == "plot_guide_entropy_vs_cell_count":
                plot_guide_entropy_vs_cell_count(
                    exp,
                    method=method,
                    confidence_threshold=args.confidence_threshold,
                )
            elif args.function == "cell_count_vs_growth_effect":
                cell_count_vs_growth_effect(
                    exp,
                    method=method,
                )
            elif args.function == "plot_link_histograms":
                plot_link_cells_per_gene_histogram(exp, method=method)
                plot_link_top_genes_by_cell_count(exp, method=method, top_n=50)
                plot_link_top_guides_by_cell_count(exp, method=method, top_n=50)
            elif args.function == "plot_link_cells_per_gene_histogram":
                plot_link_cells_per_gene_histogram(exp, method=method)
            elif args.function == "plot_link_top_genes_by_cell_count":
                plot_link_top_genes_by_cell_count(exp, method=method, top_n=50)
            elif args.function == "plot_link_top_guides_by_cell_count":
                plot_link_top_guides_by_cell_count(exp, method=method, top_n=50)
        except Exception as e:
            print(f"ERROR processing {exp} with method {method}: {e}")
            continue
