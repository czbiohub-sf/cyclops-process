"""
nd-embedding-atlas wrapper for OPS phenotyping data.

This module bridges the OPS feature extraction pipeline with nd-embedding-atlas
for interactive exploration of cell embeddings linked to source images.

The library now reads h5ad files directly (lazy-backed) and auto-detects
spatial columns (well, x_global_pheno, y_global_pheno, bbox, t), so the only
preprocessing still required is merging UMAP coordinates from a separate CSV
into the h5ad if they aren't already embedded.

Usage:
------
# Step 1: Prepare data (merges UMAP CSV into h5ad, only needed once)
python -m cyclops_process.napari.nd_embed prepare -e ops0094_20251217

# Step 2: Launch viewer (fast, reads h5ad directly)
python -m cyclops_process.napari.nd_embed serve -e ops0094_20251217

# Or do both in one command:
python -m cyclops_process.napari.nd_embed launch -e ops0094_20251217

# Preview mode (subsample for development/testing)
python -m cyclops_process.napari.nd_embed launch -e 94 --preview 10000

Python API:
-----------
>>> from cyclops_process.napari.nd_embed import prepare_embedding_data, serve_embedding_viewer
>>> prepare_embedding_data("ops0094_20251217")  # Run once
>>> serve_embedding_viewer("ops0094_20251217")  # Launch viewer

Requirements:
-------------
- conda environment: nd_embed (create per the nd_embed setup)
- pip install git+https://github.com/czbiohub-sf/nd-embedding-atlas.git@main
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
import webbrowser
import threading
import time

import numpy as np
import pandas as pd
from cyclops_process.paths import BASE_PATH

sys.path.insert(0, os.getcwd())

BASE_DIR = Path(
    os.environ.get("OPS_OUTPUT_BASE_DIR", f"{BASE_PATH}")
)


def _experiment_paths(experiment: str) -> dict[str, Path]:
    """Resolve standard OPS experiment paths without importing OpsDataset."""
    exp_dir = BASE_DIR / experiment
    assembly = exp_dir / "3-assembly"
    analysis = assembly / "feature_extraction/v1/"
    return {
        "analysis": analysis,
        "pheno_assembled_v3": assembly / "phenotyping_v3.zarr",
    }


def get_cache_paths(experiment: str, subsample: Optional[int] = None) -> tuple[Path, Path]:
    """Get the source h5ad and the prepared zarr (with UMAP embedded) paths."""
    paths = _experiment_paths(experiment)
    source = paths["analysis"] / f"{experiment}_cell_features.h5ad"
    cache_dir = paths["analysis"] / "_nd_embed_cache"
    suffix = f"_{subsample}cells" if subsample else ""
    prepared = cache_dir / f"{experiment}_cell_features{suffix}.zarr"
    return source, prepared


def prepare_embedding_data(
    experiment: str,
    umap_csv_path: Optional[Path] = None,
    subsample: Optional[int] = None,
    output_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """
    Prepare embedding data by merging UMAP coordinates and converting to zarr.

    Reads the source h5ad, adds X_umap from the UMAP CSV, optionally
    subsamples, and writes a zarr store for fast lazy access by the viewer.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., 'ops0094_20251217')
    umap_csv_path : Path, optional
        Path to UMAP coordinates CSV
    subsample : int, optional
        Subsample to this many cells
    output_path : Path, optional
        Output path for zarr store. If None, uses default cache location.
    overwrite : bool
        Whether to overwrite existing zarr store

    Returns
    -------
    Path to the prepared zarr store
    """
    import anndata as ad

    paths = _experiment_paths(experiment)

    # Determine output path
    source_h5ad, default_output = get_cache_paths(experiment, subsample)
    if output_path is None:
        output_path = default_output

    # Check if already exists and not overwriting
    if output_path.exists() and not overwrite:
        print(f"Prepared data already exists: {output_path}")
        print("  Use --overwrite to regenerate")
        return output_path

    # Resolve UMAP CSV path
    if umap_csv_path is None:
        umap_csv_path = paths["analysis"] / "graphs" / "1_cell_level" / "3_embedding" / "umap_coordinates.csv"

    print(f"Loading cell features from: {source_h5ad}")
    print(f"Loading UMAP coordinates from: {umap_csv_path}")

    if not source_h5ad.exists():
        raise FileNotFoundError(f"Cell features not found: {source_h5ad}")
    if not umap_csv_path.exists():
        raise FileNotFoundError(f"UMAP coordinates not found: {umap_csv_path}")

    # Load h5ad
    print("Loading AnnData (this may take a few minutes for large datasets)...")
    adata = ad.read_h5ad(source_h5ad)
    print(f"  Shape: {adata.shape}")

    # Load and merge UMAP coordinates
    print("Loading UMAP coordinates...")
    umap_df = pd.read_csv(umap_csv_path, index_col=0)
    print(f"  UMAP shape: {umap_df.shape}")

    X_umap = umap_df[["umap_1", "umap_2"]].values
    if len(X_umap) == adata.n_obs:
        adata.obsm["X_umap"] = X_umap
    else:
        print(f"  Warning: UMAP rows ({len(umap_df)}) != AnnData obs ({adata.n_obs})")
        print("  Aligning by index position...")
        aligned_umap = np.full((adata.n_obs, 2), np.nan)
        n = min(len(X_umap), adata.n_obs)
        aligned_umap[:n] = X_umap[:n]
        adata.obsm["X_umap"] = aligned_umap
        n_valid = np.sum(~np.isnan(aligned_umap[:, 0]))
        print(f"  Aligned {n_valid}/{adata.n_obs} cells with UMAP coordinates")

    # Subsample if requested
    if subsample is not None and subsample < adata.n_obs:
        print(f"Subsampling to {subsample} cells...")
        indices = np.random.choice(adata.n_obs, size=subsample, replace=False)
        indices = np.sort(indices)
        adata = adata[indices].copy()
        print(f"  New shape: {adata.shape}")

    # Ensure obs object columns are string-typed for zarr compatibility
    for col in adata.obs.columns:
        if adata.obs[col].dtype == object:
            adata.obs[col] = adata.obs[col].astype(str)

    # Write zarr store
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    print(f"Writing zarr store: {output_path}")
    adata.write_zarr(output_path)

    print(f"\nData prepared successfully!")
    print(f"  zarr: {output_path}")
    preview_arg = f" --preview {subsample}" if subsample else ""
    print(f"\nTo launch the viewer, run:")
    print(f"  python -m cyclops_process.napari.nd_embed serve -e {experiment}{preview_arg}")

    return output_path


def serve_embedding_viewer(
    experiment: str,
    subsample: Optional[int] = None,
    data_path: Optional[Path] = None,
    host: str = "localhost",
    port: int = 5055,
    use_plate: bool = True,
    open_browser: bool = True,
) -> None:
    """
    Launch nd-embedding-atlas viewer.

    The library auto-detects well, x_global_pheno/y_global_pheno, bbox,
    and t columns from the data.

    Parameters
    ----------
    experiment : str
        Experiment name (e.g., 'ops0094_20251217')
    subsample : int, optional
        Expected subsample size (used to find the right cached zarr)
    data_path : Path, optional
        Path to zarr store or h5ad file. If None, looks for prepared cache.
    host : str
        Server host
    port : int
        Server port
    use_plate : bool
        Whether to serve the phenotyping plate for cell crop viewing
    open_browser : bool
        Whether to automatically open the browser when server starts
    """
    from nd_embedding_atlas.io import AnnDataCollection
    from nd_embedding_atlas.vz import serve

    paths = _experiment_paths(experiment)

    # Find data path: explicit > prepared zarr cache > source h5ad
    if data_path is None:
        source_h5ad, prepared_zarr = get_cache_paths(experiment, subsample)
        if prepared_zarr.exists():
            data_path = prepared_zarr
        elif source_h5ad.exists():
            raise FileNotFoundError(
                f"No prepared zarr found at: {prepared_zarr}\n"
                f"Run 'python -m cyclops_process.napari.nd_embed prepare -e {experiment}' first."
            )
        else:
            raise FileNotFoundError(f"No data found for experiment {experiment}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    # Set up plate path for cell crop viewer
    plate_path = None
    if use_plate:
        candidate = paths["pheno_assembled_v3"]
        if candidate.exists():
            plate_path = candidate
            print(f"Using phenotyping plate for cell crops: {plate_path}")
        else:
            print("Warning: Phenotyping plate not found, cell crops will not be available")

    url = f"http://{host}:{port}"
    print(f"\nLaunching nd-embedding-atlas viewer...")
    print(f"  AnnData: {data_path}")
    print(f"  Plate path: {plate_path}")
    print(f"  Server: {url}")
    print("\nPress Ctrl+C to stop the server.\n")

    collection = AnnDataCollection()
    collection[experiment] = str(data_path)

    obs_columns = [
        "well", "gene_name", "barcode", "sgRNA", "subpool",
        "y_global_pheno", "x_global_pheno", "bbox",
    ]

    # Auto-open browser after a short delay
    if open_browser:
        def _open_browser():
            time.sleep(1.5)
            print(f"Opening browser: {url}")
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()

    serve(
        collection,
        obs_columns=obs_columns,
        plate_path=str(plate_path) if plate_path else None,
        host=host,
        port=port,
    )


def launch_embedding_viewer(
    experiment: str,
    umap_csv_path: Optional[Path] = None,
    subsample: Optional[int] = None,
    host: str = "localhost",
    port: int = 5055,
    use_plate: bool = True,
    overwrite: bool = False,
    open_browser: bool = True,
) -> None:
    """Prepare data (if needed) and launch the viewer."""
    zarr_path = prepare_embedding_data(
        experiment,
        umap_csv_path=umap_csv_path,
        subsample=subsample,
        overwrite=overwrite,
    )

    serve_embedding_viewer(
        experiment,
        subsample=subsample,
        data_path=zarr_path,
        host=host,
        port=port,
        use_plate=use_plate,
        open_browser=open_browser,
    )


def _resolve_experiment_name(user_input: str) -> str:
    """Resolve shorthand like '94' to 'ops0094_20251217' by scanning the base directory."""
    import re
    normalized = user_input.strip().lower()
    digits = re.sub(r"\D", "", normalized)

    if not digits:
        return user_input

    # Scan for matching experiment directories
    if not BASE_DIR.exists():
        return user_input

    matches = []
    for d in sorted(BASE_DIR.iterdir()):
        if d.is_dir() and re.search(rf"^ops0*{digits}(?:_|$)", d.name.lower()):
            matches.append(d.name)

    if len(matches) == 1:
        print(f"[experiment] Resolved '{user_input}' -> {matches[0]}")
        return matches[0]
    elif len(matches) > 1:
        # Prefer canonical ops####_YYYYMMDD format
        canonical = [m for m in matches if re.match(r"^ops\d{4}_\d{8}$", m)]
        if len(canonical) == 1:
            print(f"[experiment] Auto-selected canonical: {canonical[0]}")
            return canonical[0]
        print(f"[experiment] Multiple matches for '{user_input}': {matches}")
        print(f"[experiment] Using first match: {matches[0]}")
        return matches[0]

    # Exact match fallback
    candidate = BASE_DIR / user_input
    if candidate.exists():
        return user_input

    return user_input


def main():
    """Command-line interface for nd-embedding-atlas viewer."""
    parser = argparse.ArgumentParser(
        description="nd-embedding-atlas viewer for OPS experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare data (merge UMAP into h5ad, run once)
  python -m cyclops_process.napari.nd_embed prepare -e ops0094_20251217

  # Launch viewer (reads h5ad directly)
  python -m cyclops_process.napari.nd_embed serve -e ops0094_20251217

  # Prepare + launch in one command
  python -m cyclops_process.napari.nd_embed launch -e ops0094_20251217

  # Preview mode with subsampled data
  python -m cyclops_process.napari.nd_embed launch -e 94 --preview 10000
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    def add_common_args(p):
        p.add_argument(
            "-e", "--experiment",
            type=str,
            required=True,
            help="Experiment name or shorthand (e.g., '94', 'ops94', 'ops0094_20251217')",
        )
        p.add_argument(
            "--preview",
            type=int,
            default=None,
            help="Subsample to N cells for faster loading (e.g., 10000)",
        )

    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare embedding data (merge UMAP CSV + convert to zarr)"
    )
    add_common_args(prepare_parser)
    prepare_parser.add_argument(
        "--umap-csv", type=Path, default=None,
        help="Path to UMAP coordinates CSV (default: auto-detect)",
    )
    prepare_parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for prepared h5ad",
    )
    prepare_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing prepared h5ad",
    )

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch viewer from prepared zarr"
    )
    add_common_args(serve_parser)
    serve_parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to zarr store or h5ad file (default: auto-detect from cache)",
    )
    serve_parser.add_argument("--host", type=str, default="localhost")
    serve_parser.add_argument("--port", type=int, default=5055)
    serve_parser.add_argument(
        "--no-plate", action="store_true",
        help="Disable cell crop viewer (don't load phenotyping plate)",
    )
    serve_parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't automatically open browser",
    )

    # Launch command (prepare + serve)
    launch_parser = subparsers.add_parser(
        "launch",
        help="Prepare data (if needed) and launch viewer"
    )
    add_common_args(launch_parser)
    launch_parser.add_argument(
        "--umap-csv", type=Path, default=None,
        help="Path to UMAP coordinates CSV (default: auto-detect)",
    )
    launch_parser.add_argument("--host", type=str, default="localhost")
    launch_parser.add_argument("--port", type=int, default=5055)
    launch_parser.add_argument(
        "--no-plate", action="store_true",
        help="Disable cell crop viewer (don't load phenotyping plate)",
    )
    launch_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing prepared h5ad",
    )
    launch_parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't automatically open browser",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    experiment = _resolve_experiment_name(args.experiment)
    print(f"Experiment: {experiment}")

    if args.command == "prepare":
        prepare_embedding_data(
            experiment=experiment,
            umap_csv_path=args.umap_csv,
            subsample=args.preview,
            output_path=args.output,
            overwrite=args.overwrite,
        )

    elif args.command == "serve":
        serve_embedding_viewer(
            experiment=experiment,
            subsample=args.preview,
            data_path=args.data,
            host=args.host,
            port=args.port,
            use_plate=not args.no_plate,
            open_browser=not args.no_browser,
        )

    elif args.command == "launch":
        launch_embedding_viewer(
            experiment=experiment,
            umap_csv_path=args.umap_csv,
            subsample=args.preview,
            host=args.host,
            port=args.port,
            use_plate=not args.no_plate,
            overwrite=args.overwrite,
            open_browser=not args.no_browser,
        )


if __name__ == "__main__":
    main()
