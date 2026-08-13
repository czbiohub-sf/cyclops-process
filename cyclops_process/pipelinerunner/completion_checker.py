"""
File-based completion checking for PipelineRunner.

This module handles:
- Checking if pipeline steps are complete based on output files
- Handling method variants and per-well steps
- Content validation for files and directories
"""

from pathlib import Path
from ops_utils.data.experiment import OpsDataset


class CompletionChecker:
    """Handles file-based completion checks for pipeline steps."""

    def __init__(self, experiment: str, config: dict, dataset: OpsDataset):
        """Initialize the completion checker.

        Args:
            experiment: Experiment name
            config: Pipeline configuration dict
            dataset: OpsDataset instance
        """
        self.experiment = experiment
        self.config = config
        self.dataset = dataset

    # Steps that should ALWAYS run, even when their outputs already exist —
    # they intentionally re-do the work (e.g. recompute_metrics calls
    # get_metrics(force=True) at end-of-pipeline so QC reflects all final
    # downstream artifacts).
    _ALWAYS_RUN = {"recompute_metrics"}

    def is_step_complete(
        self, func, kwargs: dict | None = None, methods: list[str] | None = None
    ) -> tuple[bool, list[Path]]:
        """Check if a step is complete based on output files.

        Args:
            func: Pipeline function to check
            kwargs: Keyword arguments for the function (may contain 'method' key)
            methods: Optional list of method variants to check

        Returns:
            Tuple of (is_complete, output_files):
                - is_complete: True if all outputs exist with content
                - output_files: List of expected output file paths
        """
        # Steps in _ALWAYS_RUN bypass the completion check so the orchestrator
        # doesn't prompt the user — returning (False, []) routes through the
        # "no file-based check defined; default to running" branch in the
        # PipelineRunner.
        func_name = getattr(func, "__name__", "")
        if func_name in self._ALWAYS_RUN:
            return False, []

        # If kwargs contains a 'method' key, this is a method-specific step
        # Use get_dataset_for_kwargs to get the appropriate dataset
        if kwargs and "method" in kwargs:
            # This is a single method-specific invocation (e.g., get_metrics with method='mine')
            # Check using the method-specific dataset
            ds = get_dataset_for_kwargs(
                self.experiment, self.config, self.dataset, kwargs
            )
            output_files = self._get_output_files_for_func(func, kwargs, ds)

            if not output_files:
                return False, []

            all_exist = all(self.has_content(p) for p in output_files)
            return all_exist, output_files

        if methods:
            # Check all method variants (for menu display purposes)
            for method in methods:
                method_complete, method_outputs = self._check_method_variant(
                    func, kwargs, method
                )
                if not method_complete:
                    return False, method_outputs
            # All methods complete
            return True, []
        else:
            # Single step without method variants
            return self._check_single_step(func, kwargs)

    def _check_method_variant(
        self, func, kwargs: dict | None, method: str
    ) -> tuple[bool, list[Path]]:
        """Check completion for a specific method variant.

        Args:
            func: Pipeline function
            kwargs: Keyword arguments
            method: Method variant name

        Returns:
            Tuple of (is_complete, output_files)
        """
        ds_method = OpsDataset(self.experiment, self.config, method=str(method))
        output_files = self._get_output_files_for_func(func, kwargs, ds_method)

        if not output_files:
            return False, []

        all_exist = all(self.has_content(p) for p in output_files)
        return all_exist, output_files

    def _check_single_step(
        self, func, kwargs: dict | None = None
    ) -> tuple[bool, list[Path]]:
        """Check completion for a single step without method variants.

        Args:
            func: Pipeline function
            kwargs: Keyword arguments

        Returns:
            Tuple of (is_complete, output_files)
        """
        output_files = self._get_output_files_for_func(func, kwargs, self.dataset)

        if not output_files:
            return False, []

        all_exist = all(self.has_content(p) for p in output_files)
        return all_exist, output_files

    def _get_output_files_for_func(
        self, func, kwargs: dict | None, dataset: OpsDataset
    ) -> list[Path]:
        """Get expected output files for a function.

        Args:
            func: Pipeline function
            kwargs: Keyword arguments
            dataset: OpsDataset instance to query

        Returns:
            List of expected output file paths
        """
        try:
            func_name = func.__name__ if hasattr(func, "__name__") else str(func)
        except Exception:
            func_name = str(func)

        process = (kwargs or {}).get("process")
        well = (kwargs or {}).get("well")

        # Build the key: func_name + optional _process + optional _well
        base_key = func_name
        if process:
            base_key = f"{base_key}_{process}"
        if well:
            base_key = f"{base_key}_{str(well).replace('/', '_')}"

        return dataset.get_output_files_for_step(base_key, self.config)

    # Steps that always run with method="mine" regardless of config
    _METHOD_STEPS = {"base_calling", "get_metrics", "recompute_metrics"}
    _DEFAULT_METHOD = "mine"

    def get_step_status(self, step_key: str) -> str:
        """Get the display status dot for a step.

        Handles method-aware steps by checking outputs using the default
        method ("mine"), since the orchestrator always runs these with
        method="mine" regardless of what the config specifies.

        Returns:
            "🟢" (complete), "🔴" (incomplete), or "⚪" (no outputs defined)
        """
        if step_key in self._METHOD_STEPS:
            ds = OpsDataset(self.experiment, self.config, method=self._DEFAULT_METHOD)
            outputs = ds.get_output_files_for_step(step_key, self.config)
        else:
            outputs = self.dataset.get_output_files_for_step(step_key, self.config)

        if not outputs:
            return "⚪"
        return "🟢" if all(self.has_content(p) for p in outputs) else "🔴"

    @staticmethod
    def has_content(p: Path) -> bool:
        """Check if a path exists and has content.

        Args:
            p: Path to check

        Returns:
            True if path exists and contains data
        """
        try:
            if not p.exists():
                return False
            if p.is_file():
                return p.stat().st_size > 0
            if p.is_dir():
                # Fast Zarr validation: presence of .zgroup or .zarray (v2) or zarr.json (v3)
                if p.suffix == ".zarr" or p.name.endswith(".zarr"):
                    if (p / ".zgroup").exists() or (p / ".zarray").exists():
                        return True
                    # Zarr v3 uses zarr.json instead of .zgroup/.zarray
                    if (p / "zarr.json").exists():
                        return True
                    return False
                return next(p.iterdir(), None) is not None
            return False
        except Exception:
            return False


def get_dataset_for_kwargs(
    experiment: str, config: dict, dataset: OpsDataset, kwargs: dict | None
) -> OpsDataset:
    """Get the appropriate OpsDataset instance based on kwargs.

    If kwargs contains a 'method', return a dataset for that method variant.
    Otherwise, return the default dataset.

    Args:
        experiment: Experiment name
        config: Pipeline configuration dict
        dataset: Default OpsDataset instance
        kwargs: Keyword arguments that may contain 'method'

    Returns:
        OpsDataset instance (method-specific or default)
    """
    if kwargs and "method" in kwargs:
        return OpsDataset(experiment, config, method=str(kwargs.get("method")))
    return dataset
