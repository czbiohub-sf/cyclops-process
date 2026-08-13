"""
Interactive menu and UI components for PipelineRunner.

This module handles:
- User prompts and confirmations
- Step selection from history and planned steps
- Full step list display with status indicators
- Menu navigation and user input processing
"""

from prettytable import PrettyTable
from cyclops_utils.data.experiment import OpsDataset
from cyclops_utils.io.zarr_utils import has_fluorescence_channels_from_config


class InteractiveMenu:
    """Manages interactive prompts and step selection for the pipeline."""

    def __init__(
        self,
        experiment: str,
        config: dict,
        dataset: OpsDataset,
        completion_checker,
        format_timeout_func,
    ):
        """Initialize the interactive menu.

        Args:
            experiment: Experiment name
            config: Pipeline configuration dict
            dataset: OpsDataset instance
            completion_checker: CompletionChecker instance
            format_timeout_func: Function to format timeout display
        """
        self.experiment = experiment
        self.config = config
        self.dataset = dataset
        self._completion_checker = completion_checker
        self._format_timeout_display = format_timeout_func

        # Selection state
        self._selection_map: dict[int, tuple[str, object]] = {}
        self._next_menu_index: int = 1
        self._target_log_key_to_run_once: str | None = None
        self._abort_all: bool = False

    def show_full_step_list(self) -> bool:
        """Show full list of steps with DONE/TODO status and allow user selection.

        Returns:
            True if a target was selected, False otherwise
        """
        try:
            all_keys = self.dataset.get_all_step_keys()
        except Exception:
            all_keys = []

        if not all_keys:
            print("(No step registry available to list.)")
            return False

        print("\nFull step list:")
        table = PrettyTable()
        table.field_names = ["#", "State", "Step", "Info"]
        table.align = "l"

        has_fluor_cfg = has_fluorescence_channels_from_config(self.config)

        for idx, k in enumerate(all_keys, start=1):
            # Optional display of expected runtime/resources from Slurm config
            timeout_display = self._format_timeout_display(k)
            info = timeout_display.strip()
            if info.startswith("(") and info.endswith(")"):
                info = info[1:-1].strip()
            step_name = f"{idx} - {k}"

            # Not-applicable detection for fluorescence-related steps
            if ("lc_20x_fluor" in k or "fluor" in k) and not has_fluor_cfg:
                table.add_row([idx, "🟦", step_name, info])
                continue

            dot = self._completion_checker.get_step_status(k)

            table.add_row([idx, dot, step_name, info])

        print(table)

        # Handle user selection
        while True:
            sel = (
                input("Enter number to jump (or 'q' to quit, Enter to cancel): ")
                .strip()
                .lower()
            )
            if sel == "":
                return False
            if sel in {"q", "quit"}:
                self._abort_all = True
                return False
            if sel.isdigit():
                i = int(sel)
                if 1 <= i <= len(all_keys):
                    self._target_log_key_to_run_once = str(all_keys[i - 1])
                    print(
                        f"--- Will skip to and run: '{self._target_log_key_to_run_once}' ---"
                    )
                    return True
            print(
                f"Please enter a number between 1 and {len(all_keys)}, or press Enter to cancel."
            )

    def prompt_checkpoint(
        self,
        message: str,
        include_initial_banner: bool = False,
        completed_steps_history: list = None,
        history_index: int = None,
        initial_skip_banner_shown: bool = False,
        rerun_steps: list = None,
    ) -> tuple[str, int]:
        """Unified interactive checkpoint prompt.

        Args:
            message: Prompt message to display
            include_initial_banner: Whether to show skip banner
            completed_steps_history: List of (func, kwargs, log_key) tuples
            history_index: Current index in history
            initial_skip_banner_shown: Whether banner was already shown
            rerun_steps: List of steps to rerun (non-interactive mode)

        Returns:
            Tuple of (action, updated_history_index) where action is one of:
            "proceed", "skip", "back", "quit"
        """
        # In rerun mode we run non-interactively
        if rerun_steps is not None:
            return "proceed", history_index
        if self._abort_all:
            return "quit", history_index

        if include_initial_banner and not initial_skip_banner_shown:
            print("--- Skipping completed steps until the first incomplete step ---")

        print(f"\n{message}")

        # Default selection points to most recent completed step, if any
        if completed_steps_history and history_index is None:
            history_index = len(completed_steps_history) - 1

        while True:
            # Build multi-line options UI
            lines = [
                "Options:",
                "  [y]es  - continue (run the current executable step)",
                "  [q]uit - abort the pipeline run",
                "  [f]ull - show full list of steps for selection",
            ]

            print("\n".join(lines))

            choice_raw = input("Enter choice or number: ").strip().lower()
            if choice_raw in ("y", "yes", ""):
                return "proceed", history_index
            if choice_raw in ("q", "quit", "n", "no"):
                return "quit", history_index
            if choice_raw in ("f", "full"):
                selected = self.show_full_step_list()
                if selected:
                    return "skip", history_index
                # If user chose to quit from full list, honor immediately
                if self._abort_all:
                    return "quit", history_index
                # otherwise, continue the loop
            if choice_raw.isdigit():
                sel = int(choice_raw)
                if sel in self._selection_map:
                    kind, value = self._selection_map[sel]
                    if kind == "history":
                        # Re-run a specific completed step
                        history_index = int(value)
                        return "back", history_index
                    if kind == "upcoming":
                        # Skip ahead until this planned step is reached
                        self._target_log_key_to_run_once = str(value)
                        return "skip", history_index
            print("Invalid input. Please enter y/q or a listed number.")

    def prompt_completed_step(self, log_key: str) -> str:
        """Prompt user for action on a completed step.

        Args:
            log_key: Step identifier

        Returns:
            One of: "skip_to_incomplete", "rerun", "skip_step", "rerun_all", "quit"
        """
        print(f"--- Step '{log_key}' appears complete based on output files. ---")

        while True:
            choice = input(
                "--> Action: [s]kip to next incomplete, [y]es to re-run, [n]o to skip, [a]ll to re-run all, [f]ull list, [q]uit: "
            ).lower()
            if choice == "s":
                return "skip_to_incomplete"
            elif choice == "y":
                return "rerun"
            elif choice == "n":
                return "skip_step"
            elif choice in ["a", "all"]:
                return "rerun_all"
            elif choice in ("f", "full"):
                selected = self.show_full_step_list()
                if selected:
                    return "skip_to_selected"
                if self._abort_all:
                    return "quit"
            elif choice == "q":
                return "quit"
            else:
                print("Invalid input. Please enter 's', 'y', 'n', 'a', 'f', or 'q'.")

    def add_selection(self, kind: str, value: object) -> int:
        """Add a numbered selection option.

        Args:
            kind: Type of selection ("history" or "upcoming")
            value: Associated value (history index or log_key)

        Returns:
            The selection number assigned
        """
        number = self._next_menu_index
        self._selection_map[number] = (kind, value)
        self._next_menu_index += 1
        return number

    def get_target_log_key(self) -> str | None:
        """Get the currently selected target log key."""
        return self._target_log_key_to_run_once

    def set_target_log_key(self, key: str | None) -> None:
        """Set the target log key for one-shot execution."""
        self._target_log_key_to_run_once = str(key) if key else None

    def is_aborted(self) -> bool:
        """Check if user requested abort."""
        return self._abort_all

    def set_aborted(self, value: bool) -> None:
        """Set abort status."""
        self._abort_all = value
