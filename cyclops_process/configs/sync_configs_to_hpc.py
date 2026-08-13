#!/usr/bin/env python
"""
Sync local config files to the HPC configs directory.

This script compares local config files with their HPC counterparts and
shows a detailed diff before applying any changes.

Config files synced:
  - ops_failed_rounds.yaml
  - ops_channel_maps.yaml
  - slurm_task_config.yaml

Usage:
    python cyclops_process/configs/sync_configs_to_hpc.py           # Preview changes only
    python cyclops_process/configs/sync_configs_to_hpc.py --apply   # Apply changes after preview
    python cyclops_process/configs/sync_configs_to_hpc.py --force   # Apply without confirmation
"""

import sys
import os
from pathlib import Path
import difflib
import shutil
from datetime import datetime
import argparse
from cyclops_process.paths import BASE_PATH

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# HPC configs directory (from OpsDataset)
HPC_CONFIGS_DIR = Path(f"{BASE_PATH}/configs")

# Local configs directory (where this script lives)
LOCAL_CONFIGS_DIR = Path(__file__).resolve().parent

# Files to sync
CONFIG_FILES = [
    "ops_failed_rounds.yaml",
    "ops_channel_maps.yaml",
    "org_seg_params.yaml",
    "slurm_task_config.yaml",
]

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def read_file_lines(path: Path) -> list[str]:
    """Read file and return lines, or empty list if file doesn't exist."""
    if not path.exists():
        return []
    with open(path, "r") as f:
        return f.readlines()


def generate_diff(local_path: Path, hpc_path: Path) -> tuple[list[str], int, int]:
    """
    Generate a unified diff between local and HPC files.

    Returns:
        tuple: (diff_lines, additions, deletions)
    """
    local_lines = read_file_lines(local_path)
    hpc_lines = read_file_lines(hpc_path)

    diff = list(difflib.unified_diff(
        hpc_lines,
        local_lines,
        fromfile=f"HPC: {hpc_path}",
        tofile=f"Local: {local_path}",
        lineterm=""
    ))

    additions = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    deletions = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))

    return diff, additions, deletions


def print_diff(diff_lines: list[str], max_context: int = 500):
    """Print diff with color coding."""
    if not diff_lines:
        print(f"  {GREEN}No changes{RESET}")
        return

    lines_shown = 0
    for line in diff_lines:
        if lines_shown >= max_context:
            remaining = len(diff_lines) - lines_shown
            print(f"  {YELLOW}... and {remaining} more lines{RESET}")
            break

        # Remove trailing newline for display
        line = line.rstrip("\n")

        if line.startswith("+++") or line.startswith("---"):
            print(f"  {BOLD}{line}{RESET}")
        elif line.startswith("@@"):
            print(f"  {CYAN}{line}{RESET}")
        elif line.startswith("+"):
            print(f"  {GREEN}{line}{RESET}")
        elif line.startswith("-"):
            print(f"  {RED}{line}{RESET}")
        else:
            print(f"  {line}")

        lines_shown += 1


def backup_file(path: Path) -> Path | None:
    """Create a backup of a file in configs/backups/configs_backup_YYYYMMDD_HHMMSS/."""
    if not path.exists():
        return None

    # Create backup directory structure: configs/backups/configs_backup_YYYYMMDD_HHMMSS/
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = HPC_CONFIGS_DIR / "backups" / f"configs_backup_{timestamp_str}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Copy file to backup directory (keep original filename)
    backup_path = backup_dir / path.name
    shutil.copy(path, backup_path)
    return backup_path


def sync_file(local_path: Path, hpc_path: Path, create_backup: bool = True) -> bool:
    """
    Sync a local file to the HPC location.

    Returns:
        bool: True if file was synced, False if skipped
    """
    if not local_path.exists():
        print(f"  {RED}Error: Local file does not exist: {local_path}{RESET}")
        return False

    # Create backup if HPC file exists
    if create_backup and hpc_path.exists():
        backup_path = backup_file(hpc_path)
        if backup_path:
            # Show path relative to HPC configs dir
            rel_path = backup_path.relative_to(HPC_CONFIGS_DIR)
            print(f"  {BLUE}Backup created: {rel_path}{RESET}")

    # Remove existing file first so we create a fresh copy owned by us.
    # shutil.copyfile opens the destination in-place ('wb'), which fails when
    # the ACL mask on the existing file restricts group write access even though
    # the directory grants it.  Unlinking + recreating avoids that restriction.
    if hpc_path.exists():
        hpc_path.unlink()
    shutil.copyfile(local_path, hpc_path)
    return True


def sync_all_configs(create_backup: bool = True, prompt: bool = False) -> int:
    """Sync all local config files to HPC.

    Callable from other scripts (e.g., generate_config_files.py).

    Args:
        create_backup: If True, create backups of HPC files before overwriting.
        prompt: If True, show full diff and ask for confirmation before syncing.

    Returns:
        Number of files synced.
    """
    # Use the same logic as main() with --apply (or --force if not prompting)
    return _sync_configs_impl(
        files_to_sync=CONFIG_FILES,
        apply=True,
        force=not prompt,
        no_backup=not create_backup,
    )


def _sync_configs_impl(
    files_to_sync: list[str],
    apply: bool = False,
    force: bool = False,
    no_backup: bool = False,
) -> int:
    """Core implementation for syncing configs.

    Args:
        files_to_sync: List of filenames to sync.
        apply: If True, apply changes (otherwise dry run).
        force: If True, skip confirmation prompt.
        no_backup: If True, skip creating backups.

    Returns:
        Number of files synced (0 if dry run or cancelled).
    """
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}Config Sync: Local -> HPC{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"\n{BLUE}Local configs:{RESET}  {LOCAL_CONFIGS_DIR}")
    print(f"{BLUE}HPC configs:{RESET}    {HPC_CONFIGS_DIR}")
    print()

    # Check HPC directory exists
    if not HPC_CONFIGS_DIR.exists():
        print(f"{RED}Error: HPC configs directory does not exist: {HPC_CONFIGS_DIR}{RESET}")
        print(f"Make sure you're running this from a system with access to HPC storage.")
        return 0

    # Collect all changes
    changes = {}
    total_additions = 0
    total_deletions = 0
    files_with_changes = 0

    for filename in files_to_sync:
        local_path = LOCAL_CONFIGS_DIR / filename
        hpc_path = HPC_CONFIGS_DIR / filename

        print(f"{BOLD}[{filename}]{RESET}")
        print(f"  Local:  {local_path}")
        print(f"  HPC:    {hpc_path}")

        # Check local file exists
        if not local_path.exists():
            print(f"  {RED}Warning: Local file does not exist, skipping{RESET}")
            print()
            continue

        # Check if HPC file exists
        if not hpc_path.exists():
            print(f"  {YELLOW}HPC file does not exist - will be created{RESET}")
            local_lines = read_file_lines(local_path)
            changes[filename] = {
                "local_path": local_path,
                "hpc_path": hpc_path,
                "diff": [f"+{line}" for line in local_lines],
                "additions": len(local_lines),
                "deletions": 0,
                "is_new": True,
            }
            total_additions += len(local_lines)
            files_with_changes += 1
            print(f"  {GREEN}+{len(local_lines)} lines (new file){RESET}")
            print()
            continue

        # Generate diff
        diff, additions, deletions = generate_diff(local_path, hpc_path)

        if not diff:
            print(f"  {GREEN}Files are identical - no changes needed{RESET}")
            print()
            continue

        changes[filename] = {
            "local_path": local_path,
            "hpc_path": hpc_path,
            "diff": diff,
            "additions": additions,
            "deletions": deletions,
            "is_new": False,
        }
        total_additions += additions
        total_deletions += deletions
        files_with_changes += 1

        print(f"  {GREEN}+{additions}{RESET} / {RED}-{deletions}{RESET} lines changed")
        print()
        print(f"  {BOLD}Diff:{RESET}")
        print_diff(diff)
        print()

    # Summary
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}Summary{RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"  Files to sync: {files_with_changes}/{len(files_to_sync)}")
    print(f"  Total changes: {GREEN}+{total_additions}{RESET} / {RED}-{total_deletions}{RESET} lines")
    print()

    if files_with_changes == 0:
        print(f"{GREEN}All config files are already in sync!{RESET}")
        return 0

    # Apply changes if requested
    if apply or force:
        if not force:
            print(f"{YELLOW}The following files will be updated on HPC:{RESET}")
            for filename in changes:
                info = changes[filename]
                status = "CREATE" if info["is_new"] else "UPDATE"
                print(f"  [{status}] {info['hpc_path']}")
            print()

            response = input(f"{BOLD}Proceed with sync? [y/N]: {RESET}").strip().lower()
            if response != "y":
                print(f"{YELLOW}Sync cancelled.{RESET}")
                return 0

        print()
        print(f"{BOLD}Applying changes...{RESET}")

        files_synced = 0
        for filename, info in changes.items():
            print(f"\n  Syncing {filename}...")
            success = sync_file(
                info["local_path"],
                info["hpc_path"],
                create_backup=not no_backup
            )
            if success:
                print(f"  {GREEN}Done{RESET}")
                files_synced += 1
            else:
                print(f"  {RED}Failed{RESET}")

        print()
        print(f"{GREEN}Sync complete!{RESET}\n")
        return files_synced
    else:
        print(f"{YELLOW}This was a dry run. Use --apply to sync changes.{RESET}")
        print(f"Run: python {Path(__file__).name} --apply")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Sync local config files to HPC configs directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview changes only (dry run)
    python sync_configs_to_hpc.py

    # Apply changes after preview and confirmation
    python sync_configs_to_hpc.py --apply

    # Apply changes without confirmation
    python sync_configs_to_hpc.py --force

    # Sync specific file only
    python sync_configs_to_hpc.py --apply --file ops_failed_rounds.yaml
"""
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes after preview (will ask for confirmation)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Apply changes without confirmation"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup files"
    )
    parser.add_argument(
        "--file",
        type=str,
        choices=CONFIG_FILES,
        help="Sync only a specific file"
    )
    args = parser.parse_args()

    # Determine which files to sync
    files_to_sync = [args.file] if args.file else CONFIG_FILES

    _sync_configs_impl(
        files_to_sync=files_to_sync,
        apply=args.apply,
        force=args.force,
        no_backup=args.no_backup,
    )


if __name__ == "__main__":
    main()
