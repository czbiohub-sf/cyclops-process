#!/usr/bin/env bash
# Mirror an OPS experiment directory into a destination folder using symlinks.
#
# Creates a full directory-structure copy where every file is a symlink back
# to the original. New files/directories written into the mirror stay local
# and never touch the source tree.
#
# Uses GNU parallel or xargs -P for fast multi-threaded symlinking.
#
# Usage:
#   mirror_experiment.sh [OPTIONS] <experiment_name> [destination_base]
#
# Options:
#   --steps 0,1       Comma-separated steps to mirror (default: all)
#                     0 = 0-convert, 1 = 1-preprocess, 2 = 2-tracking, 3 = 3-assembly
#   --jobs N          Parallel workers (default: 8)
#   --dry-run         Show what would be mirrored without doing it
#
# Examples:
#   mirror_experiment.sh ops0105_20260106                       # mirror everything
#   mirror_experiment.sh --steps 0,1 ops0105_20260106           # just convert + preprocess
#   mirror_experiment.sh --steps 0,1 --jobs 16 ops0105_20260106 # 16 parallel workers

set -euo pipefail

# --- Parse arguments ---
STEPS_FLAG=""
DRY_RUN=false
JOBS=""
POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps)
            STEPS_FLAG="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

EXPERIMENT="${POSITIONAL[0]:?Usage: mirror_experiment.sh [--steps 0,1,2,3] [--jobs N] <experiment_name> [destination_base]}"

# Auto-detect available CPUs via resource_manager
if [[ -z "$JOBS" ]]; then
    JOBS=$(python -c "from cyclops_utils.hpc.resource_manager import _get_cpu_limit; print(_get_cpu_limit()[0])" 2>/dev/null || nproc 2>/dev/null || echo 8)
fi

SRC_BASE="${OPS_OUTPUT_BASE_DIR:?OPS_OUTPUT_BASE_DIR is not set}"
DST_BASE="${POSITIONAL[1]:-${SRC_BASE}/reruns}"

SRC="${SRC_BASE}/${EXPERIMENT}"
DST="${DST_BASE}/${EXPERIMENT}"

# Map step numbers to directory names
declare -A STEP_DIRS=(
    [0]="0-convert"
    [1]="1-preprocess"
    [2]="2-tracking"
    [3]="3-assembly"
)

# Determine which steps to mirror
MIRROR_DIRS=()
if [[ -n "$STEPS_FLAG" ]]; then
    IFS=',' read -ra STEP_NUMS <<< "$STEPS_FLAG"
    for num in "${STEP_NUMS[@]}"; do
        num=$(echo "$num" | tr -d ' ')
        if [[ -z "${STEP_DIRS[$num]+x}" ]]; then
            echo "ERROR: Unknown step '$num'. Valid steps: 0, 1, 2, 3" >&2
            exit 1
        fi
        dir="${SRC}/${STEP_DIRS[$num]}"
        if [[ -d "$dir" ]]; then
            MIRROR_DIRS+=("${STEP_DIRS[$num]}")
        else
            echo "WARNING: Step $num dir not found, skipping: $dir"
        fi
    done
else
    for num in 0 1 2 3; do
        dir="${SRC}/${STEP_DIRS[$num]}"
        if [[ -d "$dir" ]]; then
            MIRROR_DIRS+=("${STEP_DIRS[$num]}")
        fi
    done
fi

if [[ ${#MIRROR_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: No step directories found to mirror in $SRC" >&2
    exit 1
fi

echo "Mirroring experiment:"
echo "  SRC: $SRC"
echo "  DST: $DST"
echo "  Steps: ${MIRROR_DIRS[*]}"
echo "  Workers: $JOBS"
echo ""

if $DRY_RUN; then
    echo "[DRY RUN] Would mirror:"
    for d in "${MIRROR_DIRS[@]}"; do
        echo "  $d/"
    done
    exit 0
fi

if [[ -d "$DST" ]]; then
    echo "WARNING: Destination already exists: $DST"
    read -rp "Overwrite mirrored steps? [y/N] " confirm
    if [[ "$confirm" != [yY] ]]; then
        echo "Aborted."
        exit 0
    fi
    for d in "${MIRROR_DIRS[@]}"; do
        rm -rf "$DST/$d"
    done
fi

mkdir -p "$DST"

# --- Symlink top-level files ---
for f in "$SRC"/*; do
    [[ -f "$f" ]] || continue
    ln -sf "$f" "$DST/$(basename "$f")"
done

# --- Phase 1: Create directory structure (fast, single-threaded) ---
start_time=$(date +%s)
echo -n "Creating directories... "
for d in "${MIRROR_DIRS[@]}"; do
    (cd "$SRC" && find "$d" -type d) | while IFS= read -r dir; do
        mkdir -p "$DST/$dir"
    done
done
dir_time=$(( $(date +%s) - start_time ))
echo "done (${dir_time}s)"

# --- Phase 2: Symlink files in parallel ---
echo "Symlinking files ($JOBS workers)..."

# Write all file paths to a temp file, then fan out with xargs
tmpfile=$(mktemp)
trap 'rm -f "$tmpfile"' EXIT

for d in "${MIRROR_DIRS[@]}"; do
    (cd "$SRC" && find "$d" -type f) >> "$tmpfile"
done

total_files=$(wc -l < "$tmpfile")
echo "  $total_files files to symlink"

# Progress tracker: counter file + current zarr file
counter_file=$(mktemp)
current_zarr_file=$(mktemp)
echo "0" > "$counter_file"
echo "" > "$current_zarr_file"
trap 'rm -f "$tmpfile" "$counter_file" "$current_zarr_file"' EXIT

(
    while [[ -f "$counter_file" ]]; do
        count=$(cat "$counter_file" 2>/dev/null || echo 0)
        zarr=$(cat "$current_zarr_file" 2>/dev/null || echo "")
        elapsed=$(( $(date +%s) - start_time ))
        if [[ $count -gt 0 ]]; then
            rate=$(( count / (elapsed > 0 ? elapsed : 1) ))
            remaining=$(( (total_files - count) / (rate > 0 ? rate : 1) ))
            # Truncate zarr name for display
            display_zarr="$zarr"
            if [[ ${#display_zarr} -gt 60 ]]; then
                display_zarr="...${display_zarr: -57}"
            fi
            printf "\r  %d/%d (%d/s, ~%ds left) %s                    " \
                "$count" "$total_files" "$rate" "$remaining" "$display_zarr"
        fi
        sleep 1
    done
) &
progress_pid=$!

# Fan out symlinking across $JOBS workers
# Each worker processes a batch and updates both counter and current zarr
xargs -a "$tmpfile" -P "$JOBS" -n 500 bash -c '
    src="$1"; dst="$2"; counter="$3"; zarr_file="$4"; shift 4
    count=0
    last_zarr=""
    for file in "$@"; do
        ln -s "$src/$file" "$dst/$file"
        count=$((count + 1))
        # Extract .zarr parent path (everything up to and including .zarr)
        case "$file" in
            *.zarr/*)
                zarr="${file%%.zarr/*}.zarr"
                if [[ "$zarr" != "$last_zarr" ]]; then
                    echo "$zarr" > "$zarr_file"
                    last_zarr="$zarr"
                fi
                ;;
        esac
    done
    flock "$counter" bash -c "echo \$(( \$(cat \"$counter\") + $count )) > \"$counter\""
' _ "$SRC" "$DST" "$counter_file" "$current_zarr_file"

# Stop progress display
rm -f "$counter_file" "$current_zarr_file"
wait "$progress_pid" 2>/dev/null || true

end_time=$(date +%s)
elapsed=$((end_time - start_time))

printf "\r  %d/%d files (%ds)                                                              \n" \
    "$total_files" "$total_files" "$elapsed"

# --- Create empty dirs for non-mirrored steps ---
non_mirrored=()
for num in 0 1 2 3; do
    d="${STEP_DIRS[$num]}"
    if [[ ! -d "$DST/$d" ]]; then
        mkdir -p "$DST/$d"
        non_mirrored+=("$d")
    fi
done

# --- Summary ---
echo ""
echo "Done in ${elapsed}s."
echo "  $total_files files symlinked"
if [[ ${#non_mirrored[@]} -gt 0 ]]; then
    echo "  Empty dirs created for: ${non_mirrored[*]}"
fi
echo ""
echo "To run downstream steps on this mirror, set:"
echo "  export OPS_OUTPUT_BASE_DIR=${DST_BASE}"
echo "  export OPS_FAST_OUTPUT_BASE_DIR=${DST_BASE}"
echo ""
echo "Original data at ${SRC} is untouched."
