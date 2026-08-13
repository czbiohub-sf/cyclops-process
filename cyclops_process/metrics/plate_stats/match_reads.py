"""
ISS Read Matching and Round Handling Utilities.

This module provides functions for:
1. Matching ISS reads to codebook entries
2. Handling failed rounds (dropout and shift modes)

Failure Modes:
    - dropout: Round failed completely. Skip this position in both the read barcode
               and the codebook when matching. Both lose that position.
    - shift: Round failed but subsequent rounds shifted down in the image data.
             The read barcode positions after the shift round map to codebook positions
             that are offset by the number of shift rounds before them.
             Example: shift round 3 means read position 3 contains data from physical
             round 4, which should match codebook position 4.
"""

import pandas as pd


def _resolve_well_spec(
    well: str,
    failed_rounds_by_well: dict[str, list[int] | dict] | None = None,
):
    """Return the failed-rounds spec for a well, accepting "A/1/0" or "A1" keys."""
    if failed_rounds_by_well is None:
        return None

    # Try exact match first, then try alternate well key formats.
    # Supports both "A/1/0" (canonical pos) and "A1" (short) formats.
    if well in failed_rounds_by_well:
        return failed_rounds_by_well[well]
    # Build alternate key: "A/1/0" -> "A1", or "A1" -> "A/1/0"
    if "/" in well:
        # "A/1/0" -> "A1"
        parts = well.split("/")
        alt = parts[0] + parts[1]
    else:
        # "A1" -> "A/1/0"  (letter + digit(s) -> letter/digits/0)
        import re
        m = re.match(r"^([A-Za-z]+)(\d+)$", well)
        alt = f"{m.group(1)}/{m.group(2)}/0" if m else None
    if alt and alt in failed_rounds_by_well:
        return failed_rounds_by_well[alt]
    return None


def _get_read_offset(
    well: str,
    failed_rounds_by_well: dict[str, list[int] | dict] | None = None,
) -> int:
    """Return the leading read-frame offset for a well (0 if none).

    Offset N means the read barcode's first N rounds are junk (e.g. a leading
    no-incorporation cycle compensated by an extra cycle at the end), so read
    position ``c + N`` corresponds to codebook position ``c``. Only valid in the
    dict spec: ``{"offset": 1}``.
    """
    spec = _resolve_well_spec(well, failed_rounds_by_well)
    if isinstance(spec, dict):
        return int(spec.get("offset", 0) or 0)
    return 0


def _get_read_offset_mapping(
    iss_rounds: list[int],
    well: str,
    failed_rounds_by_well: dict[str, list[int] | dict] | None,
    code_len: int,
) -> tuple[list[int], list[int]]:
    """Map read positions to codebook positions for a leading read offset.

    read position ``r`` (a value in iss_rounds, after dropouts) maps to codebook
    position ``r - offset``. Positions before the offset (leading junk rounds)
    and positions beyond the codebook length are skipped.
    """
    dropout_rounds, _ = _parse_failed_rounds_spec(well, failed_rounds_by_well)
    offset = _get_read_offset(well, failed_rounds_by_well)
    effective_rounds = [r for r in iss_rounds if r not in dropout_rounds]

    read_positions = []
    codebook_positions = []
    for read_pos in effective_rounds:
        cb = read_pos - offset
        if cb < 0 or cb >= code_len:
            continue
        read_positions.append(read_pos)
        codebook_positions.append(cb)
    return read_positions, codebook_positions


def has_read_offset(
    well: str,
    failed_rounds_by_well: dict[str, list[int] | dict] | None = None,
) -> bool:
    """Check if a well has a leading read-frame offset configured."""
    return _get_read_offset(well, failed_rounds_by_well) > 0


def _parse_failed_rounds_spec(
    well: str,
    failed_rounds_by_well: dict[str, list[int] | dict] | None = None,
    debug: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Parse the failed_rounds_by_well specification for a given well.

    Returns:
        Tuple of (dropout_rounds, shift_rounds)
    """
    failed_spec = _resolve_well_spec(well, failed_rounds_by_well)
    if failed_spec is None:
        return [], []

    if debug:
        print(f"DEBUG _parse_failed_rounds_spec: well={well}, failed_spec={failed_spec}")

    # Backward compatibility: simple list means all dropouts
    if isinstance(failed_spec, list):
        if not failed_spec:
            return [], []
        # Check if it's a list of integers (old format) or list of dicts (new format)
        if isinstance(failed_spec[0], int):
            # Old format: simple list of rounds to drop
            if debug:
                print(f"DEBUG: Parsed as DROPOUT mode (simple list): {failed_spec}")
            return failed_spec, []
        elif isinstance(failed_spec[0], dict):
            # New format: list of {round, mode} dicts
            dropout_rounds = [item["round"] for item in failed_spec if item.get("mode") == "dropout"]
            shift_rounds = sorted([item["round"] for item in failed_spec if item.get("mode") == "shift"])
            if debug:
                print(f"DEBUG: Parsed as list of dicts - dropouts: {dropout_rounds}, shifts: {shift_rounds}")
            return dropout_rounds, shift_rounds
        else:
            return [], []

    # Dictionary format: {"dropout": [...], "shift": X or [...]}
    elif isinstance(failed_spec, dict):
        dropout_rounds = failed_spec.get("dropout", [])
        shift_spec = failed_spec.get("shift")

        # Convert single shift value to list
        if isinstance(shift_spec, int):
            shift_rounds = [shift_spec]
        elif isinstance(shift_spec, list):
            shift_rounds = shift_spec
        else:
            shift_rounds = []

        if debug:
            print(f"DEBUG: Parsed as dict format - dropouts: {dropout_rounds}, shifts: {shift_rounds}")

        return dropout_rounds, shift_rounds

    return [], []


def _get_effective_iss_rounds(
    iss_rounds: list[int],
    well: str,
    failed_rounds_by_well: dict[str, list[int] | dict] | None = None,
) -> list[int]:
    """
    Get the CODEBOOK positions to use for matching (dropout mode only).

    For dropout mode: Skip that logical position in both read and codebook.

    Note: This function does NOT handle shift mode. For shift mode, use
    _get_shift_round_mapping() to get the read-to-codebook position mapping.

    Args:
        iss_rounds: Requested logical rounds
        well: Well identifier
        failed_rounds_by_well: Failed rounds specification

    Returns:
        List of codebook positions to use for matching (with dropouts removed)

    Example:
        iss_rounds = [0,1,2,3,4,5,6,7,8,9]
        failed = {"A/1/0": [3]}  # dropout round 3
        result = [0,1,2,4,5,6,7,8,9]  # Skip position 3
    """
    dropout_rounds, shift_rounds = _parse_failed_rounds_spec(well, failed_rounds_by_well)

    if not dropout_rounds and not shift_rounds:
        return iss_rounds

    # Only filter out dropout rounds - shift rounds are handled separately
    return [r for r in iss_rounds if r not in dropout_rounds]


def _get_shift_round_mapping(
    iss_rounds: list[int],
    well: str,
    failed_rounds_by_well: dict[str, list[int] | dict] | None = None,
    debug: bool = False,
) -> tuple[list[int], list[int]]:
    """
    Get the read positions and corresponding codebook positions for shift mode.

    For shift mode, the read barcode has fewer usable positions because shifted
    rounds cause subsequent positions to map to later codebook positions.

    Args:
        iss_rounds: Requested logical rounds (typically [0,1,2,3,4,5,6,7,8,9])
        well: Well identifier
        failed_rounds_by_well: Failed rounds specification

    Returns:
        Tuple of (read_positions, codebook_positions) where:
        - read_positions: Which positions to extract from the read barcode
        - codebook_positions: Which positions to extract from the codebook
        These lists are parallel - read_positions[i] maps to codebook_positions[i]

    Example:
        iss_rounds = [0,1,2,3,4,5,6,7,8,9]
        failed = {"A/1/0": {"shift": [3]}}  # shift at round 3

        Returns:
            read_positions =     [0, 1, 2, 3, 4, 5, 6, 7, 8]
            codebook_positions = [0, 1, 2, 4, 5, 6, 7, 8, 9]

        Meaning: read[3] should match codebook[4], read[4] matches codebook[5], etc.
        Read position 9 is not used (would need codebook position 10 which doesn't exist).

    Example with multiple shifts:
        failed = {"A/1/0": {"shift": [2, 5]}}  # shifts at rounds 2 and 5

        Returns:
            read_positions =     [0, 1, 2, 3, 4, 5, 6, 7]
            codebook_positions = [0, 1, 3, 4, 6, 7, 8, 9]

        Explanation:
        - Positions 0,1 are before any shift, map directly
        - Position 2 is a shift round, so read[2] maps to codebook[3]
        - Position 3,4 have 1 shift before them, so read[3]->codebook[4], read[4]->codebook[5]
        - Position 5 is another shift, so read[5] maps to codebook[7] (skipping 5 and 6)
        - Positions 6,7 have 2 shifts before them
        - Positions 8,9 would need codebook positions 10,11 which don't exist
    """
    dropout_rounds, shift_rounds = _parse_failed_rounds_spec(well, failed_rounds_by_well, debug=debug)

    # First apply dropout filtering
    effective_rounds = [r for r in iss_rounds if r not in dropout_rounds]

    if not shift_rounds:
        # No shifts - read positions and codebook positions are the same
        return effective_rounds, effective_rounds

    # Build the mapping with shifts
    read_positions = []
    codebook_positions = []

    max_codebook_pos = max(iss_rounds) if iss_rounds else 9
    sorted_shifts = sorted(shift_rounds)

    for read_pos in range(len(effective_rounds)):
        # The codebook position is the read position plus the number of shifts
        # that occurred at or before this read position
        base_codebook_pos = effective_rounds[read_pos]

        # Count how many shift rounds are <= this codebook position
        num_shifts_before = sum(1 for s in sorted_shifts if s <= base_codebook_pos)

        # The actual codebook position we need
        actual_codebook_pos = base_codebook_pos + num_shifts_before

        # Skip shift positions in the codebook
        while actual_codebook_pos in sorted_shifts:
            actual_codebook_pos += 1
            num_shifts_before += 1

        if actual_codebook_pos > max_codebook_pos:
            # This read position would need a codebook position that doesn't exist
            break

        read_positions.append(read_pos)
        codebook_positions.append(actual_codebook_pos)

    if debug:
        print(f"DEBUG _get_shift_round_mapping: well={well}")
        print(f"  shift_rounds={shift_rounds}, dropout_rounds={dropout_rounds}")
        print(f"  read_positions={read_positions}")
        print(f"  codebook_positions={codebook_positions}")

    return read_positions, codebook_positions


def has_shift_rounds(
    well: str,
    failed_rounds_by_well: dict[str, list[int] | dict] | None = None,
) -> bool:
    """Check if a well has any shift rounds configured."""
    _, shift_rounds = _parse_failed_rounds_spec(well, failed_rounds_by_well)
    return len(shift_rounds) > 0


def get_read_codebook_positions(
    iss_rounds: list[int],
    well: str | None,
    failed_rounds_by_well: dict[str, list[int] | dict] | None,
    code_len: int,
) -> tuple[list[int], list[int]]:
    """Return parallel (read_positions, codebook_positions) for a well.

    Single source of truth for how read-barcode positions map to codebook
    positions, covering all modes:
      - none / dropout : read_positions == codebook_positions (effective rounds)
      - shift          : see ``_get_shift_round_mapping``
      - offset (N)     : read position c+N maps to codebook position c

    ``code_len`` bounds the codebook positions (read barcode may be longer, e.g.
    an extra compensating round). Use this instead of a single effective-rounds
    list wherever reads are matched to a codebook/gene-index barcode.
    """
    if well is not None and has_shift_rounds(well, failed_rounds_by_well):
        return _get_shift_round_mapping(iss_rounds, well, failed_rounds_by_well)
    if well is not None and _get_read_offset(well, failed_rounds_by_well) > 0:
        return _get_read_offset_mapping(iss_rounds, well, failed_rounds_by_well, code_len)
    eff = _get_effective_iss_rounds(iss_rounds, well, failed_rounds_by_well)
    eff = [r for r in eff if r < code_len]
    return eff, eff


def match_reads(
    read_db: pd.DataFrame,
    code_db: pd.DataFrame,
    iss_rounds: list[int] | None = None,
    debug: bool = False,
    well_name: str = None,
    failed_rounds_by_well: dict[str, list[int] | dict] | None = None,
) -> pd.DataFrame:
    """
    Matches reads found through ISS to gRNAs provided in the codebook.

    Supports three failure modes:
    - dropout: Skip that position in both read and codebook
    - shift: Read positions after the shift map to later codebook positions
    - offset: A leading read-frame offset — the first N read rounds are junk
      (e.g. a no-incorporation cycle compensated by an extra cycle at the end),
      so read position c+N matches codebook position c. Spec: {"offset": 1}

    Args:
        read_db: DataFrame containing reads with 'barcode' column.
        code_db: DataFrame containing codebook with 'sgRNA' column.
        iss_rounds: List of ISS round indices to use. If None, defaults to first 10 rounds.
        debug: If True, print detailed matching information.
        well_name: Name of the well being processed (for debug output).
        failed_rounds_by_well: Dict mapping wells to failed round specs. Supports:
            - Simple list [3, 9]: dropout rounds 3 and 9
            - Dict {"dropout": [9], "shift": [3]}: dropout 9, shift 3
            - Dict {"offset": 1}: leading read-frame offset of 1

    Returns:
        DataFrame of matched reads (subset of read_db where barcode matched codebook).
    """
    if iss_rounds is None:
        iss_rounds = list(range(10))

    # Codebook length bounds which positions can be matched (the read barcode may
    # be longer than the codebook, e.g. an extra compensating round at the end).
    code_len = len(code_db["sgRNA"].iloc[0]) if len(code_db) > 0 else (max(iss_rounds) + 1)

    use_shift_mode = well_name and has_shift_rounds(well_name, failed_rounds_by_well)
    use_offset_mode = well_name and has_read_offset(well_name, failed_rounds_by_well)

    if use_shift_mode or use_offset_mode:
        # Both modes resolve to a parallel (read_positions, codebook_positions)
        # mapping; only the way it's derived differs.
        if use_shift_mode:
            mode_label = "SHIFT MODE"
            read_positions, codebook_positions = _get_shift_round_mapping(
                iss_rounds, well_name, failed_rounds_by_well, debug=debug
            )
        else:
            mode_label = "OFFSET MODE"
            read_positions, codebook_positions = _get_read_offset_mapping(
                iss_rounds, well_name, failed_rounds_by_well, code_len
            )

        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUG: match_reads() called for well: {well_name} [{mode_label}]")
            print(f"DEBUG: read_positions: {read_positions}")
            print(f"DEBUG: codebook_positions: {codebook_positions}")
            print(f"DEBUG: Total reads to match: {len(read_db)}")

            if len(code_db) > 0:
                example_sgRNA = code_db["sgRNA"].iloc[0]
                example_filtered = "".join([example_sgRNA[i] for i in codebook_positions if i < len(example_sgRNA)])
                print(f"DEBUG: Example codebook sgRNA: {example_sgRNA}")
                print(f"DEBUG: After filtering to codebook positions {codebook_positions}: {example_filtered}")

            if len(read_db) > 0:
                example_barcode = read_db["barcode"].iloc[0]
                example_filtered = "".join([example_barcode[i] for i in read_positions if i < len(example_barcode)])
                print(f"DEBUG: Example read barcode: {example_barcode}")
                print(f"DEBUG: After filtering to read positions {read_positions}: {example_filtered}")

        # Filter codebook using codebook_positions
        codes = ["".join([a[i] for i in codebook_positions if i < len(a)]) for a in code_db["sgRNA"]]

        # Filter reads using read_positions
        read_db["matched"] = read_db["barcode"].apply(
            lambda barcode: "".join([barcode[i] for i in read_positions if i < len(barcode)])
        ).isin(codes)

    else:
        # Standard mode (dropout only, or no failed rounds)
        # Apply dropout filtering to get effective rounds
        effective_rounds = _get_effective_iss_rounds(iss_rounds, well_name, failed_rounds_by_well)
        # Never index past the codebook (e.g. an extra read round with no codebook
        # position) — keeps read and codebook strings the same length.
        effective_rounds = [r for r in effective_rounds if r < code_len]

        if debug:
            print(f"\n{'='*60}")
            print(f"DEBUG: match_reads() called for well: {well_name}")
            print(f"DEBUG: Original iss_rounds: {iss_rounds}")
            print(f"DEBUG: Effective rounds after dropout: {effective_rounds}")
            print(f"DEBUG: Total reads to match: {len(read_db)}")

            # Show example of how codebook is being filtered
            if len(code_db) > 0:
                example_sgRNA = code_db["sgRNA"].iloc[0]
                example_filtered = "".join([example_sgRNA[i] for i in effective_rounds if i < len(example_sgRNA)])
                print(f"DEBUG: Example codebook sgRNA: {example_sgRNA}")
                print(f"DEBUG: After filtering to rounds {effective_rounds}: {example_filtered}")

            # Show example reads
            if len(read_db) > 0:
                example_barcode = read_db["barcode"].iloc[0]
                example_filtered = "".join([example_barcode[i] for i in effective_rounds if i < len(example_barcode)])
                print(f"DEBUG: Example read barcode: {example_barcode}")
                print(f"DEBUG: After filtering to rounds {effective_rounds}: {example_filtered}")

        # Filter both codebook and reads using the effective rounds (with dropouts removed)
        codes = ["".join([a[i] for i in effective_rounds if i < len(a)]) for a in code_db["sgRNA"]]
        read_db["matched"] = read_db["barcode"].apply(
            lambda barcode: "".join([barcode[i] for i in effective_rounds if i < len(barcode)])
        ).isin(codes)

    matched_reads = read_db[read_db["matched"] == True]

    if debug:
        match_rate = len(matched_reads) / len(read_db) * 100 if len(read_db) > 0 else 0
        print(f"DEBUG: Matched {len(matched_reads)}/{len(read_db)} reads ({match_rate:.1f}%)")
        print(f"DEBUG: Unique codebook entries after filtering: {len(set(codes))}")
        print(f"{'='*60}\n")

    return matched_reads
