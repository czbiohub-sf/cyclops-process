"""
Pipeline DAG visualization: parse slurm_task_config.yaml and display the
dependency tree with topological levels (steps at the same level can run
in parallel).

Usage:
    python -m cyclops_process.pipelinerunner.visualize_dag [--mermaid] [--output FILE] [--tree]
"""

import argparse
import sys
import os
from collections import defaultdict, deque
from pathlib import Path

import yaml

sys.path.insert(0, os.getcwd())

# ── ANSI color helpers (disabled when piped) ────────────────────────────
_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def _dim(t: str) -> str:   return _c("2", t)
def _bold(t: str) -> str:  return _c("1", t)
def _green(t: str) -> str: return _c("32", t)
def _cyan(t: str) -> str:  return _c("36", t)
def _yellow(t: str) -> str: return _c("33", t)


def parse_dag(yaml_path: Path) -> dict[str, list[str]]:
    """Parse slurm_task_config.yaml into {step: [dependencies]}."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f) or {}

    dag: dict[str, list[str]] = {}
    for step_name, step_config in config.items():
        if not isinstance(step_config, dict):
            continue
        deps = step_config.get("dependencies")
        if deps is None or deps == "None":
            dag[step_name] = []
        elif isinstance(deps, str):
            dag[step_name] = [deps]
        elif isinstance(deps, list):
            dag[step_name] = [d for d in deps if d is not None and d != "None"]
        else:
            dag[step_name] = []
    return dag


def compute_levels(dag: dict[str, list[str]]) -> dict[int, list[str]]:
    """Topological sort with level assignment (Kahn's algorithm).

    Level 0 = no dependencies. A step's level = max(level of deps) + 1.
    Steps at the same level are independent and can run in parallel.
    """
    in_degree: dict[str, int] = {s: 0 for s in dag}
    children: dict[str, list[str]] = defaultdict(list)
    for step, deps in dag.items():
        for dep in deps:
            if dep not in in_degree:
                in_degree[dep] = 0
                dag[dep] = []
            children[dep].append(step)
        in_degree[step] = len(deps)

    level_of: dict[str, int] = {}
    queue = deque()
    for step, deg in in_degree.items():
        if deg == 0:
            queue.append(step)
            level_of[step] = 0

    while queue:
        current = queue.popleft()
        for child in children[current]:
            in_degree[child] -= 1
            candidate_level = level_of[current] + 1
            level_of[child] = max(level_of.get(child, 0), candidate_level)
            if in_degree[child] == 0:
                queue.append(child)

    levels: dict[int, list[str]] = defaultdict(list)
    for step, lvl in sorted(level_of.items(), key=lambda x: (x[1], x[0])):
        levels[lvl].append(step)
    return dict(levels)


# ── Renderers ───────────────────────────────────────────────────────────

def render_text(dag: dict[str, list[str]], levels: dict[int, list[str]]) -> str:
    """Pretty-print table of levels with parallel indicators."""
    lines = []
    lines.append("=" * 70)
    lines.append("Pipeline Dependency DAG — Topological Levels")
    lines.append("Steps at the same level are independent and CAN run in parallel.")
    lines.append("=" * 70)
    lines.append("")

    max_level = max(levels.keys()) if levels else 0
    for lvl in range(max_level + 1):
        steps = levels.get(lvl, [])
        parallel_tag = " (parallel)" if len(steps) > 1 else ""
        lines.append(f"Level {lvl}{parallel_tag}:")
        for step in steps:
            deps = dag.get(step, [])
            dep_str = f"  <- {', '.join(deps)}" if deps else ""
            lines.append(f"  {step}{dep_str}")
        lines.append("")

    total_steps = sum(len(s) for s in levels.values())
    parallel_levels = sum(1 for s in levels.values() if len(s) > 1)
    lines.append(f"Total steps: {total_steps}")
    lines.append(f"Total levels: {max_level + 1}")
    lines.append(f"Levels with parallelism: {parallel_levels}")
    return "\n".join(lines)


def _classify_step(name: str) -> str:
    """Classify a step into a pipeline phase for color-coding."""
    iss_keywords = ("iss", "convert_iss", "stack_symlinks", "correct_cycle_drift",
                    "detect_spots", "base_calling", "get_metrics", "generate_snr",
                    "register_iss_cycles")
    track_keywords = ("track", "correct_distortion")
    pheno_keywords = ("pheno", "virtual_staining", "create_max_projection",
                      "prepare_unified", "viscy_normalize")
    assembly_keywords = ("build_pyramids", "submit_registration", "submit_tracking",
                         "link_calls", "run_v3", "submit_cell_seg", "build_iss",
                         "submit_organelle", "build_organelle", "extract_features")

    for kw in assembly_keywords:
        if kw in name:
            return "assembly"
    for kw in iss_keywords:
        if kw in name:
            return "iss"
    for kw in track_keywords:
        if kw in name:
            return "track"
    for kw in pheno_keywords:
        if kw in name:
            return "pheno"
    return "other"


def _phase_color(name: str) -> str:
    """Return colored step name based on pipeline phase."""
    phase = _classify_step(name)
    if phase == "iss":
        return _c("34", name)       # blue
    elif phase == "track":
        return _c("35", name)       # magenta
    elif phase == "pheno":
        return _c("32", name)       # green
    elif phase == "assembly":
        return _c("33", name)       # yellow
    return name


def render_tree(dag: dict[str, list[str]], levels: dict[int, list[str]]) -> str:
    """Render a top-down DAG waterfall with box-drawing connectors.

    Shows levels as horizontal bands with connections to parent steps.
    Parallel groups are visually bracketed. Steps are color-coded by phase.
    """
    lines: list[str] = []

    lines.append("")
    lines.append(_bold("  Pipeline Dependency DAG"))
    lines.append(_dim("  ─" * 35))
    lines.append(f"  {_c('34', '■')} ISS   {_c('35', '■')} Tracking   "
                 f"{_c('32', '■')} Phenotyping   {_c('33', '■')} Assembly")
    lines.append(f"  {_dim('│ = sequential')}    {_dim('├┤ = parallel group')}")
    lines.append("")

    max_level = max(levels.keys()) if levels else 0

    for lvl in range(max_level + 1):
        steps = levels.get(lvl, [])
        is_parallel = len(steps) > 1

        # Level header
        if is_parallel:
            n = len(steps)
            lines.append(f"  {_dim(f'L{lvl:>2}')} ┬{'─' * 3} {_yellow(f'⟦ parallel ×{n} ⟧')}")
            for si, step in enumerate(steps):
                deps = dag.get(step, [])
                dep_str = ""
                if deps:
                    dep_str = _dim(f"  ← {', '.join(deps)}")
                is_last = (si == n - 1)
                branch = "└" if is_last else "├"
                lines.append(f"       {branch}── {_phase_color(step)}{dep_str}")
        else:
            step = steps[0]
            deps = dag.get(step, [])
            dep_str = ""
            if deps:
                dep_str = _dim(f"  ← {', '.join(deps)}")
            lines.append(f"  {_dim(f'L{lvl:>2}')} ── {_phase_color(step)}{dep_str}")

        # Vertical connector to next level
        if lvl < max_level:
            lines.append(f"       {_dim('│')}")

    # Summary
    lines.append("")
    lines.append(_dim("  ─" * 35))
    total_steps = sum(len(s) for s in levels.values())
    parallel_levels = sum(1 for s in levels.values() if len(s) > 1)
    lines.append(f"  Steps: {_bold(str(total_steps))}   "
                 f"Levels: {_bold(str(max_level + 1))}   "
                 f"Parallel opportunities: {_bold(str(parallel_levels))}")
    lines.append("")

    return "\n".join(lines)


def render_mermaid(dag: dict[str, list[str]], levels: dict[int, list[str]]) -> str:
    """Generate a Mermaid graph TD diagram with level-colored subgraphs."""
    lines = ["```mermaid", "graph TD"]

    for step, deps in sorted(dag.items()):
        if not deps:
            lines.append(f"    {step}")
        for dep in deps:
            lines.append(f"    {dep} --> {step}")

    lines.append("")
    max_level = max(levels.keys()) if levels else 0
    for lvl in range(max_level + 1):
        steps = levels.get(lvl, [])
        if len(steps) > 1:
            lines.append(f"    subgraph Level_{lvl}[\"Level {lvl} - parallel\"]")
            for step in steps:
                lines.append(f"        {step}")
            lines.append("    end")

    lines.append("```")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize pipeline dependency DAG")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to slurm_task_config.yaml (default: auto-detect)",
    )
    parser.add_argument("--mermaid", action="store_true", help="Output Mermaid diagram")
    parser.add_argument("--levels", action="store_true", help="Output flat level table (legacy)")
    parser.add_argument("--output", type=str, default=None, help="Write output to file")
    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
    else:
        from cyclops_utils.data.experiment import OpsDataset
        dummy = OpsDataset("dummy")
        config_path = dummy.config_paths["slurm_task_config"]

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    dag = parse_dag(config_path)
    levels = compute_levels(dag)

    output_parts = []

    if args.levels:
        output_parts.append(render_text(dag, levels))
    elif args.mermaid:
        output_parts.append(render_mermaid(dag, levels))
    else:
        # Default: nice tree view
        output_parts.append(render_tree(dag, levels))

    output = "\n".join(output_parts)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
