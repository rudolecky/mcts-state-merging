"""CLI: merge-vs-tree under blind UCB1-MCTS program synthesis on real
ARC-AGI-1 tasks (Stage 2 of the ARC plan). Task selection: real tasks whose
`solvers.py` reference solution uses only functions from
`classical_mcts_arc_program`'s curated 60-function subset -- the same
"verify exact reachability via a trusted oracle" principle as every other
domain's puzzle generator in this project, just reusing `solvers.py` as the
oracle instead of a hand-rolled BFS.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from . import classical_mcts_arc_program as m
from .datasets import arc_engine as ae
from .datasets.arc_program_engine import CURATED_FUNCTIONS

_VENDOR_DIR = str(Path(__file__).parent / "vendor" / "arc_dsl")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
import solvers as real_solvers  # noqa: E402

_LINE_RE = re.compile(r"^(\w+) = (\w+)\((.*)\)$")


def _max_grid_dim(task_id: str) -> int:
    raw = json.loads((Path("data/arc_agi/tasks") / f"{task_id}.json").read_text())
    dims = []
    for pair in raw["train"]:
        dims.append(max(len(pair["input"]), len(pair["input"][0])))
        dims.append(max(len(pair["output"]), len(pair["output"][0])))
    return max(dims)


def find_reachable_tasks(min_depth: int, max_depth: int, max_grid_dim: int = 15) -> list:
    """`max_grid_dim` keeps this experiment in the size regime it was
    calibrated at: real ARC grids run up to 30x30, and several curated
    primitives (upscale, objects, mapply over cell-level objects) scale
    badly with grid size -- a 30x30 task, uncapped, drove one calibration
    run to 9.5GB and 97 CPU-minutes before it was killed. A documented
    scope-narrowing choice, not a silent one."""
    curated = set(CURATED_FUNCTIONS)
    found = []
    for name in dir(real_solvers):
        if not name.startswith("solve_"):
            continue
        src = inspect.getsource(getattr(real_solvers, name))
        body = [l.strip() for l in src.splitlines()[1:] if l.strip()][:-1]
        funcs = []
        ok = True
        for line in body:
            match = _LINE_RE.match(line)
            if not match:
                ok = False
                break
            funcs.append(match.group(2))
        if ok and funcs and all(f in curated for f in funcs) and min_depth <= len(body) <= max_depth:
            task_id = name[6:]
            if _max_grid_dim(task_id) <= max_grid_dim:
                found.append((task_id, len(body)))
    return found


def _bootstrap_ci_mean_diff(diffs, seed, n_boot=2000):
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def _solve_rate(task, merge_enabled: bool, budget: int, rollout_depth: int, repeats: int, base_seed: int):
    solved = 0
    node_counts = []
    for r in range(repeats):
        rng = random.Random(base_seed * 100_003 + r)
        config = m.ClassicalMCTSConfig(merge_enabled=merge_enabled)
        graph = m.run_search(task.train_inputs, task.train_outputs, config, budget, rollout_depth, rng)
        if m.is_solved(graph):
            solved += 1
        node_counts.append(len(graph.nodes))
    return solved / repeats, float(np.mean(node_counts))


def main():
    ap = argparse.ArgumentParser(description="Merge-vs-tree experiment: blind MCTS program synthesis on real ARC-AGI-1 tasks")
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-grid-dim", type=int, default=15)
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 2000, 5000])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--rollout-depth", type=int, default=5)
    ap.add_argument("--output-dir", default="results/arc_program_experiment")
    args = ap.parse_args()

    start = time.time()
    reachable = find_reachable_tasks(args.min_depth, args.max_depth, args.max_grid_dim)
    rng = random.Random(args.seed)
    rng.shuffle(reachable)
    selected = reachable[: args.n_tasks]
    print(f"{len(reachable)} real tasks reachable at depth [{args.min_depth},{args.max_depth}]; using {len(selected)}")

    tasks = []
    for task_id, depth in selected:
        raw = json.loads(Path(f"data/arc_agi/tasks/{task_id}.json").read_text())
        task = ae.load_task(raw, task_id)
        tasks.append((task, depth))

    results = {}
    for budget in args.budgets:
        rows = []
        for i, (task, depth) in enumerate(tasks):
            base_rate, base_nodes = _solve_rate(task, False, budget, args.rollout_depth, args.repeats, base_seed=args.seed * 1000 + i)
            treat_rate, treat_nodes = _solve_rate(task, True, budget, args.rollout_depth, args.repeats, base_seed=args.seed * 1000 + i)
            rows.append({
                "task_id": task.task_id, "reference_depth": depth,
                "baseline_rate": base_rate, "treatment_rate": treat_rate,
                "baseline_nodes": base_nodes, "treatment_nodes": treat_nodes,
            })
        diffs = np.array([r["treatment_rate"] - r["baseline_rate"] for r in rows])
        ci_low, ci_high = _bootstrap_ci_mean_diff(diffs, seed=args.seed)
        base_mean = np.mean([r["baseline_rate"] for r in rows])
        treat_mean = np.mean([r["treatment_rate"] for r in rows])
        base_nodes_mean = np.mean([r["baseline_nodes"] for r in rows])
        treat_nodes_mean = np.mean([r["treatment_nodes"] for r in rows])
        try:
            _stat, wilcoxon_p = wilcoxon(diffs)
        except ValueError:
            wilcoxon_p = float("nan")
        print(f"budget={budget}: baseline={base_mean:.3f} treatment={treat_mean:.3f} "
              f"diff_mean={diffs.mean():.3f} CI=[{ci_low:.3f},{ci_high:.3f}] wilcoxon_p={wilcoxon_p:.4f} "
              f"nodes={base_nodes_mean:.1f}->{treat_nodes_mean:.1f}")
        results[budget] = {
            "rows": rows, "baseline_mean": float(base_mean), "treatment_mean": float(treat_mean),
            "diff_mean": float(diffs.mean()), "ci_low": ci_low, "ci_high": ci_high,
            "wilcoxon_p": float(wilcoxon_p),
            "baseline_nodes_mean": float(base_nodes_mean), "treatment_nodes_mean": float(treat_nodes_mean),
        }

    elapsed = time.time() - start
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "elapsed_sec": elapsed, "results": results,
                    "task_ids": [t.task_id for t, _ in tasks]}, f, indent=2)
    print(f"done in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
