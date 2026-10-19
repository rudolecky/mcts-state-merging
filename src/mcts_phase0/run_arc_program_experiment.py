"""CLI: tree vs. merge vs. reject-duplicates under blind UCB1-MCTS program
synthesis on real ARC-AGI-1 tasks (Stage 2 of the ARC plan). Task selection: real tasks whose
`solvers.py` reference solution uses only functions from
`classical_mcts_arc_program`'s curated 60-function subset -- the same
"verify exact reachability via a trusted oracle" principle as every other
domain's puzzle generator in this project, just reusing `solvers.py` as the
oracle instead of a hand-rolled BFS.

Three knobs beyond the original merge-vs-tree pair, each an independent
axis: `--arms` adds the reject-duplicates admission filter (merging's
distinctness without its statistic pooling), `--reward-mode graded` swaps
binary exact-match for cell-level goal progress (this is the sparsest-reward
domain in the repo and shows its smallest merge effect), and
`--merge-key-mode values` swaps the typed merge key for a coarser
observational-equivalence one. Wall-clock per repeat is reported for every
arm next to its node count, because a merge key here is O(state size) and a
node saving is not automatically a compute saving.

Result schema note: supporting N arms replaced this runner's old
`baseline_*`/`treatment_*` keys with `{"tree": ..., "arms": {...}}`, so its
output is not readable by tooling written against the two-arm convention the
other runners still use. Write to a fresh `--output-dir` rather than over an
existing two-arm record.
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


def _arm_config(arm: str, args) -> m.ClassicalMCTSConfig:
    """The three arms this experiment compares.

    "tree" and "merge" are the original pair. "reject" is the admission
    filter: it computes the same merge key but discards an expansion that
    lands on an already-seen state instead of pooling into the existing
    node, so expansions still land on distinct states while the backup rule
    and its statistics stay exactly as in the tree arm. That splits what
    merging does into its two halves so they can be attributed separately.
    """
    common = {"reward_mode": args.reward_mode, "merge_key_mode": args.merge_key_mode}
    if arm == "tree":
        return m.ClassicalMCTSConfig(merge_enabled=False, **common)
    if arm == "merge":
        return m.ClassicalMCTSConfig(merge_enabled=True, **common)
    if arm == "reject":
        return m.ClassicalMCTSConfig(merge_enabled=False, reject_duplicates=True, **common)
    raise ValueError(f"unknown arm {arm!r}")


def _run_arm(task, arm: str, args, budget: int, base_seed: int) -> dict:
    """Wall-clock is reported per repeat alongside the node count on
    purpose: a merge key is O(state size) to compute here (a frozenset over
    per-example values, and ARC values are grids), so a node-count saving
    is not automatically a compute saving and the two must be read
    together.

    `stop_on_solve` makes those two columns comparable across arms. It
    cannot affect the solve rate (budget is spent only on expansions), but
    without it a solved run freezes its node count at the solve point and
    then burns up to 20x budget in cheap terminal re-selection iterations,
    so an arm that solves more often would report fewer nodes and different
    wall-clock for reasons unrelated to efficiency."""
    solved = 0
    node_counts, merge_hits, rejected = [], [], []
    start = time.time()
    for r in range(args.repeats):
        rng = random.Random(base_seed * 100_003 + r)
        graph = m.run_search(task.train_inputs, task.train_outputs, _arm_config(arm, args),
                             budget, args.rollout_depth, rng, stop_on_solve=True)
        if m.is_solved(graph):
            solved += 1
        node_counts.append(len(graph.nodes))
        merge_hits.append(graph.merge_hits)
        rejected.append(graph.rejected)
    return {
        "rate": solved / args.repeats,
        "nodes": float(np.mean(node_counts)),
        "merge_hits": float(np.mean(merge_hits)),
        "rejected": float(np.mean(rejected)),
        "seconds_per_repeat": (time.time() - start) / args.repeats,
    }


def main():
    ap = argparse.ArgumentParser(description="Tree vs. merge vs. reject-duplicates: blind MCTS program synthesis on real ARC-AGI-1 tasks")
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-grid-dim", type=int, default=15)
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 2000, 5000])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--rollout-depth", type=int, default=5)
    ap.add_argument("--reward-mode", choices=["binary", "graded"], default="binary",
                    help="binary: rollouts score 1/0 on exact goal match (original). "
                         "graded: rollouts score the best cell-level goal progress seen. "
                         "Merging pools statistics, and pooling zeros carries no "
                         "information -- ARC is the sparsest-reward domain here and shows "
                         "the smallest merge effect, so reward density is a candidate mechanism.")
    ap.add_argument("--merge-key-mode", choices=["typed", "values"], default="typed",
                    help="typed: merge key is (type-tag-set, per-example-values) entries (original). "
                         "values: per-example-value tuples only -- coarser, closer to "
                         "observational equivalence, since the same concrete value reached "
                         "under different static tags is the same state observationally.")
    ap.add_argument("--arms", nargs="+", choices=["merge", "reject"], default=["merge", "reject"],
                    help="treatment arms to compare against the tree baseline")
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
            base_seed = args.seed * 1000 + i
            row = {"task_id": task.task_id, "reference_depth": depth,
                   "tree": _run_arm(task, "tree", args, budget, base_seed)}
            for arm in args.arms:
                row[arm] = _run_arm(task, arm, args, budget, base_seed)
            rows.append(row)

        per_arm = {}
        for arm in args.arms:
            diffs = np.array([r[arm]["rate"] - r["tree"]["rate"] for r in rows])
            ci_low, ci_high = _bootstrap_ci_mean_diff(diffs, seed=args.seed)
            try:
                _stat, wilcoxon_p = wilcoxon(diffs)
            except ValueError:
                wilcoxon_p = float("nan")  # defensive: scipy rejects an all-zero diff vector
            per_arm[arm] = {
                "mean": float(np.mean([r[arm]["rate"] for r in rows])),
                "diff_mean": float(diffs.mean()), "ci_low": ci_low, "ci_high": ci_high,
                "wilcoxon_p": float(wilcoxon_p),
                "nodes_mean": float(np.mean([r[arm]["nodes"] for r in rows])),
                "merge_hits_mean": float(np.mean([r[arm]["merge_hits"] for r in rows])),
                "rejected_mean": float(np.mean([r[arm]["rejected"] for r in rows])),
                "seconds_mean": float(np.mean([r[arm]["seconds_per_repeat"] for r in rows])),
            }
        tree_summary = {
            "mean": float(np.mean([r["tree"]["rate"] for r in rows])),
            "nodes_mean": float(np.mean([r["tree"]["nodes"] for r in rows])),
            "seconds_mean": float(np.mean([r["tree"]["seconds_per_repeat"] for r in rows])),
        }
        print(f"budget={budget}: tree={tree_summary['mean']:.3f} "
              f"nodes={tree_summary['nodes_mean']:.1f} sec={tree_summary['seconds_mean']:.2f}")
        for arm, stats in per_arm.items():
            print(f"  {arm:7s}: rate={stats['mean']:.3f} diff={stats['diff_mean']:+.3f} "
                  f"CI=[{stats['ci_low']:+.3f},{stats['ci_high']:+.3f}] p={stats['wilcoxon_p']:.4f} "
                  f"nodes={stats['nodes_mean']:.1f} sec={stats['seconds_mean']:.2f} "
                  f"merge_hits={stats['merge_hits_mean']:.1f} rejected={stats['rejected_mean']:.1f}")
        results[budget] = {"rows": rows, "tree": tree_summary, "arms": per_arm}

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
