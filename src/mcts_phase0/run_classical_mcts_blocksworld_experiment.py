"""CLI: classical MCTS merging on Blocksworld -- no LLM, no learned metric.
Fifth classical domain, reusing pyperplan's own Task/Operator/BFS
primitives (see blocksworld_engine.py). Mirrors the other classical
runners' structure and reuses the generic bootstrap-CI helper.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .classical_mcts_blocksworld import ClassicalMCTSConfig, is_solved, run_search
from .datasets import blocksworld_engine
from .run_classical_mcts_experiment import _bootstrap_ci_mean_diff


def _solve_rate(instance, task, goal, merge_enabled: bool, budget: int, rollout_depth: int, repeats: int, base_seed: int) -> tuple[float, float]:
    solved = 0
    node_counts = []
    for r in range(repeats):
        rng = random.Random(base_seed * 100_003 + r)
        config = ClassicalMCTSConfig(merge_enabled=merge_enabled)
        graph = run_search(instance.start_state, goal, task, config, budget, rollout_depth, rng)
        if is_solved(graph):
            solved += 1
        node_counts.append(len(graph.nodes))
    return solved / repeats, float(np.mean(node_counts))


def main():
    ap = argparse.ArgumentParser(description="Classical (no-LLM) MCTS merge-vs-tree experiment on Blocksworld")
    ap.add_argument("--num-blocks", type=int, default=5)
    ap.add_argument("--plan-lengths", type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--n-puzzles", type=int, default=15, help="puzzles per plan length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--rollout-depth", type=int, default=30)
    ap.add_argument("--output-dir", default="results/classical_mcts_blocksworld_experiment")
    args = ap.parse_args()

    start = time.time()
    task = blocksworld_engine.make_task(args.num_blocks)
    goal = blocksworld_engine.goal_state(args.num_blocks)
    puzzles = []
    for pl in args.plan_lengths:
        puzzles.extend(blocksworld_engine.generate_puzzles(n=args.n_puzzles, seed=args.seed, num_blocks=args.num_blocks, target_plan_length=pl))
    print(f"generated {len(puzzles)} puzzles across plan_lengths {args.plan_lengths} in {time.time() - start:.2f}s")

    results = {}
    for budget in args.budgets:
        rows = []
        for i, inst in enumerate(puzzles):
            base_rate, base_nodes = _solve_rate(inst, task, goal, False, budget, args.rollout_depth, args.repeats, base_seed=args.seed * 1000 + i)
            treat_rate, treat_nodes = _solve_rate(inst, task, goal, True, budget, args.rollout_depth, args.repeats, base_seed=args.seed * 1000 + i)
            rows.append({
                "id": inst.id, "plan_length": inst.plan_length,
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
            wilcoxon_stat, wilcoxon_p = wilcoxon(diffs)
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
        json.dump({"args": vars(args), "elapsed_sec": elapsed, "results": results}, f, indent=2)
    print(f"done in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
