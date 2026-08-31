"""CLI: K-sampling classical MCTS merge-vs-tree experiment on Three Men's
Morris -- no LLM, no learned metric. Tests whether Connect Four's finding
(guidance-independent null flips to guidance-dependent once the expansion
mechanism loses its structural anti-duplicate guarantee) generalizes past
Connect Four, or whether Morris's adversarial-interleaving null survives
even here. Mirrors run_classical_mcts_ksample_experiment.py's structure.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .classical_mcts_morris_ksample import ClassicalMCTSConfig, is_solved, run_search
from .datasets import morris_engine
from .run_classical_mcts_experiment import _bootstrap_ci_mean_diff


def _solve_rate(instance, merge_enabled: bool, budget: int, K: int, rollout_depth: int, guidance_depth_cap: int | None, repeats: int, base_seed: int) -> tuple[float, float]:
    board = morris_engine.replay_placements(instance.pre_moves)
    solved = 0
    node_counts = []
    for r in range(repeats):
        rng = random.Random(base_seed * 100_003 + r)
        config = ClassicalMCTSConfig(merge_enabled=merge_enabled, K=K, rollout_depth=rollout_depth, guidance_depth_cap=guidance_depth_cap)
        graph = run_search(board, instance.to_move, config, budget, rng)
        if is_solved(graph):
            solved += 1
        node_counts.append(len(graph.nodes))
    return solved / repeats, float(np.mean(node_counts))


def main():
    ap = argparse.ArgumentParser(description="K-sampling classical (no-LLM) MCTS merge-vs-tree experiment on Three Men's Morris")
    ap.add_argument("--k-plies", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--n-puzzles", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[2, 3, 5])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--rollout-depth", type=int, default=20)
    ap.add_argument("--guidance-depth-cap", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--output-dir", default="results/classical_mcts_morris_ksample_experiment")
    args = ap.parse_args()

    start = time.time()
    puzzles = []
    for kp in args.k_plies:
        puzzles.extend(morris_engine.generate_puzzles(n=args.n_puzzles, seed=args.seed, k_plies=kp))
    print(f"generated {len(puzzles)} puzzles across k_plies {args.k_plies} in {time.time() - start:.2f}s")

    results = {}
    for budget in args.budgets:
        rows = []
        for i, inst in enumerate(puzzles):
            base_rate, base_nodes = _solve_rate(inst, False, budget, args.K, args.rollout_depth, args.guidance_depth_cap, args.repeats, base_seed=args.seed * 1000 + i)
            treat_rate, treat_nodes = _solve_rate(inst, True, budget, args.K, args.rollout_depth, args.guidance_depth_cap, args.repeats, base_seed=args.seed * 1000 + i)
            rows.append({
                "id": inst.id, "k_plies": inst.k_plies,
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
