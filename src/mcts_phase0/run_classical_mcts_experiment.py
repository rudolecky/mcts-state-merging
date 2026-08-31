"""CLI: classical MCTS merging on Connect Four -- no LLM, no learned metric.
Generates N forced-win puzzles (exact difficulty by construction, no
calibration step needed), runs baseline (tree) vs treatment (exact-state DAG)
at matched simulation budgets, R independent repeats each, and reports the
per-puzzle solve-rate comparison plus mechanism diagnostics.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .classical_mcts import ClassicalMCTSConfig, is_solved, run_search
from .datasets import connect_four


def _solve_rate(instance, merge_enabled: bool, budget: int, repeats: int, base_seed: int, guidance_depth_cap: int | None = None) -> tuple[float, float]:
    """Returns (solve_rate, mean_unique_nodes) over `repeats` independent runs."""
    solved = 0
    node_counts = []
    for r in range(repeats):
        rng = random.Random(base_seed * 100_003 + r)
        config = ClassicalMCTSConfig(merge_enabled=merge_enabled, guidance_depth_cap=guidance_depth_cap)
        graph = run_search(instance.pre_moves, instance.to_move, instance.width, instance.height, config, budget, rng)
        if is_solved(graph):
            solved += 1
        node_counts.append(len(graph.nodes))
    return solved / repeats, float(np.mean(node_counts))


def _bootstrap_ci_mean_diff(diffs: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    ap = argparse.ArgumentParser(description="Classical (no-LLM) MCTS merge-vs-tree experiment on Connect Four")
    ap.add_argument("--n-low", type=int, default=30, help="number of puzzles (path_count-stratified low bucket)")
    ap.add_argument("--n-high", type=int, default=0)
    ap.add_argument("--width", type=int, default=5)
    ap.add_argument("--height", type=int, default=4)
    ap.add_argument("--k-plies", type=int, default=3)
    ap.add_argument("--max-pre-moves", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100])
    ap.add_argument("--repeats", type=int, default=50, help="independent MCTS runs per puzzle per condition per budget")
    ap.add_argument("--guidance-depth-cap", type=int, default=None,
                     help="cap the rollout at this many plies, returning a neutral 0.5 if exhausted without a win "
                          "(cheap/noisy guidance ablation); omit for the honest full-rollout default")
    ap.add_argument("--output-dir", default="results/classical_mcts_experiment")
    args = ap.parse_args()

    start = time.time()
    puzzles = connect_four.generate_puzzles(
        n_low=args.n_low, n_high=args.n_high, seed=args.seed,
        width=args.width, height=args.height, k_plies=args.k_plies, max_pre_moves=args.max_pre_moves,
    )
    print(f"generated {len(puzzles)} puzzles in {time.time() - start:.2f}s")

    results = {}
    for budget in args.budgets:
        rows = []
        for i, inst in enumerate(puzzles):
            base_rate, base_nodes = _solve_rate(inst, False, budget, args.repeats, base_seed=args.seed * 1000 + i, guidance_depth_cap=args.guidance_depth_cap)
            treat_rate, treat_nodes = _solve_rate(inst, True, budget, args.repeats, base_seed=args.seed * 1000 + i, guidance_depth_cap=args.guidance_depth_cap)
            rows.append({
                "id": inst.id, "path_count": inst.path_count,
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
            wilcoxon_p = float("nan")  # all-zero differences

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
