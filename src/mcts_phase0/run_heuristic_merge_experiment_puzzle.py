"""8-puzzle: real Manhattan-distance-heuristic-guided tree search, two
stages, matched budget, no LLM (fast, no pilot needed -- see
cheerful-jumping-moler.md's plan).

Stage 1 -- does heuristic-guided tree actually beat random restarts? (the
prerequisite the whole question hinges on): value_source="heuristic",
merge_enabled=False vs. run_random_search.

Stage 2 -- the target question: does merging still help once tree
genuinely beats random? value_source="heuristic", merge_enabled=False vs.
merge_enabled=True.

Calibration note (see REAL_HEURISTIC_MERGE_FINDINGS.md): UCB1's exploration
constant c=1.4 (theoretically calibrated for noisy stochastic rollout
rewards) nearly nullifies a deterministic heuristic signal -- at c=1.4 the
heuristic-guided tree solved 0/30 on an 8-move puzzle at budget=40, but
28-35/50 at c=0.02-0.1 same budget. A deterministic per-state value has no
sampling noise to justify UCB1's Hoeffding-derived exploration bonus, so
value_source="heuristic" needs a much smaller c than blind rollout mode --
exposed here as an explicit --c argument per difficulty tier, not
hardcoded, since the right value is domain/difficulty-dependent.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .classical_mcts_puzzle import ClassicalMCTSConfig, is_solved, run_random_search, run_search
from .datasets import sliding_puzzle_engine as spe


def _stats(rows, key_a, key_b):
    diffs = np.array([r[key_b] - r[key_a] for r in rows])
    rng = np.random.default_rng(0)
    boot = [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(3000)]
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    try:
        _, p = wilcoxon(diffs)
    except ValueError:
        p = float("nan")
    return diffs.mean(), ci, p


def run_tier(width, height, target_distance, c, budgets, n_puzzles, repeats, rollout_depth, seed=0):
    dist = spe.bfs_distances(width, height)
    puzzles = spe.generate_puzzles(n=n_puzzles, seed=seed, width=width, height=height,
                                    target_distance=target_distance, distances=dist)
    tier_results = {"target_distance": target_distance, "c": c, "stage1": [], "stage2": []}

    for budget in budgets:
        t0 = time.time()
        rows1, rows2 = [], []
        for i, inst in enumerate(puzzles):
            heuristic_solved = random_solved = merge_solved = 0
            for r in range(repeats):
                rng = random.Random(i * 100003 + r)
                cfg_tree = ClassicalMCTSConfig(merge_enabled=False, value_source="heuristic", c=c)
                g = run_search(inst.start_state, width, height, cfg_tree, budget, rollout_depth, rng)
                heuristic_solved += int(is_solved(g))

                rng2 = random.Random(i * 100003 + r)
                random_solved += int(run_random_search(inst.start_state, width, height, budget, rollout_depth, rng2))

                rng3 = random.Random(i * 100003 + r)
                cfg_merge = ClassicalMCTSConfig(merge_enabled=True, value_source="heuristic", c=c)
                g_merge = run_search(inst.start_state, width, height, cfg_merge, budget, rollout_depth, rng3)
                merge_solved += int(is_solved(g_merge))
            rows1.append({"id": inst.id, "tree": heuristic_solved / repeats, "random": random_solved / repeats})
            rows2.append({"id": inst.id, "tree": heuristic_solved / repeats, "merge": merge_solved / repeats})

        tm, rm, dm1, ci1, p1 = np.mean([r["tree"] for r in rows1]), np.mean([r["random"] for r in rows1]), *_stats(rows1, "tree", "random")
        _, _, dm2, ci2, p2 = 0, 0, *_stats(rows2, "tree", "merge")
        mm = np.mean([r["merge"] for r in rows2])
        elapsed = time.time() - t0
        print(f"  td={target_distance} budget={budget}: tree={tm:.3f} random={rm:.3f} diff={dm1:.3f} "
              f"CI=[{ci1[0]:.3f},{ci1[1]:.3f}] p={p1:.4f} | merge={mm:.3f} merge_diff={dm2:.3f} "
              f"CI=[{ci2[0]:.3f},{ci2[1]:.3f}] p={p2:.4f} (t={elapsed:.1f}s)")
        tier_results["stage1"].append({"budget": budget, "tree": tm, "random": rm, "diff": dm1, "ci": ci1, "p": p1})
        tier_results["stage2"].append({"budget": budget, "tree": tm, "merge": mm, "diff": dm2, "ci": ci2, "p": p2})
    return tier_results


def main():
    import argparse

    ap = argparse.ArgumentParser(description="8-puzzle: heuristic-guided tree vs random vs merge, two stages")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--height", type=int, default=3)
    ap.add_argument("--n-puzzles", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--rollout-depth", type=int, default=30)
    ap.add_argument("--output", default="results/heuristic_merge_experiment_puzzle.json")
    args = ap.parse_args()

    all_results = []
    print("=== 8-PUZZLE, td=8, c=0.02 ===")
    all_results.append(run_tier(args.width, args.height, target_distance=8, c=0.02,
                                 budgets=[10, 20, 30, 40, 60, 80, 120],
                                 n_puzzles=args.n_puzzles, repeats=args.repeats, rollout_depth=args.rollout_depth))
    print("=== 8-PUZZLE, td=12, c=0.1 ===")
    all_results.append(run_tier(args.width, args.height, target_distance=12, c=0.1,
                                 budgets=[100, 300, 500, 800],
                                 n_puzzles=args.n_puzzles, repeats=args.repeats, rollout_depth=args.rollout_depth))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
