"""2x2x2 Rubik's Cube: GUCT-Uniform (LCB1-Uniform + Full Bellman min backup,
exact-oracle reward), two stages, matched budget. Mirrors
run_heuristic_merge_experiment_rubiks.py's structure -- the real redesign
attempt at rescuing merging after two threshold-based mitigations
(merge_parent_cap, merge_visit_cap on classical_mcts_rubiks.py) both hit
an identical cliff (see RUBIKS_CUBE_FINDINGS.md). No exploration constant
to calibrate (LCB1-Uniform has none), but budget/difficulty still needed
their own sweep -- this bandit shows sharp floor-to-ceiling jumps rather
than a gradual solve-rate curve (same behavior already documented for
Blocksworld in BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md).
"""

from __future__ import annotations

import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .classical_mcts_rubiks import run_random_search
from .datasets import rubiks_cube_engine as rc
from .guct_uniform_rubiks import GUCTUniformConfig, is_solved, run_search


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


def run_tier(target_distance, budgets, n_puzzles, repeats, distances, seed=0):
    puzzles = rc.generate_puzzles(n=n_puzzles, seed=seed, target_distance=target_distance, distances=distances)
    tier_results = {"target_distance": target_distance, "stage1": [], "stage2": []}

    for budget in budgets:
        t0 = time.time()
        rows1, rows2 = [], []
        for i, inst in enumerate(puzzles):
            tree_solved = random_solved = merge_solved = 0
            for r in range(repeats):
                rng = random.Random(i * 100003 + r)
                cfg_tree = GUCTUniformConfig(merge_enabled=False)
                g = run_search(inst.start_state, cfg_tree, budget, rng, distances)
                tree_solved += int(is_solved(g))

                rng2 = random.Random(i * 100003 + r)
                random_solved += int(run_random_search(inst.start_state, budget, rollout_depth=15, rng=rng2))

                rng3 = random.Random(i * 100003 + r)
                cfg_merge = GUCTUniformConfig(merge_enabled=True)
                g_merge = run_search(inst.start_state, cfg_merge, budget, rng3, distances)
                merge_solved += int(is_solved(g_merge))
            rows1.append({"id": inst.id, "tree": tree_solved / repeats, "random": random_solved / repeats})
            rows2.append({"id": inst.id, "tree": tree_solved / repeats, "merge": merge_solved / repeats})

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

    ap = argparse.ArgumentParser(description="Rubik's Cube: GUCT-Uniform tree vs random vs merge, two stages")
    ap.add_argument("--n-puzzles", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--distances-path", default="results/rubiks_cube_bfs_distances.pkl")
    ap.add_argument("--output", default="results/guct_uniform_experiment_rubiks.json")
    args = ap.parse_args()

    with open(args.distances_path, "rb") as f:
        distances = pickle.load(f)
    print(f"loaded {len(distances)} states from {args.distances_path}")

    all_results = []
    print("=== RUBIK'S CUBE GUCT-UNIFORM, td=6 ===")
    all_results.append(run_tier(target_distance=6, budgets=[40, 60, 70, 90],
                                 n_puzzles=args.n_puzzles, repeats=args.repeats, distances=distances))
    print("=== RUBIK'S CUBE GUCT-UNIFORM, td=8 ===")
    all_results.append(run_tier(target_distance=8, budgets=[50, 70, 80, 100],
                                 n_puzzles=args.n_puzzles, repeats=args.repeats, distances=distances))
    print("=== RUBIK'S CUBE GUCT-UNIFORM, td=10 ===")
    all_results.append(run_tier(target_distance=10, budgets=[90, 110, 130, 170],
                                 n_puzzles=args.n_puzzles, repeats=args.repeats, distances=distances))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": {k: v for k, v in vars(args).items()}, "results": all_results}, f, indent=2,
                   default=lambda o: list(o) if isinstance(o, tuple) else o)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
