"""Sokoban: real box-goal + player-box heuristic-guided tree search, two
stages, matched budget, no LLM. Mirrors
run_heuristic_merge_experiment_puzzle.py's structure exactly -- see that
module's docstring for the shared calibration finding (deterministic
heuristic needs a much smaller UCB1 exploration constant than blind
rollout mode's usual c=1.4).

Calibration note specific to Sokoban (see REAL_HEURISTIC_MERGE_FINDINGS.md):
the box-goal-only heuristic gives zero discrimination among purely-walking
moves (the box doesn't move, so h is unchanged), which starved UCB1 of any
signal for most legal moves and solved 0/50 across a full (c, budget)
sweep. Adding a player-to-nearest-box distance term
(_player_box_heuristic) fixed this -- see classical_mcts_sokoban.py.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .classical_mcts_sokoban import ClassicalMCTSConfig, is_solved, run_random_search, run_search
from .datasets import sokoban_engine as se


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


def run_tier(target_distance, c, budgets, n_puzzles, repeats, rollout_depth, distances, seed=0):
    puzzles = se.generate_puzzles(n=n_puzzles, seed=seed, target_distance=target_distance, distances=distances)
    tier_results = {"target_distance": target_distance, "c": c, "stage1": [], "stage2": []}

    for budget in budgets:
        t0 = time.time()
        rows1, rows2 = [], []
        for i, inst in enumerate(puzzles):
            heuristic_solved = random_solved = merge_solved = 0
            for r in range(repeats):
                rng = random.Random(i * 100003 + r)
                cfg_tree = ClassicalMCTSConfig(merge_enabled=False, value_source="heuristic", c=c)
                g = run_search(inst.start_state, cfg_tree, budget, rollout_depth, rng)
                heuristic_solved += int(is_solved(g))

                rng2 = random.Random(i * 100003 + r)
                random_solved += int(run_random_search(inst.start_state, budget, rollout_depth, rng2))

                rng3 = random.Random(i * 100003 + r)
                cfg_merge = ClassicalMCTSConfig(merge_enabled=True, value_source="heuristic", c=c)
                g_merge = run_search(inst.start_state, cfg_merge, budget, rollout_depth, rng3)
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

    ap = argparse.ArgumentParser(description="Sokoban: heuristic-guided tree vs random vs merge, two stages")
    ap.add_argument("--n-puzzles", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--rollout-depth", type=int, default=20)
    ap.add_argument("--output", default="results/heuristic_merge_experiment_sokoban.json")
    args = ap.parse_args()

    distances = se.bfs_distances()
    all_results = []
    print("=== SOKOBAN, td=6, c=0.02 ===")
    all_results.append(run_tier(target_distance=6, c=0.02, budgets=[20, 40, 80, 160, 300],
                                 n_puzzles=args.n_puzzles, repeats=args.repeats,
                                 rollout_depth=args.rollout_depth, distances=distances))
    print("=== SOKOBAN, td=8, c=0.02 ===")
    all_results.append(run_tier(target_distance=8, c=0.02, budgets=[20, 40, 80, 160, 300, 500],
                                 n_puzzles=args.n_puzzles, repeats=args.repeats,
                                 rollout_depth=args.rollout_depth, distances=distances))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
