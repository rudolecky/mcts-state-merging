"""Blocksworld: real hFFHeuristic-guided tree search, two stages, matched
budget, no LLM. Mirrors run_heuristic_merge_experiment_puzzle.py's
structure -- see that module's docstring for the shared calibration
finding (deterministic heuristic needs a much smaller UCB1 exploration
constant than blind rollout mode's usual c=1.4). Unlike the 8-puzzle and
Sokoban, hFFHeuristic discriminates well immediately (a real relaxed-plan
heuristic, not a hand-rolled one) -- no Sokoban-style degenerate-heuristic
fix was needed here.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from pyperplan.heuristics.relaxation import hFFHeuristic
from scipy.stats import wilcoxon

from .classical_mcts_blocksworld import ClassicalMCTSConfig, is_solved, run_random_search, run_search
from .datasets.blocksworld_engine import generate_puzzles, goal_state, make_task


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


def run_tier(num_blocks, target_plan_length, c, budgets, n_puzzles, repeats, rollout_depth, seed=0):
    task = make_task(num_blocks)
    goal = goal_state(num_blocks)
    heuristic = hFFHeuristic(task)
    puzzles = generate_puzzles(n=n_puzzles, seed=seed, num_blocks=num_blocks, target_plan_length=target_plan_length)
    tier_results = {"target_plan_length": target_plan_length, "c": c, "stage1": [], "stage2": []}

    for budget in budgets:
        t0 = time.time()
        rows1, rows2 = [], []
        for i, inst in enumerate(puzzles):
            heuristic_solved = random_solved = merge_solved = 0
            for r in range(repeats):
                rng = random.Random(i * 100003 + r)
                cfg_tree = ClassicalMCTSConfig(merge_enabled=False, value_source="heuristic", c=c)
                g = run_search(inst.start_state, goal, task, cfg_tree, budget, rollout_depth, rng, heuristic=heuristic)
                heuristic_solved += int(is_solved(g))

                rng2 = random.Random(i * 100003 + r)
                random_solved += int(run_random_search(inst.start_state, goal, task, budget, rollout_depth, rng2))

                rng3 = random.Random(i * 100003 + r)
                cfg_merge = ClassicalMCTSConfig(merge_enabled=True, value_source="heuristic", c=c)
                g_merge = run_search(inst.start_state, goal, task, cfg_merge, budget, rollout_depth, rng3, heuristic=heuristic)
                merge_solved += int(is_solved(g_merge))
            rows1.append({"id": inst.id, "tree": heuristic_solved / repeats, "random": random_solved / repeats})
            rows2.append({"id": inst.id, "tree": heuristic_solved / repeats, "merge": merge_solved / repeats})

        tm, rm, dm1, ci1, p1 = np.mean([r["tree"] for r in rows1]), np.mean([r["random"] for r in rows1]), *_stats(rows1, "tree", "random")
        _, _, dm2, ci2, p2 = 0, 0, *_stats(rows2, "tree", "merge")
        mm = np.mean([r["merge"] for r in rows2])
        elapsed = time.time() - t0
        print(f"  pl={target_plan_length} budget={budget}: tree={tm:.3f} random={rm:.3f} diff={dm1:.3f} "
              f"CI=[{ci1[0]:.3f},{ci1[1]:.3f}] p={p1:.4f} | merge={mm:.3f} merge_diff={dm2:.3f} "
              f"CI=[{ci2[0]:.3f},{ci2[1]:.3f}] p={p2:.4f} (t={elapsed:.1f}s)")
        tier_results["stage1"].append({"budget": budget, "tree": tm, "random": rm, "diff": dm1, "ci": ci1, "p": p1})
        tier_results["stage2"].append({"budget": budget, "tree": tm, "merge": mm, "diff": dm2, "ci": ci2, "p": p2})
    return tier_results


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Blocksworld: heuristic-guided tree vs random vs merge, two stages")
    ap.add_argument("--num-blocks", type=int, default=5)
    ap.add_argument("--n-puzzles", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--rollout-depth", type=int, default=15)
    ap.add_argument("--output", default="results/heuristic_merge_experiment_blocksworld.json")
    args = ap.parse_args()

    all_results = []
    print("=== BLOCKSWORLD, pl=6, c=0.02 ===")
    # only 7 distinct num_blocks=5 states exist at exactly plan_length=6 (confirmed via
    # rejection-sampling exhaustion, not an arbitrary cap) -- capped below args.n_puzzles's
    # default of 8 for this tier only.
    all_results.append(run_tier(args.num_blocks, target_plan_length=6, c=0.02, budgets=[10, 20, 40, 80],
                                 n_puzzles=min(args.n_puzzles, 7), repeats=args.repeats, rollout_depth=args.rollout_depth))
    print("=== BLOCKSWORLD, pl=10, c=0.02 ===")
    all_results.append(run_tier(args.num_blocks, target_plan_length=10, c=0.02, budgets=[20, 40, 80, 150, 300],
                                 n_puzzles=args.n_puzzles, repeats=args.repeats, rollout_depth=args.rollout_depth))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
