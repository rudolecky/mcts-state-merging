"""8-puzzle and Sokoban under GUCT-Uniform (Full Bellman min-backup +
LCB1-Uniform, exact oracle) -- tests whether the Rubik's Cube's own
"merge and tree become solve-rate-identical, merge still wins on node
count" result generalizes to domains where merging was already positive
under UCB1+MC-average (REAL_HEURISTIC_MERGE_FINDINGS.md), or whether it
was cube-specific. Mirrors run_guct_uniform_rubiks_experiment.py's
structure; also records mean node count per condition (not just solve
rate) since that's the efficiency signal the cube's own answer hinged on.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import mcts_phase0.guct_uniform_puzzle as gup
import mcts_phase0.guct_uniform_sokoban as gus
from .datasets import sliding_puzzle_engine as spe
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


def run_puzzle_tier(target_distance, budgets, n_puzzles, repeats, distances, width=3, height=3, seed=0):
    puzzles = spe.generate_puzzles(n=n_puzzles, seed=seed, width=width, height=height,
                                    target_distance=target_distance, distances=distances)
    tier = {"domain": "8-puzzle", "target_distance": target_distance, "budgets": []}
    for budget in budgets:
        t0 = time.time()
        rows = []
        for i, inst in enumerate(puzzles):
            tree_solved = merge_solved = 0
            tree_nodes = merge_nodes = 0
            for r in range(repeats):
                rng = random.Random(i * 100003 + r)
                g = gup.run_search(inst.start_state, width, height, gup.GUCTUniformConfig(merge_enabled=False), budget, rng, distances)
                tree_solved += int(gup.is_solved(g))
                tree_nodes += len(g.nodes)
                rng2 = random.Random(i * 100003 + r)
                g2 = gup.run_search(inst.start_state, width, height, gup.GUCTUniformConfig(merge_enabled=True), budget, rng2, distances)
                merge_solved += int(gup.is_solved(g2))
                merge_nodes += len(g2.nodes)
            rows.append({"id": inst.id, "tree": tree_solved / repeats, "merge": merge_solved / repeats,
                         "tree_nodes": tree_nodes / repeats, "merge_nodes": merge_nodes / repeats})
        tm = np.mean([r["tree"] for r in rows])
        mm = np.mean([r["merge"] for r in rows])
        _, _, dm, ci, p = 0, 0, *_stats(rows, "tree", "merge")
        tn = np.mean([r["tree_nodes"] for r in rows])
        mn = np.mean([r["merge_nodes"] for r in rows])
        elapsed = time.time() - t0
        print(f"  8-puzzle td={target_distance} budget={budget}: tree={tm:.3f} merge={mm:.3f} diff={dm:.3f} "
              f"CI=[{ci[0]:.3f},{ci[1]:.3f}] p={p:.4f} nodes tree={tn:.1f} merge={mn:.1f} (t={elapsed:.1f}s)")
        tier["budgets"].append({"budget": budget, "tree": tm, "merge": mm, "diff": dm, "ci": ci, "p": p,
                                 "tree_nodes": tn, "merge_nodes": mn})
    return tier


def run_sokoban_tier(target_distance, budgets, n_puzzles, repeats, distances, seed=0):
    puzzles = se.generate_puzzles(n=n_puzzles, seed=seed, target_distance=target_distance, distances=distances)
    tier = {"domain": "sokoban", "target_distance": target_distance, "budgets": []}
    for budget in budgets:
        t0 = time.time()
        rows = []
        for i, inst in enumerate(puzzles):
            tree_solved = merge_solved = 0
            tree_nodes = merge_nodes = 0
            for r in range(repeats):
                rng = random.Random(i * 100003 + r)
                g = gus.run_search(inst.start_state, gus.GUCTUniformConfig(merge_enabled=False), budget, rng, distances)
                tree_solved += int(gus.is_solved(g))
                tree_nodes += len(g.nodes)
                rng2 = random.Random(i * 100003 + r)
                g2 = gus.run_search(inst.start_state, gus.GUCTUniformConfig(merge_enabled=True), budget, rng2, distances)
                merge_solved += int(gus.is_solved(g2))
                merge_nodes += len(g2.nodes)
            rows.append({"id": inst.id, "tree": tree_solved / repeats, "merge": merge_solved / repeats,
                         "tree_nodes": tree_nodes / repeats, "merge_nodes": merge_nodes / repeats})
        tm = np.mean([r["tree"] for r in rows])
        mm = np.mean([r["merge"] for r in rows])
        _, _, dm, ci, p = 0, 0, *_stats(rows, "tree", "merge")
        tn = np.mean([r["tree_nodes"] for r in rows])
        mn = np.mean([r["merge_nodes"] for r in rows])
        elapsed = time.time() - t0
        print(f"  sokoban td={target_distance} budget={budget}: tree={tm:.3f} merge={mm:.3f} diff={dm:.3f} "
              f"CI=[{ci[0]:.3f},{ci[1]:.3f}] p={p:.4f} nodes tree={tn:.1f} merge={mn:.1f} (t={elapsed:.1f}s)")
        tier["budgets"].append({"budget": budget, "tree": tm, "merge": mm, "diff": dm, "ci": ci, "p": p,
                                 "tree_nodes": tn, "merge_nodes": mn})
    return tier


def main():
    import argparse

    ap = argparse.ArgumentParser(description="8-puzzle + Sokoban under GUCT-Uniform: does merge/tree solve-rate parity generalize?")
    ap.add_argument("--n-puzzles", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--output", default="results/guct_uniform_generalization_experiment.json")
    args = ap.parse_args()

    puzzle_dist = spe.bfs_distances(3, 3)
    sokoban_dist = se.bfs_distances()

    all_results = []
    print("=== 8-PUZZLE GUCT-UNIFORM, td=8 ===")
    all_results.append(run_puzzle_tier(8, [15, 20, 25, 30], args.n_puzzles, args.repeats, puzzle_dist))
    print("=== 8-PUZZLE GUCT-UNIFORM, td=12 ===")
    all_results.append(run_puzzle_tier(12, [35, 40, 45, 50], args.n_puzzles, args.repeats, puzzle_dist))
    print("=== SOKOBAN GUCT-UNIFORM, td=6 ===")
    all_results.append(run_sokoban_tier(6, [15, 20, 25, 30], args.n_puzzles, args.repeats, sokoban_dist))
    print("=== SOKOBAN GUCT-UNIFORM, td=8 ===")
    all_results.append(run_sokoban_tier(8, [20, 25, 30, 40], args.n_puzzles, args.repeats, sokoban_dist))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": all_results}, f, indent=2,
                   default=lambda o: list(o) if isinstance(o, tuple) else o)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
