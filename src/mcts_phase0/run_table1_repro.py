"""CLI: the minimal-slice Table 1 reproduction -- GBFS, plain GUCT, and
GUCT-Uniform (tree, matching the paper's own protocol -- it never runs a
merge condition), all on `h^FF`, against real IPC Blocksworld and Gripper
instances (see data/pddl_benchmarks/README.md for exact provenance). Also
reports a bonus fourth condition, GUCT-Uniform with merging enabled, since
it turned out to be directly relevant to what was found (see
results/TABLE1_REPRODUCTION_FINDINGS.md).

Protocol matches the paper's own where it's the thing being reproduced: a
node-evaluation cap (default 10,000, their number) and 5 seeds per
randomized algorithm (their number). GBFS is deterministic (no
tie-breaking randomness in this implementation), so it's run once per
instance, not five times.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from pyperplan.heuristics.relaxation import hFFHeuristic

from .guct_uniform_blocksworld import GUCTUniformConfig, is_solved as gu_is_solved, run_search as gu_run_search
from .paper_repro import guct as guct_mod
from .paper_repro.capped_gbfs import capped_gbfs
from .paper_repro.pddl_loader import load_task

INSTANCES = [
    ("blocks", "probBLOCKS-4-0"),
    ("blocks", "probBLOCKS-5-0"),
    ("blocks", "probBLOCKS-6-0"),
    ("blocks", "probBLOCKS-7-0"),
    ("blocks", "probBLOCKS-8-0"),
    ("gripper", "prob01"),
    ("gripper", "prob02"),
    ("gripper", "prob03"),
    ("gripper", "prob04"),
    ("gripper", "prob06"),
    ("gripper", "prob08"),
    ("gripper", "prob10"),
]


def main():
    ap = argparse.ArgumentParser(description="Minimal-slice reproduction of arXiv:2405.18248's Table 1")
    ap.add_argument("--benchmarks-dir", default="data/pddl_benchmarks")
    ap.add_argument("--node-eval-cap", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--output-dir", default="results/table1_repro_experiment")
    args = ap.parse_args()

    base = Path(args.benchmarks_dir)
    start = time.time()
    rows = []
    for domain_name, problem_name in INSTANCES:
        domain_file = base / domain_name / "domain.pddl"
        problem_file = base / domain_name / f"{problem_name}.pddl"
        task = load_task(str(domain_file), str(problem_file))
        heuristic = hFFHeuristic(task)

        gbfs_solved, gbfs_evals = capped_gbfs(task, heuristic, args.node_eval_cap)

        guct_solved = 0
        gu_tree_solved = 0
        gu_merge_solved = 0
        for seed in range(args.seeds):
            rng = random.Random(seed)
            g = guct_mod.run_search(task.initial_state, task, guct_mod.GUCTConfig(merge_enabled=False), args.node_eval_cap, rng, heuristic)
            guct_solved += int(guct_mod.is_solved(g))

            rng = random.Random(seed)
            gu_tree = gu_run_search(task.initial_state, task.goals, task, GUCTUniformConfig(merge_enabled=False), args.node_eval_cap, rng, heuristic)
            gu_tree_solved += int(gu_is_solved(gu_tree))

            rng = random.Random(seed)
            gu_merge = gu_run_search(task.initial_state, task.goals, task, GUCTUniformConfig(merge_enabled=True), args.node_eval_cap, rng, heuristic)
            gu_merge_solved += int(gu_is_solved(gu_merge))

        row = {
            "domain": domain_name, "instance": problem_name, "num_operators": len(task.operators),
            "gbfs_solved": gbfs_solved, "gbfs_evals": gbfs_evals,
            "guct_solved_of_seeds": guct_solved,
            "guct_uniform_tree_solved_of_seeds": gu_tree_solved,
            "guct_uniform_merge_solved_of_seeds": gu_merge_solved,
        }
        rows.append(row)
        print(f"{domain_name}/{problem_name}: ops={row['num_operators']} "
              f"GBFS={'solved' if gbfs_solved else 'FAILED'}({gbfs_evals}) "
              f"GUCT={guct_solved}/{args.seeds} "
              f"GUCT-Uniform(tree)={gu_tree_solved}/{args.seeds} "
              f"GUCT-Uniform(merge)={gu_merge_solved}/{args.seeds}")

    n = len(rows)
    summary = {
        "gbfs_instances_solved": sum(1 for r in rows if r["gbfs_solved"]),
        "guct_instance_solve_rate": sum(r["guct_solved_of_seeds"] for r in rows) / (n * args.seeds),
        "guct_uniform_tree_instance_solve_rate": sum(r["guct_uniform_tree_solved_of_seeds"] for r in rows) / (n * args.seeds),
        "guct_uniform_merge_instance_solve_rate": sum(r["guct_uniform_merge_solved_of_seeds"] for r in rows) / (n * args.seeds),
        "n_instances": n,
    }
    print(f"\nSummary (of {n} instances, {args.seeds} seeds each):")
    print(f"  GBFS: {summary['gbfs_instances_solved']}/{n} instances solved")
    print(f"  GUCT: {summary['guct_instance_solve_rate']:.3f} seed-solve rate")
    print(f"  GUCT-Uniform (tree, matches paper's protocol): {summary['guct_uniform_tree_instance_solve_rate']:.3f} seed-solve rate")
    print(f"  GUCT-Uniform (merge, this project's own bonus condition): {summary['guct_uniform_merge_instance_solve_rate']:.3f} seed-solve rate")

    elapsed = time.time() - start
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "elapsed_sec": elapsed, "rows": rows, "summary": summary}, f, indent=2)
    print(f"\ndone in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
