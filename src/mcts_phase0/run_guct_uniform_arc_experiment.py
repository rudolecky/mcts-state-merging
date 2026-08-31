"""ARC-AGI program synthesis under GUCT-Uniform (Full Bellman min-backup +
LCB1-Uniform, real _arc_heuristic reward) -- tests whether Blocksworld's
own "still a huge persistent gap under GUCT-Uniform" result
(`BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`) is unique to that domain, or
shared by ARC-AGI, this project's other established large-merge-benefit
domain. Reuses `run_arc_program_experiment.py`'s own task-curation
(`find_reachable_tasks`) and established scale (n_tasks=12,
budgets=[500,2000,5000]) for direct comparability with the blind-UCB1
ARC findings.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

import mcts_phase0.guct_uniform_arc as m
from .datasets import arc_engine as ae
from .run_arc_program_experiment import find_reachable_tasks


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


def main():
    import argparse

    ap = argparse.ArgumentParser(description="ARC-AGI under GUCT-Uniform: tree vs merge, real tasks")
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--max-grid-dim", type=int, default=15)
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 2000, 5000])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--output-dir", default="results/guct_uniform_arc_experiment")
    args = ap.parse_args()

    start = time.time()
    reachable = find_reachable_tasks(args.min_depth, args.max_depth, args.max_grid_dim)
    rng = random.Random(args.seed)
    rng.shuffle(reachable)
    selected = reachable[: args.n_tasks]
    print(f"selected {len(selected)} tasks: {[t for t, _ in selected]}", flush=True)

    tasks = []
    for task_id, depth in selected:
        raw = json.loads(Path(f"data/arc_agi/tasks/{task_id}.json").read_text())
        tasks.append(ae.load_task(raw, task_id))

    results = []
    for budget in args.budgets:
        t0 = time.time()
        rows = []
        for i, task in enumerate(tasks):
            task_t0 = time.time()
            tree_solved = merge_solved = 0
            tree_nodes = merge_nodes = 0
            for r in range(args.repeats):
                rng1 = random.Random(i * 100_003 + r)
                g = m.run_search(task.train_inputs, task.train_outputs, m.GUCTUniformConfig(merge_enabled=False), budget, rng1)
                tree_solved += int(m.is_solved(g))
                tree_nodes += len(g.nodes)
                rng2 = random.Random(i * 100_003 + r)
                g2 = m.run_search(task.train_inputs, task.train_outputs, m.GUCTUniformConfig(merge_enabled=True), budget, rng2)
                merge_solved += int(m.is_solved(g2))
                merge_nodes += len(g2.nodes)
            rows.append({"task_id": task.task_id, "tree": tree_solved / args.repeats, "merge": merge_solved / args.repeats,
                         "tree_nodes": tree_nodes / args.repeats, "merge_nodes": merge_nodes / args.repeats})
            print(f"  budget={budget} task={task.task_id} ({i+1}/{len(tasks)}): "
                  f"tree={tree_solved}/{args.repeats} merge={merge_solved}/{args.repeats} (t={time.time()-task_t0:.1f}s)", flush=True)
        tm = np.mean([r["tree"] for r in rows])
        mm = np.mean([r["merge"] for r in rows])
        dm, ci, p = _stats(rows, "tree", "merge")
        tn = np.mean([r["tree_nodes"] for r in rows])
        mn = np.mean([r["merge_nodes"] for r in rows])
        elapsed = time.time() - t0
        print(f"budget={budget}: tree={tm:.3f} merge={mm:.3f} diff={dm:.3f} CI=[{ci[0]:.3f},{ci[1]:.3f}] "
              f"p={p:.4f} nodes tree={tn:.1f} merge={mn:.1f} (t={elapsed:.1f}s)", flush=True)
        results.append({"budget": budget, "tree": tm, "merge": mm, "diff": dm, "ci": ci, "p": p,
                         "tree_nodes": tn, "merge_nodes": mn, "rows": rows})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "task_ids": [t for t, _ in selected], "results": results,
                   "elapsed_sec": time.time() - start}, f, indent=2,
                  default=lambda o: list(o) if isinstance(o, tuple) else o)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
