"""CLI: does merging help PUCT search once the prior is a REAL trained
policy/value network -- not a uniform-1/K stand-in (the LLM H3 arc), not
random-rollout UCB1 (the classical arc)? Loads a checkpoint from
`run_selfplay_train.py` and runs the same merge-vs-tree comparison every
other domain in this project has, on the same k_plies-controlled puzzle
set as `classical_mcts.py`'s own Connect Four work.

Statistical design differs from the classical arc on purpose: evaluation
with a fixed, trained network and no exploration noise (`dirichlet_alpha=
None`) is fully deterministic -- no random rollout, no root noise -- so
there is nothing to average over with repeats. Solve/not-solve is a
single binary outcome per puzzle per condition, the same paired-McNemar
design the LLM H3 arc used (Countdown, ProsQA, Connect-Four-LLM), not the
classical arc's 50-repeats-averaged solve rate (which existed
specifically to average over random-rollout variance this module never
has).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from ..datasets import connect_four
from .network import ConnectFourNet
from .network import evaluate as net_evaluate
from .puct import PUCTConfig, is_solved, run_search


def _bootstrap_ci_mean_diff(diffs: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = [rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def main():
    ap = argparse.ArgumentParser(description="Does merging help PUCT search with a real trained policy/value network?")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--channels", type=int, default=48)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--n-puzzles", type=int, default=40)
    ap.add_argument("--k-plies", type=int, default=3)
    ap.add_argument("--max-pre-moves", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[10, 25, 50, 100])
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-dir", default="results/alphazero_connect_four_merge_experiment")
    args = ap.parse_args()

    net = ConnectFourNet(channels=args.channels, num_blocks=args.num_blocks).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["net"])
    net.eval()
    print(f"loaded checkpoint from iteration {ckpt['iteration']}")

    def evaluate_fn(board, to_move):
        return net_evaluate(net, board, to_move, args.device)

    puzzles = connect_four.generate_puzzles(
        n_low=args.n_puzzles, n_high=0, seed=args.seed,
        width=5, height=4, k_plies=args.k_plies, max_pre_moves=args.max_pre_moves,
    )
    print(f"generated {len(puzzles)} puzzles")

    results = {}
    for budget in args.budgets:
        rows = []
        for inst in puzzles:
            base_cfg = PUCTConfig(merge_enabled=False, c_puct=args.c_puct, dirichlet_alpha=None)
            treat_cfg = PUCTConfig(merge_enabled=True, c_puct=args.c_puct, dirichlet_alpha=None)
            base_graph = run_search(inst.pre_moves, inst.to_move, 5, base_cfg, budget, evaluate_fn)
            treat_graph = run_search(inst.pre_moves, inst.to_move, 5, treat_cfg, budget, evaluate_fn)
            rows.append({
                "id": inst.id,
                "baseline_solved": is_solved(base_graph, hero=inst.to_move),
                "treatment_solved": is_solved(treat_graph, hero=inst.to_move),
                "baseline_nodes": len(base_graph.nodes),
                "treatment_nodes": len(treat_graph.nodes),
            })

        n = len(rows)
        base_solved = sum(r["baseline_solved"] for r in rows)
        treat_solved = sum(r["treatment_solved"] for r in rows)
        both = sum(r["baseline_solved"] and r["treatment_solved"] for r in rows)
        base_only = sum(r["baseline_solved"] and not r["treatment_solved"] for r in rows)
        treat_only = sum(r["treatment_solved"] and not r["baseline_solved"] for r in rows)
        neither = n - both - base_only - treat_only
        discordant = base_only + treat_only
        mcnemar_p = binomtest(min(base_only, treat_only), discordant, 0.5).pvalue if discordant > 0 else float("nan")

        diffs = np.array([int(r["treatment_solved"]) - int(r["baseline_solved"]) for r in rows], dtype=float)
        ci_low, ci_high = _bootstrap_ci_mean_diff(diffs, seed=args.seed)
        base_nodes_mean = float(np.mean([r["baseline_nodes"] for r in rows]))
        treat_nodes_mean = float(np.mean([r["treatment_nodes"] for r in rows]))

        print(f"budget={budget}: baseline={base_solved}/{n} treatment={treat_solved}/{n} "
              f"contingency(both={both} base_only={base_only} treat_only={treat_only} neither={neither}) "
              f"McNemar_p={mcnemar_p:.4f} CI=[{ci_low:.3f},{ci_high:.3f}] "
              f"nodes={base_nodes_mean:.1f}->{treat_nodes_mean:.1f}")

        results[budget] = {
            "rows": rows, "baseline_solved": base_solved, "treatment_solved": treat_solved, "n": n,
            "both": both, "base_only": base_only, "treat_only": treat_only, "neither": neither,
            "mcnemar_p": float(mcnemar_p), "ci_low": ci_low, "ci_high": ci_high,
            "baseline_nodes_mean": base_nodes_mean, "treatment_nodes_mean": treat_nodes_mean,
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "checkpoint_iteration": ckpt["iteration"], "results": results}, f, indent=2)
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
