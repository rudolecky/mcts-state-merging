"""CLI: {vanilla PUCT, DeepSearch-style} x {tree, merge} on ProsQA, reusing
the exact projection/tau/model already deployed there for this project's
own H3 trust-bar result (see entropy_check.py and
results/search_experiment_qwen7b_prosqa/'s own recorded args). Mirrors
run_search_experiment.py's structure, restricted to ProsQA and 4 arms.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import prosqa
from .model import build_chat_prompt_ids, load_model
from .probe import run_probe
from .projection import calibrate_tau, fit_final_projection
from .search import SearchConfig
from .search import is_solved as puct_is_solved
from .search import run_search as run_puct_search
from .search_deepsearch import DeepSearchConfig
from .search_deepsearch import is_solved as deepsearch_is_solved
from .search_deepsearch import run_search_deepsearch


def main():
    import argparse

    ap = argparse.ArgumentParser(description="4-arm merge-vs-tree x vanilla-PUCT-vs-DeepSearch comparison on ProsQA")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--projection-data", default="results/raw_qwen7b_prosqa/snapshots_seed80.pkl")
    ap.add_argument("--projection-layer", default="mid")
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--n-low", type=int, default=30, help="prosqa n_positive")
    ap.add_argument("--n-high", type=int, default=12, help="prosqa n_negative")
    ap.add_argument("--prosqa-chain-min", type=int, default=3)
    ap.add_argument("--prosqa-chain-max", type=int, default=5)
    ap.add_argument("--held-out-seed", type=int, default=90)
    ap.add_argument("--calibrated-from", default="results/calibration_qwen7b_prosqa_heldout/calibration_seed90.json")
    ap.add_argument("--n-problems", type=int, default=None, help="cap the (already calibration-filtered) held-out set for a cheaper pilot run")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max-new-tokens-step", type=int, default=40)
    ap.add_argument("--budget", type=int, default=15)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--iterations-per-backup-round", type=int, default=4)
    ap.add_argument("--output-dir", default="results/algorithm_comparison_experiment")
    args = ap.parse_args()

    start = time.time()
    with open(args.projection_data, "rb") as f:
        training_records = [r for r in pickle.load(f)["records"] if r.dataset == "prosqa"]
    training_ids = {r.problem_id for r in training_records}
    print(f"loaded {len(training_records)} training records from {len(training_ids)} problems")

    fp = fit_final_projection(training_records, args.projection_layer)
    print(f"fitted projection: layer={fp.layer} alpha={fp.alpha}")
    probe_result = run_probe(training_records, args.projection_layer, n_permutations=0)
    calib = calibrate_tau(training_records, probe_result["oof_predictions"], step_tolerance=0)
    print(f"using tau={args.tau} (calibration table: {[(r['percentile'], r['cutoff']) for r in calib]})")

    held_out = prosqa.generate_dataset(
        n_positive=args.n_low, n_negative=args.n_high, seed=args.held_out_seed,
        chain_len_range=(args.prosqa_chain_min, args.prosqa_chain_max),
    )
    print(f"generated {len(held_out)} held-out candidates")

    with open(args.calibrated_from) as f:
        report = json.load(f)["report"]
    in_band = {r["id"] for r in report if r["in_band"]}
    before = len(held_out)
    held_out = [inst for inst in held_out if inst.id in in_band]
    print(f"calibration filter: {before} -> {len(held_out)} (in-band only)")
    if args.n_problems is not None:
        held_out = held_out[: args.n_problems]
        print(f"pilot cap: using {len(held_out)} problems")

    lm = load_model(args.model, device=args.device, dtype=torch.float32)

    results = []
    for inst in tqdm(held_out, desc="problems"):
        prompt_text = prosqa.build_prompt(inst)
        prompt_ids = build_chat_prompt_ids(lm, prompt_text)
        prompt_len = prompt_ids.shape[0]

        per_problem = {"problem_id": inst.id}
        for algo in ("puct", "deepsearch"):
            for condition, merge_enabled in [("baseline", False), ("treatment", True)]:
                t0 = time.time()
                if algo == "puct":
                    config = SearchConfig(K=args.K, max_new_tokens_step=args.max_new_tokens_step,
                                           temperature=args.temperature, c_puct=args.c_puct,
                                           projection=fp, merge_enabled=merge_enabled, tau=args.tau)
                    graph = run_puct_search(lm, prompt_ids, prompt_len, inst, prosqa.parse_and_verify, config, args.budget)
                    solved = puct_is_solved(graph)
                else:
                    config = DeepSearchConfig(K=args.K, max_new_tokens_step=args.max_new_tokens_step,
                                               temperature=args.temperature, projection=fp,
                                               merge_enabled=merge_enabled, tau=args.tau,
                                               iterations_per_backup_round=args.iterations_per_backup_round)
                    graph = run_search_deepsearch(lm, prompt_ids, prompt_len, inst, prosqa.parse_and_verify, config, args.budget)
                    solved = deepsearch_is_solved(graph)
                elapsed = time.time() - t0
                n_step_nodes = sum(1 for n in graph.nodes.values() if n.kind in ("step", "answer"))
                per_problem[f"{algo}_{condition}"] = {"solved": solved, "unique_nodes": n_step_nodes, "elapsed_sec": elapsed}
        results.append(per_problem)
        print(f"  {inst.id}: " + " ".join(f"{k}={v['solved']}" for k, v in per_problem.items() if isinstance(v, dict)))

    elapsed_total = time.time() - start
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"comparison_results_seed{args.held_out_seed}.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "elapsed_sec": elapsed_total, "results": results}, f, indent=2)

    for arm in ["puct_baseline", "puct_treatment", "deepsearch_baseline", "deepsearch_treatment"]:
        n_solved = sum(r[arm]["solved"] for r in results)
        print(f"{arm}: {n_solved}/{len(results)}")
    print(f"total time: {elapsed_total:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
