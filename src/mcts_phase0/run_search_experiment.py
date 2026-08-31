"""CLI: run the merge-search H3 experiment. Loads the model, fits (or loads)
a frozen projection from previously-collected Phase 0 data, generates a
held-out Countdown problem set with zero overlap with the projection's
training data, runs both conditions (baseline vs. merge-enabled) at a
matched expansion budget, and pickles raw per-problem graphs + a summary.

Analysis is a separate step (analyze_search_experiment.py), same
collect/analyze split as the rest of this project.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import connect_four, countdown, gsm8k, prosqa
from .model import build_chat_prompt_ids, load_model
from .probe import run_probe
from .projection import calibrate_tau, fit_final_projection
from .search import SearchConfig, is_solved, run_search


def load_training_records(path: str, dataset: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return [r for r in data["records"] if r.dataset == dataset]


def main():
    ap = argparse.ArgumentParser(description="H3 merge-search experiment: does merging help at matched budget?")
    ap.add_argument("--dataset", choices=["countdown", "prosqa", "connect_four", "gsm8k"], default="countdown")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--projection-data", default="results/raw_qwen_countdown_combined/snapshots_combined.pkl")
    ap.add_argument("--projection-layer", default="3/4")
    ap.add_argument("--tau-percentile", type=float, default=40.0, help="used only to print the calibration table")
    ap.add_argument("--tau", type=float, default=0.045)
    ap.add_argument(
        "--n-low", type=int, default=10,
        help="countdown: n_low; prosqa: n_positive; gsm8k: contributes to total sample size (n_low+n_high)",
    )
    ap.add_argument(
        "--n-high", type=int, default=10,
        help="countdown: n_high; prosqa: n_negative; gsm8k: contributes to total sample size (n_low+n_high)",
    )
    ap.add_argument("--countdown-num-numbers", type=int, default=3)
    ap.add_argument("--countdown-number-low", type=int, default=1)
    ap.add_argument("--countdown-number-high", type=int, default=10)
    ap.add_argument("--prosqa-chain-min", type=int, default=3)
    ap.add_argument("--prosqa-chain-max", type=int, default=5)
    ap.add_argument("--connect-four-width", type=int, default=5)
    ap.add_argument("--connect-four-height", type=int, default=4)
    ap.add_argument("--connect-four-k-plies", type=int, default=3)
    ap.add_argument("--connect-four-max-pre-moves", type=int, default=8)
    ap.add_argument("--held-out-seed", type=int, default=99)
    ap.add_argument(
        "--calibrated-from", default=None,
        help="calibration JSON from calibrate.py; restricts the held-out set to in-band instance "
             "ids (same --n-low/--n-high/--countdown-* args must match the calibration run's pool "
             "params so the deterministic draw reproduces the same candidate pool)",
    )
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--max-new-tokens-step", type=int, default=48)
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument(
        "--value-source", choices=["projection", "rollout"], default="projection",
        help="C6 ablation: guide PUCT selection with the frozen projection (default, shares the "
             "merge criterion's signal) or with an actual rollout estimate (score_rollouts), to "
             "separate merge-efficiency from value-guidance quality",
    )
    ap.add_argument("--num-rollouts", type=int, default=4, help="only used when --value-source=rollout")
    ap.add_argument("--rollout-max-new-tokens", type=int, default=80, help="only used when --value-source=rollout")
    ap.add_argument("--output-dir", default="results/search_experiment")
    args = ap.parse_args()

    ds = {"countdown": countdown, "prosqa": prosqa, "connect_four": connect_four, "gsm8k": gsm8k}[args.dataset]

    start = time.time()
    training_records = load_training_records(args.projection_data, args.dataset)
    training_ids = {r.problem_id for r in training_records}
    print(f"loaded {len(training_records)} training records from {len(training_ids)} problems")

    fp = fit_final_projection(training_records, args.projection_layer)
    print(f"fitted projection: layer={fp.layer} alpha={fp.alpha}")

    probe_result = run_probe(training_records, args.projection_layer, n_permutations=0)
    calib = calibrate_tau(training_records, probe_result["oof_predictions"], step_tolerance=0)
    print("tau calibration table (from training data, LOPO out-of-fold):")
    for row in calib:
        print(f"  p{row['percentile']:.0f}: cutoff={row['cutoff']:.4f} "
              f"n_bucket={row['n_bucket']} false_merge_rate={row['false_merge_rate']}")
    print(f"using tau={args.tau}")

    if args.dataset == "countdown":
        held_out = countdown.generate_stratified(
            n_low=args.n_low, n_high=args.n_high, seed=args.held_out_seed,
            num_numbers=args.countdown_num_numbers, number_low=args.countdown_number_low,
            number_high=args.countdown_number_high, exclude_ids=training_ids,
        )
        print(f"generated {len(held_out)} held-out candidates (zero overlap with training)")
    elif args.dataset == "connect_four":
        held_out = connect_four.generate_puzzles(
            n_low=args.n_low, n_high=args.n_high, seed=args.held_out_seed,
            width=args.connect_four_width, height=args.connect_four_height,
            k_plies=args.connect_four_k_plies, max_pre_moves=args.connect_four_max_pre_moves,
            exclude_ids=training_ids,
        )
        print(f"generated {len(held_out)} held-out candidates (zero overlap with training)")
    elif args.dataset == "prosqa":
        # ProsQA has no exclude_ids mechanism (unlike countdown's): its entity
        # names are drawn from a per-call-fresh, large combinatorial naming
        # space, so cross-seed collision with the training set is already
        # astronomically unlikely -- a distinct held-out seed is sufficient.
        held_out = prosqa.generate_dataset(
            n_positive=args.n_low, n_negative=args.n_high, seed=args.held_out_seed,
            chain_len_range=(args.prosqa_chain_min, args.prosqa_chain_max),
        )
        print(f"generated {len(held_out)} held-out candidates (distinct seed from training)")
    else:
        # gsm8k: no exclude_ids mechanism either, but for a stronger reason than
        # prosqa's -- training data comes from gsm8k's official "train" split and
        # held-out comes from its official "test" split, a structural zero-overlap
        # guarantee (not a probabilistic one) that needs no synthetic bookkeeping.
        held_out = gsm8k.generate_sample(n=args.n_low + args.n_high, seed=args.held_out_seed, split="test")
        assert training_ids.isdisjoint({inst.id for inst in held_out}), (
            "gsm8k train/test overlap detected -- wrong split loaded somewhere"
        )
        print(f"generated {len(held_out)} held-out candidates (test split, zero overlap with train)")

    if args.calibrated_from:
        with open(args.calibrated_from) as f:
            report = json.load(f)["report"]
        in_band = {r["id"] for r in report if r["in_band"]}
        before = len(held_out)
        held_out = [inst for inst in held_out if inst.id in in_band]
        print(f"calibration filter: {before} -> {len(held_out)} (in-band only)")
        if not held_out:
            raise SystemExit(
                "no held-out instances survived the calibration filter -- check that "
                "--n-low/--n-high/--countdown-* and --held-out-seed match the calibration run exactly"
            )

    lm = load_model(args.model, device=args.device, dtype=torch.float32)

    results = []
    for inst in tqdm(held_out, desc="problems"):
        prompt_text = ds.build_prompt(inst)
        prompt_ids = build_chat_prompt_ids(lm, prompt_text)
        prompt_len = prompt_ids.shape[0]

        per_problem = {"problem_id": inst.id, "path_count": inst.path_count}
        for condition, merge_enabled in [("baseline", False), ("treatment", True)]:
            config = SearchConfig(
                K=args.K, max_new_tokens_step=args.max_new_tokens_step, temperature=args.temperature,
                c_puct=args.c_puct, projection=fp, merge_enabled=merge_enabled, tau=args.tau,
                value_source=args.value_source, num_rollouts=args.num_rollouts,
                rollout_max_new_tokens=args.rollout_max_new_tokens,
            )
            t0 = time.time()
            graph = run_search(lm, prompt_ids, prompt_len, inst, ds.parse_and_verify, config, args.budget)
            elapsed = time.time() - t0
            n_step_nodes = sum(1 for n in graph.nodes.values() if n.kind in ("step", "answer"))
            per_problem[condition] = {
                "solved": is_solved(graph),
                "unique_nodes": n_step_nodes,
                "elapsed_sec": elapsed,
                "graph": graph,
            }
        results.append(per_problem)

    elapsed_total = time.time() - start
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"search_results_seed{args.held_out_seed}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "args": vars(args), "elapsed_sec": elapsed_total, "tau_calibration": calib}, f)

    n_baseline_solved = sum(r["baseline"]["solved"] for r in results)
    n_treatment_solved = sum(r["treatment"]["solved"] for r in results)
    print(f"baseline solved: {n_baseline_solved}/{len(results)}")
    print(f"treatment solved: {n_treatment_solved}/{len(results)}")
    print(f"total time: {elapsed_total:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
