"""CLI: run the model over a dataset, capturing hidden states at reasoning-
step boundaries and rollout-scored empirical value V-hat for each snapshot.
Writes raw SnapshotRecord data to disk; analyze.py consumes it separately so
the (expensive, MPS-bound) collection step never has to be re-run just to
try a different statistical slicing.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import connect_four, countdown, gsm8k, gsm8k_native, prosqa
from .datasets.common import split_steps
from .geometry import SnapshotRecord
from .model import (
    build_chat_prompt_ids,
    find_position_based_boundaries,
    find_step_boundaries,
    generate_traces,
    hidden_states_for_sequence,
    load_model,
    resolve_layers,
    score_rollouts,
)


def _select_snapshot_indices(n_boundaries: int, cap: int, selection: str = "even") -> list[int]:
    """Which indices (0-based, into the full boundary list) to snapshot,
    capped at `cap`. Returns all indices if n_boundaries <= cap.

    "even": evenly spaced across [0, n_boundaries - 1] -- the default, used
    by every dataset with bounded/uniform solution lengths.

    "first": the first `cap` boundaries, regardless of how many more exist.
    For a dataset whose solution length varies a lot (GSM8K: 1-2 steps to
    17+), "even" spacing means a given step_idx represents a wildly
    different fraction of the solution depending on total length, diluting
    any within-problem signal when snapshots are pooled across problems.
    "first" fixes what a snapshot means (always "right after step k",
    independent of how the solution eventually ends) at the cost of never
    seeing later reasoning in long traces.
    """
    if n_boundaries <= cap:
        return list(range(n_boundaries))
    if selection == "first":
        return list(range(cap))
    # evenly spaced across [0, n_boundaries - 1], cap points, no duplicates
    step = (n_boundaries - 1) / (cap - 1) if cap > 1 else 0
    return sorted({round(i * step) for i in range(cap)})


def collect_for_dataset(
    lm,
    dataset_name: str,
    instances: list,
    build_prompt_fn,
    verifier_fn,
    layers: dict[str, int],
    num_traces: int,
    max_snapshots_per_trace: int,
    num_rollouts: int,
    max_new_tokens_trace: int,
    max_new_tokens_rollout: int,
    temperature: float,
    ground_truth_key_fn=None,
    snapshot_selection: str = "even",
    split_fn=split_steps,
) -> list[SnapshotRecord]:
    records: list[SnapshotRecord] = []
    for inst in tqdm(instances, desc=f"{dataset_name} problems"):
        prompt_text = build_prompt_fn(inst)
        prompt_ids = build_chat_prompt_ids(lm, prompt_text)
        traces = generate_traces(lm, prompt_ids, num_traces, max_new_tokens_trace, temperature)

        for trace_idx, full_ids in enumerate(traces):
            boundary_kind = "step"
            boundaries = find_step_boundaries(lm.tokenizer, prompt_ids.shape[0], full_ids)
            if not boundaries:
                # model doesn't emit "Step N:" lines at all (e.g. free-form
                # <think>-style reasoning) -- fall back to relative-position
                # snapshots so the pipeline still has something to compare.
                boundary_kind = "position"
                boundaries = find_position_based_boundaries(
                    prompt_ids.shape[0], full_ids, max_snapshots_per_trace
                )
            if not boundaries:
                continue
            hidden = hidden_states_for_sequence(lm, full_ids)
            chosen = _select_snapshot_indices(len(boundaries), max_snapshots_per_trace, snapshot_selection)

            for boundary_pos in chosen:
                token_idx = boundaries[boundary_pos]
                step_idx = boundary_pos + 1  # 1-based depth position (step number or position bucket)
                hidden_vecs = {
                    layer_name: hidden[layer_idx][0, token_idx, :].float().cpu().numpy()
                    for layer_name, layer_idx in layers.items()
                }
                prefix_ids = full_ids[: token_idx + 1]
                v_hat = score_rollouts(
                    lm, inst, verifier_fn, prefix_ids, prompt_ids.shape[0],
                    num_rollouts, max_new_tokens_rollout, temperature, split_fn=split_fn,
                )
                ground_truth_key = None
                step_bodies_so_far = None
                if ground_truth_key_fn is not None:
                    prefix_text = lm.tokenizer.decode(
                        full_ids[prompt_ids.shape[0] : token_idx + 1].tolist(), skip_special_tokens=True
                    )
                    step_bodies_so_far, _ = split_steps(prefix_text)
                    ground_truth_key = ground_truth_key_fn(inst, step_bodies_so_far)
                records.append(
                    SnapshotRecord(
                        dataset=dataset_name,
                        problem_id=inst.id,
                        trace_idx=trace_idx,
                        step_idx=step_idx,
                        v_hat=v_hat,
                        hidden=hidden_vecs,
                        boundary_kind=boundary_kind,
                        ground_truth_key=ground_truth_key,
                        step_bodies=tuple(step_bodies_so_far) if step_bodies_so_far is not None else None,
                    )
                )
    return records


def main():
    ap = argparse.ArgumentParser(description="Phase 0 collection: hidden states + rollout-scored V-hat")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-countdown-low", type=int, default=5)
    ap.add_argument("--n-countdown-high", type=int, default=5)
    ap.add_argument("--n-prosqa-positive", type=int, default=5)
    ap.add_argument("--n-prosqa-negative", type=int, default=5)
    ap.add_argument("--num-traces", type=int, default=4, help="K: sampled traces per problem")
    ap.add_argument("--max-snapshots-per-trace", type=int, default=3)
    ap.add_argument("--num-rollouts", type=int, default=8, help="N: rollouts per snapshot")
    ap.add_argument("--max-new-tokens-trace", type=int, default=200)
    ap.add_argument("--max-new-tokens-rollout", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/raw")
    ap.add_argument(
        "--calibrated-from",
        default=None,
        help="calibration JSON from calibrate.py; restricts collection to in-band instance ids",
    )
    ap.add_argument("--countdown-num-numbers", type=int, default=3)
    ap.add_argument("--countdown-number-low", type=int, default=1)
    ap.add_argument("--countdown-number-high", type=int, default=10)
    ap.add_argument("--prosqa-chain-min", type=int, default=2)
    ap.add_argument("--prosqa-chain-max", type=int, default=3)
    ap.add_argument("--n-connect-four-low", type=int, default=0,
                     help="0 (default) skips connect_four collection entirely")
    ap.add_argument("--n-connect-four-high", type=int, default=0)
    ap.add_argument("--connect-four-width", type=int, default=5)
    ap.add_argument("--connect-four-height", type=int, default=4)
    ap.add_argument("--connect-four-k-plies", type=int, default=3)
    ap.add_argument("--connect-four-max-pre-moves", type=int, default=8)
    ap.add_argument("--n-gsm8k-train", type=int, default=0,
                     help="0 (default) skips gsm8k collection entirely")
    ap.add_argument("--gsm8k-max-new-tokens-trace", type=int, default=400,
                     help="gsm8k's natural-language solutions run longer than the other datasets'; "
                          "overrides --max-new-tokens-trace for gsm8k only")
    ap.add_argument("--gsm8k-max-new-tokens-rollout", type=int, default=200)
    ap.add_argument("--n-gsm8k-native", type=int, default=0,
                     help="0 (default) skips gsm8k_native collection entirely -- a reasoning-RL "
                          "model probed at its own natural boundaries, not forced Step N: lines")
    ap.add_argument("--gsm8k-native-max-new-tokens-trace", type=int, default=900)
    ap.add_argument("--gsm8k-native-max-new-tokens-rollout", type=int, default=400)
    args = ap.parse_args()

    start = time.time()
    lm = load_model(args.model, device=args.device, dtype=torch.float32)
    layers = resolve_layers(lm.num_hidden_layers)
    print(f"resolved layers: {layers}")

    cd_instances = countdown.generate_stratified(
        n_low=args.n_countdown_low, n_high=args.n_countdown_high, seed=args.seed,
        num_numbers=args.countdown_num_numbers, number_low=args.countdown_number_low,
        number_high=args.countdown_number_high,
    )
    pq_instances = prosqa.generate_dataset(
        n_positive=args.n_prosqa_positive, n_negative=args.n_prosqa_negative, seed=args.seed,
        chain_len_range=(args.prosqa_chain_min, args.prosqa_chain_max),
    )
    cf_instances = []
    if args.n_connect_four_low > 0 or args.n_connect_four_high > 0:
        cf_instances = connect_four.generate_puzzles(
            n_low=args.n_connect_four_low, n_high=args.n_connect_four_high, seed=args.seed,
            width=args.connect_four_width, height=args.connect_four_height,
            k_plies=args.connect_four_k_plies, max_pre_moves=args.connect_four_max_pre_moves,
        )
    gs_instances = []
    if args.n_gsm8k_train > 0:
        gs_instances = gsm8k.generate_sample(n=args.n_gsm8k_train, seed=args.seed, split="train")
    gsn_instances = []
    if args.n_gsm8k_native > 0:
        gsn_instances = gsm8k.generate_sample(n=args.n_gsm8k_native, seed=args.seed, split="train")

    if args.calibrated_from:
        with open(args.calibrated_from) as f:
            report = json.load(f)["report"]
        in_band = {r["id"] for r in report if r["in_band"]}
        before = (len(cd_instances), len(pq_instances), len(cf_instances), len(gs_instances), len(gsn_instances))
        cd_instances = [i for i in cd_instances if i.id in in_band]
        pq_instances = [i for i in pq_instances if i.id in in_band]
        cf_instances = [i for i in cf_instances if i.id in in_band]
        gs_instances = [i for i in gs_instances if i.id in in_band]
        gsn_instances = [i for i in gsn_instances if i.id in in_band]
        print(f"calibration filter: countdown {before[0]}->{len(cd_instances)}, "
              f"prosqa {before[1]}->{len(pq_instances)}, connect_four {before[2]}->{len(cf_instances)}, "
              f"gsm8k {before[3]}->{len(gs_instances)}, gsm8k_native {before[4]}->{len(gsn_instances)} "
              "(in-band only)")
        if not cd_instances and not pq_instances and not cf_instances and not gs_instances and not gsn_instances:
            raise SystemExit(
                "no instances survived the calibration filter -- check that the "
                "generation parameters and seed match the calibration run exactly"
            )

    all_records: list[SnapshotRecord] = []
    all_records += collect_for_dataset(
        lm, "countdown", cd_instances, countdown.build_prompt, countdown.parse_and_verify,
        layers, args.num_traces, args.max_snapshots_per_trace, args.num_rollouts,
        args.max_new_tokens_trace, args.max_new_tokens_rollout, args.temperature,
    )
    all_records += collect_for_dataset(
        lm, "prosqa", pq_instances, prosqa.build_prompt, prosqa.parse_and_verify,
        layers, args.num_traces, args.max_snapshots_per_trace, args.num_rollouts,
        args.max_new_tokens_trace, args.max_new_tokens_rollout, args.temperature,
        ground_truth_key_fn=prosqa.canonical_state_at,
    )
    if cf_instances:
        all_records += collect_for_dataset(
            lm, "connect_four", cf_instances, connect_four.build_prompt, connect_four.parse_and_verify,
            layers, args.num_traces, args.max_snapshots_per_trace, args.num_rollouts,
            args.max_new_tokens_trace, args.max_new_tokens_rollout, args.temperature,
            ground_truth_key_fn=connect_four.canonical_state_at,
        )
    if gs_instances:
        all_records += collect_for_dataset(
            lm, "gsm8k", gs_instances, gsm8k.build_prompt, gsm8k.parse_and_verify,
            layers, args.num_traces, args.max_snapshots_per_trace, args.num_rollouts,
            args.gsm8k_max_new_tokens_trace, args.gsm8k_max_new_tokens_rollout, args.temperature,
            snapshot_selection="first",  # fixed early steps, not evenly-spaced -- see
            # _select_snapshot_indices' docstring: gsm8k's solution length varies far more
            # (1-2 to 17+ steps) than the other datasets, diluting within-problem signal
            # when "even" spacing makes step_idx mean a different fraction of the solution
            # depending on total length.
        )
    if gsn_instances:
        all_records += collect_for_dataset(
            lm, "gsm8k_native", gsn_instances, gsm8k_native.build_prompt, gsm8k_native.parse_and_verify,
            layers, args.num_traces, args.max_snapshots_per_trace, args.num_rollouts,
            args.gsm8k_native_max_new_tokens_trace, args.gsm8k_native_max_new_tokens_rollout, args.temperature,
            split_fn=gsm8k_native.native_split,
            # "even" spacing (the default) is fine here: find_step_boundaries will always find
            # zero "Step N:" lines for this prompt, so every trace uses the position-based
            # fallback regardless of snapshot_selection.
        )

    elapsed = time.time() - start
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"snapshots_seed{args.seed}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"records": all_records, "layers": layers, "args": vars(args), "elapsed_sec": elapsed}, f)

    print(f"collected {len(all_records)} snapshot records in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
