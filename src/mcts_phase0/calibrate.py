"""CLI: difficulty pre-pass. Estimates each candidate instance's pass rate
and keeps only those in a usable middle band.

Why this exists: if per-problem accuracy saturates near 0% or 100%, the
within-problem V-hat variance collapses and the distance-vs-value regression
is starved of signal regardless of whether the pipeline mechanics are
correct. That would read as "H1 fails" when it is really a floor/ceiling
artifact of a task/model difficulty mismatch. Calibrating first keeps the
Phase 0 result interpretable.
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import connect_four, countdown, gsm8k, gsm8k_native, prosqa
from .datasets.common import split_steps
from .model import build_chat_prompt_ids, generate_traces, load_model


def estimate_pass_rate(
    lm, instance, build_prompt_fn, verifier_fn, num_samples: int, max_new_tokens: int, temperature: float,
    split_fn=split_steps,
) -> tuple[float, dict[str, int]]:
    prompt_ids = build_chat_prompt_ids(lm, build_prompt_fn(instance))
    traces = generate_traces(lm, prompt_ids, num_samples, max_new_tokens, temperature)
    n_ok = 0
    reasons: dict[str, int] = {}
    for full_ids in traces:
        text = lm.tokenizer.decode(full_ids[prompt_ids.shape[0] :].tolist(), skip_special_tokens=True)
        step_bodies, answer_body = split_fn(text)
        ok, info = verifier_fn(instance, step_bodies, answer_body)
        n_ok += bool(ok)
        reason = str(info.get("reason", "?"))[:60]
        reasons[reason] = reasons.get(reason, 0) + 1
    return n_ok / num_samples, reasons


def calibrate_pool(
    lm, dataset_name, instances, build_prompt_fn, verifier_fn,
    num_samples, max_new_tokens, temperature, band_low, band_high,
    split_fn=split_steps,
) -> tuple[list, list[dict]]:
    kept, report = [], []
    for inst in tqdm(instances, desc=f"calibrating {dataset_name}"):
        rate, reasons = estimate_pass_rate(
            lm, inst, build_prompt_fn, verifier_fn, num_samples, max_new_tokens, temperature, split_fn=split_fn,
        )
        in_band = band_low <= rate <= band_high
        if in_band:
            kept.append(inst)
        report.append({
            "dataset": dataset_name,
            "id": inst.id,
            "pass_rate": rate,
            "in_band": in_band,
            "path_count": getattr(inst, "path_count", None),
            "top_reasons": sorted(reasons.items(), key=lambda kv: -kv[1])[:3],
        })
    return kept, report


def main():
    ap = argparse.ArgumentParser(description="Phase 0 difficulty calibration pre-pass")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--pool-countdown-low", type=int, default=10)
    ap.add_argument("--pool-countdown-high", type=int, default=10)
    ap.add_argument("--pool-prosqa-positive", type=int, default=10)
    ap.add_argument("--pool-prosqa-negative", type=int, default=10)
    ap.add_argument("--countdown-num-numbers", type=int, default=3)
    ap.add_argument("--countdown-number-low", type=int, default=1)
    ap.add_argument("--countdown-number-high", type=int, default=10)
    ap.add_argument("--prosqa-chain-min", type=int, default=2)
    ap.add_argument("--prosqa-chain-max", type=int, default=4)
    ap.add_argument("--pool-connect-four-low", type=int, default=0,
                     help="0 (default) skips connect_four calibration entirely")
    ap.add_argument("--pool-connect-four-high", type=int, default=0)
    ap.add_argument("--connect-four-width", type=int, default=5)
    ap.add_argument("--connect-four-height", type=int, default=4)
    ap.add_argument("--connect-four-k-plies", type=int, default=3)
    ap.add_argument("--connect-four-max-pre-moves", type=int, default=8)
    ap.add_argument("--connect-four-test-scratch-variant", action="store_true",
                     help="also calibrate the encouraged-scratch-board prompt (Stage 1 found it "
                          "worse; off by default so later runs don't re-spend time confirming that)")
    ap.add_argument("--pool-gsm8k", type=int, default=0,
                     help="0 (default) skips gsm8k calibration entirely; single count, no low/high "
                          "split -- gsm8k has no a-priori difficulty proxy to stratify on")
    ap.add_argument("--gsm8k-max-new-tokens", type=int, default=400,
                     help="gsm8k's natural-language solutions run longer than the other datasets' "
                          "terse grammars; overrides --max-new-tokens for gsm8k only")
    ap.add_argument("--pool-gsm8k-native", type=int, default=0,
                     help="0 (default) skips gsm8k_native calibration entirely -- a reasoning-RL "
                          "model probed at its own natural boundaries, not forced Step N: lines")
    ap.add_argument("--gsm8k-native-max-new-tokens", type=int, default=900,
                     help="reasoning-RL models (DeepSeek-R1-Distill) think in long free-form "
                          "<think> blocks, longer even than plain gsm8k's own budget")
    ap.add_argument("--num-samples", type=int, default=6, help="samples per instance for the rate estimate")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--band-low", type=float, default=0.15)
    ap.add_argument("--band-high", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/calibration")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    start = time.time()
    lm = load_model(args.model, device=args.device, dtype=dtype)

    cd_pool = countdown.generate_stratified(
        n_low=args.pool_countdown_low, n_high=args.pool_countdown_high, seed=args.seed,
        num_numbers=args.countdown_num_numbers, number_low=args.countdown_number_low,
        number_high=args.countdown_number_high,
    )
    pq_pool = prosqa.generate_dataset(
        n_positive=args.pool_prosqa_positive, n_negative=args.pool_prosqa_negative, seed=args.seed,
        chain_len_range=(args.prosqa_chain_min, args.prosqa_chain_max),
    )

    cd_kept, cd_report = calibrate_pool(
        lm, "countdown", cd_pool, countdown.build_prompt, countdown.parse_and_verify,
        args.num_samples, args.max_new_tokens, args.temperature, args.band_low, args.band_high,
    )
    pq_kept, pq_report = calibrate_pool(
        lm, "prosqa", pq_pool, prosqa.build_prompt, prosqa.parse_and_verify,
        args.num_samples, args.max_new_tokens, args.temperature, args.band_low, args.band_high,
    )

    cf_report = []
    if args.pool_connect_four_low > 0 or args.pool_connect_four_high > 0:
        cf_pool = connect_four.generate_puzzles(
            n_low=args.pool_connect_four_low, n_high=args.pool_connect_four_high, seed=args.seed,
            width=args.connect_four_width, height=args.connect_four_height,
            k_plies=args.connect_four_k_plies, max_pre_moves=args.connect_four_max_pre_moves,
        )
        _, cf_strict_report = calibrate_pool(
            lm, "connect_four_strict", cf_pool, connect_four.build_prompt, connect_four.parse_and_verify,
            args.num_samples, args.max_new_tokens, args.temperature, args.band_low, args.band_high,
        )
        cf_report = cf_strict_report
        if args.connect_four_test_scratch_variant:
            # Same puzzle pool through both prompt variants -- a clean A/B, not
            # a confound between "harder puzzles" and "harder prompt."
            scratch_prompt = functools.partial(connect_four.build_prompt, encourage_scratch_board=True)
            _, cf_scratch_report = calibrate_pool(
                lm, "connect_four_scratch", cf_pool, scratch_prompt, connect_four.parse_and_verify,
                args.num_samples, args.max_new_tokens, args.temperature, args.band_low, args.band_high,
            )
            cf_report = cf_strict_report + cf_scratch_report

    gs_report = []
    if args.pool_gsm8k > 0:
        gs_pool = gsm8k.generate_sample(n=args.pool_gsm8k, seed=args.seed, split="train")
        _, gs_report = calibrate_pool(
            lm, "gsm8k", gs_pool, gsm8k.build_prompt, gsm8k.parse_and_verify,
            args.num_samples, args.gsm8k_max_new_tokens, args.temperature, args.band_low, args.band_high,
        )

    gsn_report = []
    if args.pool_gsm8k_native > 0:
        gsn_pool = gsm8k.generate_sample(n=args.pool_gsm8k_native, seed=args.seed, split="train")
        _, gsn_report = calibrate_pool(
            lm, "gsm8k_native", gsn_pool, gsm8k_native.build_prompt, gsm8k_native.parse_and_verify,
            args.num_samples, args.gsm8k_native_max_new_tokens, args.temperature, args.band_low, args.band_high,
            split_fn=gsm8k_native.native_split,
        )

    elapsed = time.time() - start
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = cd_report + pq_report + cf_report + gs_report + gsn_report
    with open(out_dir / f"calibration_seed{args.seed}.json", "w") as f:
        json.dump({"args": vars(args), "elapsed_sec": elapsed, "report": report}, f, indent=2)

    def _summarize(name, rep):
        rates = [r["pass_rate"] for r in rep]
        in_band = sum(r["in_band"] for r in rep)
        zeros = sum(1 for x in rates if x == 0.0)
        ones = sum(1 for x in rates if x == 1.0)
        mean = sum(rates) / len(rates) if rates else float("nan")
        print(f"{name}: n={len(rep)} mean_pass={mean:.3f} in_band={in_band} at_0={zeros} at_1={ones}")

    _summarize("countdown", cd_report)
    _summarize("prosqa", pq_report)
    if cf_report:
        _summarize("connect_four_strict", [r for r in cf_report if r["dataset"] == "connect_four_strict"])
        if args.connect_four_test_scratch_variant:
            _summarize("connect_four_scratch", [r for r in cf_report if r["dataset"] == "connect_four_scratch"])
    if gs_report:
        _summarize("gsm8k", gs_report)
    if gsn_report:
        _summarize("gsm8k_native", gsn_report)
    print(f"calibration done in {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
