"""ProsQA's first real ground-truth transposition validation -- mirrors
Connect Four's own Stage 0-3 methodology (`results/CONNECT_FOUR_FINDINGS.md`,
never a checked-in script there, only ad hoc REPL runs) but as a real,
reusable script, and extended one step further: for every false merge the
*already-deployed* projection/tau would make, log both snapshots' full
trajectories and check whether false merges share a path-shaped signature --
the concrete evidence a path-conditioned merge criterion would need before
being worth designing at all (see the approved plan, "ProsQA Ground-Truth
Transposition Validation, with False-Merge Path Signatures").

Ground truth: `prosqa.canonical_state_at` keys on the *current entity
reached*, not the full fact-sequence -- ProsQA's actual Markov state (facts
are never consumed, so two prefixes reaching the same entity are
interchangeable for everything that follows). The dataset's own
reconverging shortcut edge (`reconverge_fraction`, default 0.4) is a real,
by-construction transposition: a short (often 1-hop) route and the longer
main-chain route reaching the identical downstream entity.

Methodology note, found while implementing (not assumed at plan time):
`geometry.build_pairs` restricts pairs to the *same problem*, and
`same_step_only=True` (Connect Four's own default) additionally restricts to
close `step_idx`. That's the right restriction for Connect Four, where a
real transposition is by definition same-move-count (different order, same
total moves). It is the WRONG restriction here: ProsQA's real transposition
source, the reconverging shortcut, is explicitly a *different* number of
steps to reach the same entity. Run this script with `same_step_only=False`
as the headline check (or the reconverge cases fall outside the pairing
window and recall would be measuring the wrong thing), keeping
`same_step_only=True` only as a secondary, Connect-Four-comparable sanity
number.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .collect import collect_for_dataset
from .datasets import prosqa
from .geometry import SnapshotRecord, ground_truth_merge_confusion
from .model import load_model, resolve_layers
from .projection import apply_projection, load_projection


def _path_length_mismatch_stats(pairs: list[tuple[int, int]], records: list[SnapshotRecord]) -> dict:
    if not pairs:
        return {"n": 0}
    diffs = [abs(records[i].step_idx - records[j].step_idx) for i, j in pairs]
    same_trace = sum(1 for i, j in pairs if records[i].trace_idx == records[j].trace_idx)
    return {
        "n": len(pairs),
        "step_idx_diff_mean": float(np.mean(diffs)),
        "step_idx_diff_median": float(np.median(diffs)),
        "step_idx_diff_max": int(np.max(diffs)),
        "same_trace_fraction": same_trace / len(pairs),
    }


def _run_confusion(records, projected, tau, same_step_only) -> dict:
    result = ground_truth_merge_confusion(
        records, projected, tau=tau, same_step_only=same_step_only, return_pairs=True,
    )
    return result


def main():
    ap = argparse.ArgumentParser(description="ProsQA ground-truth transposition validation")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--projection-path", default="results/qwen7b_prosqa_projection.json")
    ap.add_argument("--tau", type=float, default=0.01, help="the already-deployed ProsQA tau")
    ap.add_argument("--n-positive", type=int, default=18)
    ap.add_argument("--num-traces", type=int, default=4, help="K, matches collect.py's own default")
    ap.add_argument("--max-snapshots-per-trace", type=int, default=3)
    ap.add_argument("--num-rollouts", type=int, default=4, help="v_hat is a byproduct, unused by this check")
    ap.add_argument("--max-new-tokens-trace", type=int, default=200)
    ap.add_argument("--max-new-tokens-rollout", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prosqa-chain-min", type=int, default=3)
    ap.add_argument("--prosqa-chain-max", type=int, default=5)
    ap.add_argument("--reconverge-fraction", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/prosqa_ground_truth_validation")
    args = ap.parse_args()

    start = time.time()
    lm = load_model(args.model, device=args.device, dtype=torch.float32)
    layers = resolve_layers(lm.num_hidden_layers)
    fp = load_projection(Path(args.projection_path))
    print(f"loaded projection: layer={fp.layer}, tau={args.tau}")

    instances = prosqa.generate_dataset(
        n_positive=args.n_positive, n_negative=0, seed=args.seed,
        reconverge_fraction=args.reconverge_fraction,
        chain_len_range=(args.prosqa_chain_min, args.prosqa_chain_max),
    )
    n_reconverge = sum(1 for i in instances if i.path_count >= 2)
    print(f"generated {len(instances)} positive instances ({n_reconverge} with path_count>=2, "
          f"i.e. a real reconverging alternate route)")

    records = collect_for_dataset(
        lm, "prosqa", instances, prosqa.build_prompt, prosqa.parse_and_verify,
        layers, args.num_traces, args.max_snapshots_per_trace, args.num_rollouts,
        args.max_new_tokens_trace, args.max_new_tokens_rollout, args.temperature,
        ground_truth_key_fn=prosqa.canonical_state_at,
    )
    n_keyed = sum(1 for r in records if r.ground_truth_key is not None)
    print(f"collected {len(records)} snapshots ({n_keyed} with a real ground-truth key, "
          f"{len(records) - n_keyed} skipped -- malformed/non-contiguous prefixes)")

    projected = np.array([apply_projection(r.hidden[fp.layer], fp) for r in records])

    print("\n=== headline: same_step_only=False (captures cross-depth transpositions, e.g. the shortcut) ===")
    headline = _run_confusion(records, projected, args.tau, same_step_only=False)
    for k in ("n_pairs", "true_merge", "false_merge", "missed_merge", "correct_non_merge", "precision", "recall"):
        print(f"  {k}: {headline.get(k)}")

    print("\n=== secondary, Connect-Four-comparable: same_step_only=True, step_tolerance=1 ===")
    secondary = _run_confusion(records, projected, args.tau, same_step_only=True)
    for k in ("n_pairs", "true_merge", "false_merge", "missed_merge", "correct_non_merge", "precision", "recall"):
        print(f"  {k}: {secondary.get(k)}")

    all_pairs = headline["false_merge_pairs"] + headline["true_merge_pairs"] + \
        headline["missed_merge_pairs"] + headline["correct_non_merge_pairs"]
    false_pairs = headline["false_merge_pairs"]

    print(f"\n=== false-merge trajectory audit ({len(false_pairs)} pairs) ===")
    false_merge_detail = []
    for i, j in false_pairs:
        ri, rj = records[i], records[j]
        detail = {
            "problem_id_i": ri.problem_id, "trace_idx_i": ri.trace_idx, "step_idx_i": ri.step_idx,
            "ground_truth_key_i": ri.ground_truth_key, "step_bodies_i": ri.step_bodies,
            "problem_id_j": rj.problem_id, "trace_idx_j": rj.trace_idx, "step_idx_j": rj.step_idx,
            "ground_truth_key_j": rj.ground_truth_key, "step_bodies_j": rj.step_bodies,
            "same_trace": ri.trace_idx == rj.trace_idx, "step_idx_diff": abs(ri.step_idx - rj.step_idx),
        }
        false_merge_detail.append(detail)
        print(f"  [{ri.problem_id}] trace{ri.trace_idx}@step{ri.step_idx}={ri.ground_truth_key!r} "
              f"vs trace{rj.trace_idx}@step{rj.step_idx}={rj.ground_truth_key!r} "
              f"(same_trace={detail['same_trace']}, step_idx_diff={detail['step_idx_diff']})")
        print(f"      i: {ri.step_bodies}")
        print(f"      j: {rj.step_bodies}")

    print("\n=== path-shaped signature check: false-merge pairs vs. full same-problem pair pool ===")
    false_stats = _path_length_mismatch_stats(false_pairs, records)
    pool_stats = _path_length_mismatch_stats(all_pairs, records)
    print(f"  false-merge pairs:  {false_stats}")
    print(f"  full pair pool:     {pool_stats}")
    if false_stats["n"] > 0 and pool_stats["n"] > 0:
        lift = false_stats["step_idx_diff_mean"] / pool_stats["step_idx_diff_mean"] if pool_stats["step_idx_diff_mean"] else float("nan")
        same_trace_lift = (false_stats["same_trace_fraction"] / pool_stats["same_trace_fraction"]
                            if pool_stats["same_trace_fraction"] else float("nan"))
        print(f"  step_idx_diff_mean ratio (false / pool): {lift:.2f}")
        print(f"  same_trace_fraction ratio (false / pool): {same_trace_lift:.2f}")
    else:
        print("  no false merges found -- no signature to characterize (a real, reportable null result).")

    elapsed = time.time() - start
    print(f"\ndone in {elapsed:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "records.pkl", "wb") as f:
        pickle.dump(records, f)
    with open(out_dir / "confusion.json", "w") as f:
        json.dump(
            {
                "headline_same_step_only_false": {k: v for k, v in headline.items() if not k.endswith("_pairs")},
                "secondary_same_step_only_true": {k: v for k, v in secondary.items() if not k.endswith("_pairs")},
                "false_merge_detail": false_merge_detail,
                "false_stats": false_stats, "pool_stats": pool_stats,
                "n_instances": len(instances), "n_reconverge_instances": n_reconverge,
                "n_records": len(records), "n_keyed_records": n_keyed,
                "args": vars(args),
            },
            f, indent=2, default=str,
        )
    print(f"saved records.pkl and confusion.json to {out_dir}")


if __name__ == "__main__":
    main()
