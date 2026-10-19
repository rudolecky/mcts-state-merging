"""Forces the one case the offline ground-truth validation
(`run_prosqa_ground_truth_validation.py`) never managed to test: two
genuinely different histories -- a short route and a longer route -- both
reaching the identical downstream entity. That run found 8/18 instances had
a real reconverging shortcut, but zero collected pairs actually landed on
matching ground-truth keys at different depths, because free-form K=4
sampling never happened to produce both routes for the same problem.

No sampling here: a route is a *known* fact-sequence, so its hidden state
comes from one teacher-forced forward pass per route (mirroring how
`collect_for_dataset` extracts a hidden vector at a step boundary), not
generation + rollouts. Cheaper than the previous run, and decisive instead
of merely absent -- see the approved plan ("Synthetic Reconvergence Probe:
Force the Untested Case") for the full rationale.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .datasets import prosqa
from .model import build_chat_prompt_ids, find_step_boundaries, hidden_states_for_sequence, load_model, resolve_layers
from .projection import apply_projection, load_projection


def _route_to_step_bodies(route: list[str]) -> list[str]:
    return [f"{route[k]} is a {route[k + 1]}" for k in range(len(route) - 1)]


def _route_continuation_text(route: list[str]) -> str:
    bodies = _route_to_step_bodies(route)
    return "".join(f"Step {k + 1}: {body}\n" for k, body in enumerate(bodies))


def _hidden_vec_at_route_end(lm, prompt_ids: torch.Tensor, layers: dict[str, int], route: list[str]) -> dict:
    text = _route_continuation_text(route)
    continuation_ids = lm.tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    full_ids = torch.cat([prompt_ids, continuation_ids])
    prompt_len = prompt_ids.shape[0]
    boundaries = find_step_boundaries(lm.tokenizer, prompt_len, full_ids)
    assert len(boundaries) == len(route) - 1, (
        f"expected {len(route) - 1} step boundaries for route {route}, found {len(boundaries)} -- "
        "constructed text didn't tokenize/decode back into the expected step lines"
    )
    token_idx = boundaries[-1]
    hidden = hidden_states_for_sequence(lm, full_ids)
    return {name: hidden[idx][0, token_idx, :].float().cpu().numpy() for name, idx in layers.items()}


def main():
    ap = argparse.ArgumentParser(description="ProsQA synthetic reconvergence probe")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--projection-path", default="results/qwen7b_prosqa_projection.json")
    ap.add_argument("--tau", type=float, default=0.01, help="the already-deployed ProsQA tau")
    ap.add_argument("--n-positive", type=int, default=40)
    ap.add_argument("--prosqa-chain-min", type=int, default=3)
    ap.add_argument("--prosqa-chain-max", type=int, default=5)
    ap.add_argument("--reconverge-fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="results/prosqa_reconvergence_probe")
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
    print(f"generated {len(instances)} positive instances")

    cases = []
    for inst in instances:
        found = prosqa.find_reconvergent_routes(inst.facts, inst.start)
        if found is not None:
            entity, short_route, long_route = found
            cases.append((inst, entity, short_route, long_route))
    print(f"{len(cases)}/{len(instances)} instances have a genuine reconvergence case")

    results = []
    n_recognized = 0
    for inst, entity, short_route, long_route in cases:
        prompt_text = prosqa.build_prompt(inst)
        prompt_ids = build_chat_prompt_ids(lm, prompt_text)

        # self-check: both routes must independently resolve to the same
        # ground-truth entity via the exact same key function the offline
        # validation used, before trusting anything about their embeddings.
        key_short = prosqa.canonical_state_at(inst, _route_to_step_bodies(short_route))
        key_long = prosqa.canonical_state_at(inst, _route_to_step_bodies(long_route))
        assert key_short == key_long == entity, (
            f"route construction bug: {inst.id} short={key_short} long={key_long} expected={entity}"
        )

        hidden_short = _hidden_vec_at_route_end(lm, prompt_ids, layers, short_route)
        hidden_long = _hidden_vec_at_route_end(lm, prompt_ids, layers, long_route)
        proj_short = apply_projection(hidden_short[fp.layer], fp)
        proj_long = apply_projection(hidden_long[fp.layer], fp)
        dist = abs(proj_short - proj_long)
        recognized = dist < args.tau
        n_recognized += recognized

        results.append({
            "problem_id": inst.id, "entity": entity,
            "short_route": short_route, "long_route": long_route,
            "short_hops": len(short_route) - 1, "long_hops": len(long_route) - 1,
            "proj_short": proj_short, "proj_long": proj_long, "dist": dist,
            "recognized_as_same_state": bool(recognized),
        })
        print(f"  [{inst.id}] entity={entity!r} short={short_route} (proj={proj_short:.4f}) "
              f"long={long_route} (proj={proj_long:.4f}) dist={dist:.4f} "
              f"{'RECOGNIZED' if recognized else 'MISSED'}")

    n_cases = len(cases)
    recall = n_recognized / n_cases if n_cases else float("nan")
    print(f"\n=== result: {n_recognized}/{n_cases} genuine reconvergence cases recognized as one state (recall={recall:.3f}) ===")

    elapsed = time.time() - start
    print(f"done in {elapsed:.1f}s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(
            {"n_cases": n_cases, "n_recognized": n_recognized, "recall": recall, "cases": results, "args": vars(args)},
            f, indent=2, default=str,
        )
    print(f"saved results.json to {out_dir}")


if __name__ == "__main__":
    main()
