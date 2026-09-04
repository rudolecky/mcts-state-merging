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
import hashlib
import json
import os
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


# --- checkpointing -----------------------------------------------------------
# A four-arm ProsQA run is ~3.5h and the runner used to pickle everything only
# after the last problem, so an interruption discarded the whole spend (it did,
# twice: 50 and 90 minutes). Results are now written per (problem, arm) as they
# complete, and a restart reuses them.
#
# Per (problem, ARM), not per problem, because each arm is seeded from
# (held_out_seed, problem_id, arm) and therefore does not depend on which other
# arms ran or in what order. That makes an arm's shard reusable even when
# `--arms` changes -- so adding a fourth arm to a finished three-arm run costs
# only the fourth arm.

_ROW_SCHEMA = 1
"""Bumped whenever the per-arm row's key set changes. `load_arm` treats a row
from a different schema as absent, so a stale shard is recomputed instead of
crashing the summary after the results file has already been written."""


def arm_row(graph, elapsed_sec: float, K: int) -> dict:
    """The per-(problem, arm) record, built in exactly one place.

    Shared by the compute path and the schema test, so renaming a key cannot
    pass the tests while silently breaking resume."""
    return {
        "schema": _ROW_SCHEMA,
        "solved": is_solved(graph),
        "unique_nodes": sum(1 for n in graph.nodes.values() if n.kind in ("step", "answer")),
        "new_nodes": graph.new_nodes,
        "merge_hits": graph.merge_hits,
        "rejected": graph.rejected,
        # What the arms actually equalise is LLM sample batches per expansion
        # (expand_node always draws K), not nodes and not aggregate expansions
        # -- run_search can also exit on its iteration cap rather than on
        # budget. Recording both makes the matched-budget claim checkable.
        "expansions_used": graph.expansions_used,
        "llm_samples": graph.expansions_used * K,
        "elapsed_sec": elapsed_sec,
        "graph": graph,
    }


_FINGERPRINT_IGNORED = frozenset({
    "output_dir",       # where results go cannot change what a search does
    "tau_percentile",   # dead argument, never read -- see its help text
    "arms",             # arm identity lives in the shard filename instead
    "tau",              # ditto, but only for the arms that read it -- see _uses_tau
})


def _file_digest(path) -> str | None:
    """Content hash of a file an argument merely *names*, or None if absent."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


# Modules whose contents decide what a search computes. Hashed rather than
# using git HEAD: research work here runs from a dirty tree for days, so
# "HEAD + dirty flag" is the same string across every edit and would not
# discriminate the changes that actually matter.
_CODE_FILES = ("search.py", "model.py", "projection.py")


def _code_version() -> str:
    """Content hash of the modules that turn arguments into searches, so a
    checkpoint is not reused across a source change."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in _CODE_FILES:
        digest = _file_digest(here / name)
        h.update(f"{name}:{digest}".encode())
    return h.hexdigest()[:16]


def _fingerprint(args_dict: dict) -> str:
    """Hash of everything that changes what a search computes.

    Argument *values* are not sufficient. Two of them name files whose
    contents decide the search: `--projection-data` fixes the projection, and
    a regenerated snapshot pickle at the same path moves every node's
    `proj_value` -- which moves `guide_value`, so it changes the BASELINE arm
    too, not only the arms that match. And the code that turns arguments into
    searches can change under a checkpoint just as easily. Both are folded in
    here so a stale checkpoint is refused instead of quietly mixing two
    experiments into one results file.
    """
    payload = {k: v for k, v in sorted(args_dict.items()) if k not in _FINGERPRINT_IGNORED}
    payload["__projection_data_digest"] = _file_digest(args_dict.get("projection_data", ""))
    payload["__code_version"] = _code_version()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _uses_tau(arm: str) -> bool:
    """Whether this arm's result depends on tau at all.

    `find_merge_target` is only consulted when an arm merges or filters, so
    the `baseline` arm never reads tau and its result is bit-identical across
    a sweep (the per-arm seed is derived from held_out_seed/problem/arm, not
    from tau). Recording that lets a tau sweep reuse the baseline instead of
    recomputing it at every point -- on ProsQA that is ~67 minutes saved per
    additional tau."""
    kwargs = _ARM_KWARGS[arm]
    return bool(kwargs.get("merge_enabled") or kwargs.get("reject_duplicates"))


def _shard_path(partial_dir: Path, problem_id, arm: str, tau: float) -> Path:
    """Filesystem-safe, collision-free shard name for one (problem, arm, tau).

    tau is part of the name only for the arms that read it, so a sweep shares
    one baseline across every tau while keeping the treatment arms separate.
    """
    raw = str(problem_id)
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in raw)[:80]
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]  # keeps ids distinct after sanitising
    suffix = f".tau{tau:g}" if _uses_tau(arm) else ""
    return partial_dir / f"{safe}.{digest}.{arm}{suffix}.pkl"


def results_filename(held_out_seed: int, tau: float, arms) -> str:
    """Name for the assembled results pickle.

    Both varying keys are in the name, because the design deliberately has a
    sweep share one --output-dir (that is what makes the checkpoint reusable)
    and the assembled file must not be overwritten by the next point. Both
    cases were real: the first tau sweep point silently replaced the tau=0.01
    four-arm file, and "add a fourth arm cheaply" -- which this module's own
    output message recommends -- would otherwise replace a four-arm file with
    a one-arm file. The shards survive either way, so nothing is
    unrecoverable, but the assembled file should not need recovering.

    tau appears only when some arm actually reads it (see `_uses_tau`).
    """
    tau_tag = f"_tau{tau:g}" if any(_uses_tau(a) for a in arms) else ""
    arm_tag = "_" + "-".join(sorted(arms))
    return f"search_results_seed{held_out_seed}{tau_tag}{arm_tag}.pkl"


def open_checkpoint(out_dir: Path, args_dict: dict) -> Path:
    """Prepare (and validate) the checkpoint directory, returning its path.

    A checkpoint written under different search arguments is refused rather
    than silently mixed with new results -- reusing it would produce a file
    whose rows came from two different experiments.
    """
    partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    meta_path = partial_dir / "_meta.json"
    fingerprint = _fingerprint(args_dict)
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous.get("fingerprint") != fingerprint:
            # Name what differs. This directory may hold hours of compute, so
            # "delete it" is the last resort, not the first suggestion.
            old_args = previous.get("args", {})
            diffs = [f"{k}: {old_args.get(k)!r} -> {v!r}"
                     for k, v in sorted(args_dict.items())
                     if k not in _FINGERPRINT_IGNORED and str(old_args.get(k)) != str(v)]
            detail = "\n  ".join(diffs) if diffs else (
                "no argument differs -- so the projection-data file or the code changed "
                "(both are part of the fingerprint)")
            raise SystemExit(
                f"{partial_dir} holds a checkpoint from a different configuration "
                f"({previous.get('fingerprint')} != {fingerprint}):\n  {detail}\n"
                "Fix the differing argument to reuse the checkpoint, point --output-dir "
                "elsewhere, or delete the directory to start fresh."
            )
    else:
        meta_path.write_text(json.dumps({"fingerprint": fingerprint, "args": args_dict}, default=str, indent=2))
    return partial_dir


def load_arm(partial_dir: Path, problem_id, arm: str, tau: float):
    """The recorded result for one (problem, arm, tau), or None."""
    path = _shard_path(partial_dir, problem_id, arm, tau)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            row = pickle.load(f)
    except Exception as exc:
        # Usually a write truncated by an interruption, in which case
        # recomputing is exactly right. But this also catches a moved class,
        # a permission problem, or a corrupt file -- and silence there is
        # dangerous: every shard failing would look identical to "no
        # checkpoint", inviting the user to delete hours of recoverable work.
        print(f"  unreadable shard {path.name}: {exc!r} -- recomputing")
        return None
    if not isinstance(row, dict) or row.get("schema") != _ROW_SCHEMA:
        print(f"  shard {path.name} has schema {row.get('schema') if isinstance(row, dict) else '?'} "
              f"(expected {_ROW_SCHEMA}) -- recomputing")
        return None
    return row


def save_arm(partial_dir: Path, problem_id, arm: str, tau: float, payload) -> None:
    """Written to a temporary name and renamed, so an interruption during the
    write cannot leave a half-shard that a later run would read as complete."""
    path = _shard_path(partial_dir, problem_id, arm, tau)
    # Process-unique: the design encourages a sweep to share one --output-dir,
    # so two points running in parallel would otherwise interleave bytes into
    # one temp path. Correctness survives (the result fails to unpickle and is
    # recomputed) but the shared-baseline saving is silently lost.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    tmp.replace(path)


_ARM_KWARGS = {
    "baseline": {"merge_enabled": False},
    "treatment": {"merge_enabled": True},
    "reject": {"merge_enabled": False, "reject_duplicates": True},
    "reject_all": {"merge_enabled": False, "reject_duplicates": True, "reject_answer_nodes": True},
}


def main():
    ap = argparse.ArgumentParser(description="H3 merge-search experiment: does merging help at matched budget?")
    ap.add_argument("--dataset", choices=["countdown", "prosqa", "connect_four", "gsm8k"], default="countdown")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--projection-data", default="results/raw_qwen_countdown_combined/snapshots_combined.pkl")
    ap.add_argument("--projection-layer", default="3/4")
    ap.add_argument("--tau-percentile", type=float, default=40.0,
                    help="DEAD -- accepted for backwards compatibility with existing invocations "
                         "but never read. calibrate_tau hardcodes its own percentile list and "
                         "nothing passes this to it. Kept rather than removed so recorded args "
                         "from past runs still parse; excluded from the checkpoint fingerprint "
                         "for the same reason it is harmless.")
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
    ap.add_argument(
        "--arms", nargs="+", choices=list(_ARM_KWARGS), default=["baseline", "treatment"],
        help="search conditions to run per problem. baseline = no equivalence relation; "
             "treatment = merge (pool statistics of tau-equivalent nodes, the original arm); "
             "reject = admission filter (discard a tau-equivalent candidate instead of pooling, "
             "leaving answer candidates alone so a verifiable terminal is never dropped on the "
             "projection's word); reject_all = the same filter applied to answers too, which is "
             "the variant symmetric with merge.",
    )
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

    conditions = args.arms
    if len(set(conditions)) != len(conditions):
        raise SystemExit(f"--arms contains duplicates ({conditions}); each arm would be run "
                         "and then overwritten by its own rerun, doubling GPU time for nothing")

    out_dir = Path(args.output_dir)
    partial_dir = open_checkpoint(out_dir, vars(args))
    already = sum(1 for inst in held_out for arm in conditions
                  if _shard_path(partial_dir, inst.id, arm, args.tau).exists())
    total_units = len(held_out) * len(conditions)
    if already:
        print(f"checkpoint: {already}/{total_units} (problem, arm) results already recorded "
              f"in {partial_dir} -- reusing them")

    lm = load_model(args.model, device=args.device, dtype=torch.float32)

    results = []
    for inst in tqdm(held_out, desc="problems"):
        prompt_text = ds.build_prompt(inst)
        prompt_ids = build_chat_prompt_ids(lm, prompt_text)
        prompt_len = prompt_ids.shape[0]

        per_problem = {"problem_id": inst.id, "path_count": inst.path_count}
        for condition in conditions:
            recorded = load_arm(partial_dir, inst.id, condition, args.tau)
            if recorded is not None:
                per_problem[condition] = recorded
                continue
            # Seed per (held-out seed, problem, arm). Two reasons, both real:
            # `generate_traces` samples off global RNG state, so (a) arms run
            # sequentially would otherwise consume one stream in order --
            # meaning adding an arm changes the draws every later arm sees,
            # and no arm can be re-run without re-running the whole set --
            # and (b) seeding per arm pairs them on the same LM draws, which
            # is what the McNemar test in analyze_search_experiment.py
            # assumes it is getting.
            torch.manual_seed(
                int(hashlib.sha256(f"{args.held_out_seed}|{inst.id}|{condition}".encode()).hexdigest()[:8], 16)
            )
            config = SearchConfig(
                K=args.K, max_new_tokens_step=args.max_new_tokens_step, temperature=args.temperature,
                c_puct=args.c_puct, projection=fp, tau=args.tau,
                value_source=args.value_source, num_rollouts=args.num_rollouts,
                rollout_max_new_tokens=args.rollout_max_new_tokens,
                **_ARM_KWARGS[condition],
            )
            t0 = time.time()
            graph = run_search(lm, prompt_ids, prompt_len, inst, ds.parse_and_verify, config, args.budget)
            elapsed = time.time() - t0
            per_problem[condition] = arm_row(graph, elapsed, args.K)
            save_arm(partial_dir, inst.id, condition, args.tau, per_problem[condition])
        results.append(per_problem)

    elapsed_total = time.time() - start
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / results_filename(args.held_out_seed, args.tau, conditions)
    with open(out_path, "wb") as f:
        pickle.dump({"results": results, "args": vars(args), "elapsed_sec": elapsed_total, "tau_calibration": calib}, f)

    for condition in conditions:
        solved = sum(r[condition]["solved"] for r in results)
        exps = sum(r[condition]["expansions_used"] for r in results) / max(len(results), 1)
        samples = sum(r[condition]["llm_samples"] for r in results) / max(len(results), 1)
        nodes = sum(r[condition]["unique_nodes"] for r in results) / max(len(results), 1)
        secs = sum(r[condition]["elapsed_sec"] for r in results) / max(len(results), 1)
        merged = sum(r[condition]["merge_hits"] for r in results)
        dropped = sum(r[condition]["rejected"] for r in results)
        print(f"{condition:11s} solved: {solved}/{len(results)}  nodes/problem={nodes:.1f}  "
              f"expansions/problem={exps:.1f}  llm_samples/problem={samples:.0f}  "
              f"sec/problem={secs:.1f}  merge_hits={merged}  rejected={dropped}")
    print(f"total time: {elapsed_total:.1f}s -> {out_path}")
    print(f"(the {partial_dir} checkpoint is kept; delete it to force a full recompute, "
          f"or leave it to add another arm cheaply)")


if __name__ == "__main__":
    main()
