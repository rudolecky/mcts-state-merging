"""One-off analysis closing Phase 2's remaining gate (H5): is the deployed
merge threshold specifically worse on high-entropy (superposed/ambiguous)
states, not just worse on average? Reuses already-collected training
snapshots -- no new generation. Entropy is read straight off the already-
saved "final"-layer hidden vector via the model's own output head (one
matmul per record), so the only GPU cost here is a single model load.
"""

from __future__ import annotations

import pickle

import numpy as np

from .model import load_model, next_token_entropy_from_hidden
from .probe import run_probe
from .projection import apply_projection, calibrate_tau, entropy_stratified_false_merge, fit_final_projection

DATASETS = [
    {
        "name": "countdown",
        "path": "results/raw_qwen7b_countdown4_combined/snapshots_combined.pkl",
        "layer": "final",
        "tau": 0.0334,  # the tau actually deployed in the Countdown H3 runs
    },
    {
        "name": "prosqa",
        "path": "results/raw_qwen7b_prosqa/snapshots_seed80.pkl",
        "layer": "mid",
        "tau": 0.01,  # the tau actually deployed in the ProsQA H3 runs
    },
    {
        "name": "gsm8k",
        "path": "results/raw_qwen7b_gsm8k_combined.pkl",
        "layer": "final",  # the layer with the significant within-problem probe result
        "tau": None,  # no H3 search run yet -- calibrate on the fly (p40, matching how the
        # other two datasets' deployed taus were originally derived) instead of a guess
    },
]


def main() -> None:
    lm = load_model("Qwen/Qwen2.5-7B-Instruct", device="mps")

    for cfg in DATASETS:
        with open(cfg["path"], "rb") as f:
            data = pickle.load(f)
        records = [r for r in data["records"] if r.dataset == cfg["name"]]

        fp = fit_final_projection(records, cfg["layer"])
        projected = np.array([apply_projection(r.hidden[cfg["layer"]], fp) for r in records])
        entropies = np.array([next_token_entropy_from_hidden(lm, r.hidden["final"]) for r in records])

        tau = cfg["tau"]
        if tau is None:
            probe_result = run_probe(records, cfg["layer"], n_permutations=0)
            calib = calibrate_tau(records, probe_result["oof_predictions"], step_tolerance=0,
                                   percentiles=(10, 20, 30, 40, 50, 60, 70, 80))
            # pick the first percentile with a nonzero cutoff -- a tau of exactly 0 makes the
            # "< tau" merge test vacuous (nothing satisfies strictly-less-than-zero), and for
            # gsm8k specifically p10-p40 all land on exactly 0 (40%+ of same-step pairs are
            # exact projected-distance duplicates, an even higher rate than countdown/prosqa).
            row = next(r for r in calib if r["cutoff"] > 0)
            tau = row["cutoff"]
            print(f"  (no deployed tau for {cfg['name']} yet -- calibrated p{row['percentile']:.0f} "
                  f"tau={tau:.4f}, false_merge_rate={row['false_merge_rate']})")
        cfg = {**cfg, "tau": tau}

        print(f"--- {cfg['name']} (n_records={len(records)}) ---")
        print(f"  entropy range: {entropies.min():.3f} - {entropies.max():.3f}")
        for label, pct in [("median split (p50)", 50.0), ("top-quartile split (p75)", 75.0)]:
            result = entropy_stratified_false_merge(
                records, projected, entropies, tau=cfg["tau"], step_tolerance=0, split_percentile=pct,
            )
            print(f"  [{label}, cutoff={result['entropy_split_value']:.3f}]")
            for stratum_name, stats in result["strata"].items():
                print(f"    {stratum_name}: n_pairs_in_stratum={stats['n_pairs_in_stratum']} "
                      f"n_would_merge={stats['n_would_merge']} false_merge_rate={stats['false_merge_rate']}")


if __name__ == "__main__":
    main()
