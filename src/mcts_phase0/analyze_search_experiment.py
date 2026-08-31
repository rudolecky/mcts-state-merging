"""CLI: analyze a run_search_experiment.py output -- primary metric (paired
McNemar's exact test + problem-level bootstrap CI on the accuracy delta) and
diagnostics (merge activity, unique-node counts, wall-clock).
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(b: int, c: int) -> float:
    """b = baseline-wrong/treatment-right count, c = the reverse. Exact
    two-sided McNemar test via the binomial distribution on discordant pairs
    (no continuity correction needed for the exact version)."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def bootstrap_accuracy_delta_ci(solved_baseline: list[bool], solved_treatment: list[bool],
                                 n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Resample problems (not conditions) with replacement -- respects the
    paired structure, same problem-level-resampling convention used
    throughout this project's geometry.bootstrap_ci_rho."""
    rng = np.random.default_rng(seed)
    n = len(solved_baseline)
    b = np.array(solved_baseline, dtype=float)
    t = np.array(solved_treatment, dtype=float)
    point = t.mean() - b.mean()
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(t[idx].mean() - b[idx].mean())
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description="Analyze the H3 merge-search experiment")
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    results = data["results"]
    n = len(results)
    print(f"loaded {n} problems, budget/K/tau from run: "
          f"budget={data['args']['budget']} K={data['args']['K']} tau={data['args']['tau']}")

    solved_baseline = [r["baseline"]["solved"] for r in results]
    solved_treatment = [r["treatment"]["solved"] for r in results]

    n_b = sum(solved_baseline)
    n_t = sum(solved_treatment)
    print(f"\nbaseline solved:  {n_b}/{n} ({n_b/n:.1%})")
    print(f"treatment solved: {n_t}/{n} ({n_t/n:.1%})")

    # discordant pairs
    b_only = sum(1 for r in results if r["baseline"]["solved"] and not r["treatment"]["solved"])
    t_only = sum(1 for r in results if r["treatment"]["solved"] and not r["baseline"]["solved"])
    both = sum(1 for r in results if r["baseline"]["solved"] and r["treatment"]["solved"])
    neither = sum(1 for r in results if not r["baseline"]["solved"] and not r["treatment"]["solved"])
    print(f"\ncontingency: both_solved={both} baseline_only={b_only} treatment_only={t_only} neither={neither}")

    p_value = mcnemar_exact(b_only, t_only)
    print(f"McNemar exact p-value: {p_value:.4f} (discordant pairs: {b_only} vs {t_only})")

    point, lo, hi = bootstrap_accuracy_delta_ci(solved_baseline, solved_treatment)
    print(f"accuracy delta (treatment - baseline): {point:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")

    # diagnostics
    print("\n--- diagnostics ---")
    for cond in ("baseline", "treatment"):
        unique_nodes = [r[cond]["unique_nodes"] for r in results]
        elapsed = [r[cond]["elapsed_sec"] for r in results]
        print(f"{cond}: mean unique_nodes={np.mean(unique_nodes):.1f}  mean wall-clock={np.mean(elapsed):.1f}s")

    merge_rates = []
    for r in results:
        g = r["treatment"]["graph"]
        non_root = [nd for nd in g.nodes.values() if nd.node_id != g.root_id]
        if not non_root:
            continue
        merged = sum(1 for nd in non_root if len(nd.parents) > 1)
        merge_rates.append(merged / len(non_root))
    print(f"treatment merge rate (fraction of nodes with >1 parent): mean={np.mean(merge_rates):.3f}")

    saturated = sum(
        1 for r in results for cond in ("baseline", "treatment")
        if r[cond]["unique_nodes"] < data["args"]["budget"] * 0.5
    )
    print(f"\n(informational) condition-runs with notably fewer nodes than budget "
          f"(possible early saturation): {saturated}/{2*n}")


if __name__ == "__main__":
    main()
