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
    ap.add_argument("--baseline-key", default="baseline")
    ap.add_argument(
        "--treatment-key", default="treatment",
        help="which arm to compare against the baseline. The runner can now record more than "
             "two ('reject'/'reject_all' for the admission filter), and every statistic below "
             "is a paired baseline-vs-one-treatment comparison, so the arm has to be named.",
    )
    args = ap.parse_args()
    base, treat = args.baseline_key, args.treatment_key

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    results = data["results"]
    n = len(results)
    print(f"loaded {n} problems, budget/K/tau from run: "
          f"budget={data['args']['budget']} K={data['args']['K']} tau={data['args']['tau']}")

    if base == treat:
        raise SystemExit("--baseline-key and --treatment-key are the same arm; every statistic "
                         "below would be a comparison of an arm with itself (p=1.0, delta 0)")
    for key in (base, treat):
        missing = [i for i, r in enumerate(results) if key not in r]
        if missing:
            raise SystemExit(
                f"arm {key!r} missing from {len(missing)} of {len(results)} problems "
                f"(first: index {missing[0]}); recorded arms in problem 0: "
                f"{sorted(k for k, v in results[0].items() if isinstance(v, dict))}")
    solved_baseline = [r[base]["solved"] for r in results]
    solved_treatment = [r[treat]["solved"] for r in results]

    n_b = sum(solved_baseline)
    n_t = sum(solved_treatment)
    print(f"\n{base} solved:  {n_b}/{n} ({n_b/n:.1%})")
    print(f"{treat} solved: {n_t}/{n} ({n_t/n:.1%})")

    # discordant pairs
    b_only = sum(1 for r in results if r[base]["solved"] and not r[treat]["solved"])
    t_only = sum(1 for r in results if r[treat]["solved"] and not r[base]["solved"])
    both = sum(1 for r in results if r[base]["solved"] and r[treat]["solved"])
    neither = sum(1 for r in results if not r[base]["solved"] and not r[treat]["solved"])
    print(f"\ncontingency: both_solved={both} {base}_only={b_only} {treat}_only={t_only} neither={neither}")

    p_value = mcnemar_exact(b_only, t_only)
    print(f"McNemar exact p-value: {p_value:.4f} (discordant pairs: {b_only} vs {t_only})")

    point, lo, hi = bootstrap_accuracy_delta_ci(solved_baseline, solved_treatment)
    print(f"accuracy delta ({treat} - {base}): {point:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]")

    # diagnostics
    print("\n--- diagnostics ---")
    for cond in (base, treat):
        unique_nodes = [r[cond]["unique_nodes"] for r in results]
        elapsed = [r[cond]["elapsed_sec"] for r in results]
        extra = ""
        if "merge_hits" in results[0][cond]:
            extra = (f"  merge_hits={sum(r[cond]['merge_hits'] for r in results)}"
                     f"  rejected={sum(r[cond]['rejected'] for r in results)}")
        print(f"{cond}: mean unique_nodes={np.mean(unique_nodes):.1f}  "
              f"mean wall-clock={np.mean(elapsed):.1f}s{extra}")
    print("  NOTE on wall-clock: a resumed or swept run reuses shards, and a reused arm carries "
          "the timing of the run that computed it. In a tau sweep the baseline is shared across "
          "every point by design, so its time was measured once, possibly under different machine "
          "load than the arms it is being compared against. Treat cross-run time comparisons as "
          "indicative only.")

    # Fraction of nodes with more than one parent -- 0 by construction in any
    # arm that filters instead of pooling, which is the property that makes
    # the filter immune to the compounding value-corruption merging risks.
    merge_rates = []
    for r in results:
        g = r[treat]["graph"]
        non_root = [nd for nd in g.nodes.values() if nd.node_id != g.root_id]
        if not non_root:
            continue
        merged = sum(1 for nd in non_root if len(nd.parents) > 1)
        merge_rates.append(merged / len(non_root))
    print(f"{treat} merge rate (fraction of nodes with >1 parent): mean={np.mean(merge_rates):.3f}")

    # Deliberately NOT "unique_nodes < budget/2". That threshold was calibrated
    # when every arm produced ~K nodes per expansion; an arm that deduplicates
    # produces far fewer by design, so the old test fired for the treatment as
    # an artifact of the treatment working. What actually indicates early
    # saturation is an arm not spending the expansions it was given.
    if "expansions_used" in results[0][base]:
        budget = data["args"]["budget"]
        for cond in (base, treat):
            short = sum(1 for r in results if r[cond]["expansions_used"] < budget)
            mean_exp = np.mean([r[cond]["expansions_used"] for r in results])
            print(f"(informational) {cond}: {short}/{n} problems spent fewer than the full "
                  f"{budget} expansions (mean {mean_exp:.1f}) -- run_search can exit on its "
                  f"iteration cap instead of on budget")
    else:
        print("\n(informational) expansions_used not recorded in this run "
              "(pre-dates the field); matched-budget delivery is unverifiable here")


if __name__ == "__main__":
    main()
