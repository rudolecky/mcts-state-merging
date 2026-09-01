"""Value-vs-hidden-state-distance geometry analysis.

Terminology note: what the original plan calls "whitening" is, at this
sample size, per-dimension z-scoring (mean/std standardization), not full
covariance whitening -- with a few hundred pooled states against a
1000+-dim hidden layer, a full inverse-covariance transform would be badly
ill-conditioned. Z-scoring is the right tool here.

Scope: z-scoring statistics are fit per (dataset, layer), never pooled
across datasets -- the correlation analysis itself only ever uses
within-problem pairs, so cross-dataset pooling of the standardization stats
would only risk one dataset's scale distorting the other's normalized
distances, with no compensating benefit.

Pair construction: the *primary* pair pool restricts to same-step-index
(+/- a small tolerance) pairs within a problem, to avoid conflating
reasoning-depth with semantic distance (a step-1 state and a step-3 state
differ partly just because they represent different amounts of progress,
which is a confound, not evidence about "same-meaning-implies-similar-value").
The unrestricted all-pairs version is reported as a secondary/sanity
comparison, not the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass(frozen=True)
class SnapshotRecord:
    dataset: str
    problem_id: str
    trace_idx: int
    step_idx: int  # 1-based position among that trace's detected step boundaries
    v_hat: float
    hidden: dict[str, np.ndarray]  # layer_name -> 1D vector
    boundary_kind: str = "step"  # "step" (Step N: lines) or "position" (fallback; see model.py)
    ground_truth_key: object | None = None  # exact "same state" label, dataset-populated only
    # where one exists (e.g. connect_four's canonical board+side-to-move, prosqa's current
    # entity reached); None for datasets with no ground-truth state identity (countdown),
    # which is also what makes ground_truth_merge_confusion() below a natural no-op for them.
    step_bodies: tuple[str, ...] | None = None  # the actual step-line text reaching this
    # snapshot, populated alongside ground_truth_key -- lets a false-merge audit show the two
    # conflicting trajectories directly instead of needing to regenerate anything.


def filter_consistent_boundary_kind(records: list[SnapshotRecord]) -> list[SnapshotRecord]:
    """Enforce one boundary_kind per problem (majority vote), dropping the
    minority-kind records. Boundary detection can pick "step" for one trace
    of a problem and fall back to "position" for another (e.g. a trace that
    happens to write a few loose "Step N:"-labeled lines without actually
    following the template) -- mixing them makes step_idx not comparable
    across traces of the same problem, which same-step pairing assumes.
    """
    from collections import Counter, defaultdict

    by_problem: dict[str, list[SnapshotRecord]] = defaultdict(list)
    for r in records:
        by_problem[r.problem_id].append(r)

    kept: list[SnapshotRecord] = []
    for recs in by_problem.values():
        kinds = Counter(r.boundary_kind for r in recs)
        majority_kind = kinds.most_common(1)[0][0]
        kept.extend(r for r in recs if r.boundary_kind == majority_kind)
    return kept


def fit_zscore(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/std over a pooled sample of vectors [n, dim]."""
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    std = np.maximum(std, 1e-8)  # floor to avoid divide-by-zero on dead dimensions
    return mean, std


def apply_zscore(vectors: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (vectors - mean) / std


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def build_pairs(
    records: list[SnapshotRecord], same_step_only: bool, step_tolerance: int = 1
) -> list[tuple[int, int]]:
    """Index pairs (into `records`) within the same problem_id, optionally
    restricted to |step_idx difference| <= step_tolerance.
    """
    by_problem: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        by_problem.setdefault(r.problem_id, []).append(i)

    pairs: list[tuple[int, int]] = []
    for indices in by_problem.values():
        for i, j in combinations(indices, 2):
            if same_step_only and abs(records[i].step_idx - records[j].step_idx) > step_tolerance:
                continue
            pairs.append((i, j))
    return pairs


def analyze_layer(
    records: list[SnapshotRecord],
    layer: str,
    metric: str,
    z_scored: bool,
    same_step_only: bool,
    step_tolerance: int = 1,
    false_merge_threshold: float = 0.3,
) -> dict:
    """Core Phase 0 statistic for one (layer, metric, z_scored, pair_mode)
    combination, over records assumed to already belong to a single dataset.
    """
    vectors = np.stack([r.hidden[layer] for r in records])
    if z_scored:
        mean, std = fit_zscore(vectors)
        vectors = apply_zscore(vectors, mean, std)

    dist_fn = cosine_distance if metric == "cosine" else l2_distance
    pairs = build_pairs(records, same_step_only=same_step_only, step_tolerance=step_tolerance)

    distances = np.array([dist_fn(vectors[i], vectors[j]) for i, j in pairs])
    abs_dv = np.array([abs(records[i].v_hat - records[j].v_hat) for i, j in pairs])

    n = len(pairs)
    if n < 5:
        return {
            "layer": layer, "metric": metric, "z_scored": z_scored,
            "same_step_only": same_step_only, "n_pairs": n,
            "spearman_rho": float("nan"), "spearman_pvalue": float("nan"),
            "false_merge_rate": float("nan"), "bottom_decile_cutoff": float("nan"),
        }

    rho, pvalue = spearmanr(distances, abs_dv)
    cutoff = float(np.quantile(distances, 0.10))
    bottom_mask = distances <= cutoff
    n_bottom = int(bottom_mask.sum())
    false_merge_rate = (
        float((abs_dv[bottom_mask] > false_merge_threshold).mean()) if n_bottom > 0 else float("nan")
    )

    return {
        "layer": layer, "metric": metric, "z_scored": z_scored,
        "same_step_only": same_step_only, "n_pairs": n, "n_bottom_decile": n_bottom,
        "spearman_rho": float(rho), "spearman_pvalue": float(pvalue),
        "false_merge_rate": false_merge_rate, "bottom_decile_cutoff": cutoff,
    }


def analyze_dataset(
    records: list[SnapshotRecord],
    layers: list[str],
    metrics: tuple[str, ...] = ("cosine", "l2"),
    z_scored_options: tuple[bool, ...] = (False, True),
    pair_modes: tuple[bool, ...] = (True, False),  # same_step_only values
) -> pd.DataFrame:
    """Run analyze_layer over every combination, for one dataset's records."""
    rows = []
    for layer in layers:
        for metric in metrics:
            for z_scored in z_scored_options:
                for same_step_only in pair_modes:
                    rows.append(
                        analyze_layer(records, layer, metric, z_scored, same_step_only)
                    )
    return pd.DataFrame(rows)


def bootstrap_ci_rho(
    records: list[SnapshotRecord],
    layer: str,
    metric: str,
    z_scored: bool,
    same_step_only: bool,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Bootstrap CI for the same-step (or all-pairs) Spearman rho, resampling
    at the *problem* level (with replacement), not the pair level -- pairs
    from the same problem aren't independent draws, so resampling pairs
    directly would understate the true uncertainty.
    """
    rng = np.random.default_rng(seed)
    by_problem: dict[str, list[SnapshotRecord]] = {}
    for r in records:
        by_problem.setdefault(r.problem_id, []).append(r)
    problem_ids = list(by_problem.keys())
    if len(problem_ids) < 3:
        return {"layer": layer, "metric": metric, "z_scored": z_scored, "rho": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}

    point = analyze_layer(records, layer, metric, z_scored, same_step_only)
    point_rho = point["spearman_rho"]

    boot_rhos = []
    for _ in range(n_boot):
        sampled_ids = rng.choice(problem_ids, size=len(problem_ids), replace=True)
        resampled = []
        for i, pid in enumerate(sampled_ids):
            # relabel so resampled copies of the same problem don't collide
            # with each other inside build_pairs' per-problem grouping
            for r in by_problem[pid]:
                resampled.append(replace_problem_id(r, f"{pid}__boot{i}"))
        result = analyze_layer(resampled, layer, metric, z_scored, same_step_only)
        if np.isfinite(result["spearman_rho"]):
            boot_rhos.append(result["spearman_rho"])

    if not boot_rhos:
        return {"layer": layer, "metric": metric, "z_scored": z_scored, "rho": point_rho,
                "ci_low": float("nan"), "ci_high": float("nan"), "n_boot_valid": 0}

    alpha = (1 - ci) / 2
    ci_low, ci_high = np.quantile(boot_rhos, [alpha, 1 - alpha])
    return {
        "layer": layer, "metric": metric, "z_scored": z_scored,
        "rho": point_rho, "ci_low": float(ci_low), "ci_high": float(ci_high),
        "n_boot_valid": len(boot_rhos),
    }


def replace_problem_id(record: SnapshotRecord, new_id: str) -> SnapshotRecord:
    return SnapshotRecord(
        dataset=record.dataset, problem_id=new_id, trace_idx=record.trace_idx,
        step_idx=record.step_idx, v_hat=record.v_hat, hidden=record.hidden,
        boundary_kind=record.boundary_kind, ground_truth_key=record.ground_truth_key,
    )


def ground_truth_merge_confusion(
    records: list[SnapshotRecord],
    projected: np.ndarray,
    tau: float,
    same_step_only: bool = True,
    step_tolerance: int = 1,
    return_pairs: bool = False,
) -> dict:
    """Cross ground-truth exact-state-equality against the projection's
    tau-threshold merge decision, at the actual deployed tau -- a strictly
    sharper validation than the value-outcome-based false-merge-rate proxy
    used everywhere else in this project, for a dataset that can supply a
    real "same state" label independent of the projection (so far:
    connect_four, prosqa).

    Pairs where either record's ground_truth_key is None are skipped, which
    makes this a natural no-op for countdown -- dataset-agnostic at the
    logic level even though the field is dataset-populated.

    Returns counts for the four confusion-matrix cells plus precision/recall
    of the merge decision against ground truth (precision = of the pairs the
    projection would merge, how many are real transpositions; recall = of
    the real transpositions, how many the projection would actually merge).

    If return_pairs is True, also returns the four categorized (i, j) index
    lists (true_merge_pairs, false_merge_pairs, missed_merge_pairs,
    correct_non_merge_pairs), letting a caller go back to `records` and
    inspect exactly which snapshots a false merge involved -- e.g. their
    step_bodies, for a false-merge trajectory audit. Omitted by default to
    keep the return shape identical to every existing caller.
    """
    pairs = build_pairs(records, same_step_only=same_step_only, step_tolerance=step_tolerance)
    pairs = [
        (i, j) for i, j in pairs
        if records[i].ground_truth_key is not None and records[j].ground_truth_key is not None
    ]
    if not pairs:
        result = {"n_pairs": 0, "tau": tau}
        if return_pairs:
            result.update(true_merge_pairs=[], false_merge_pairs=[], missed_merge_pairs=[], correct_non_merge_pairs=[])
        return result

    true_merge = false_merge = missed_merge = correct_non_merge = 0
    true_merge_pairs, false_merge_pairs, missed_merge_pairs, correct_non_merge_pairs = [], [], [], []
    for i, j in pairs:
        same_state = records[i].ground_truth_key == records[j].ground_truth_key
        would_merge = abs(projected[i] - projected[j]) < tau
        if same_state and would_merge:
            true_merge += 1
            true_merge_pairs.append((i, j))
        elif not same_state and would_merge:
            false_merge += 1
            false_merge_pairs.append((i, j))
        elif same_state and not would_merge:
            missed_merge += 1
            missed_merge_pairs.append((i, j))
        else:
            correct_non_merge += 1
            correct_non_merge_pairs.append((i, j))

    n_would_merge = true_merge + false_merge
    n_same_state = true_merge + missed_merge
    precision = true_merge / n_would_merge if n_would_merge > 0 else float("nan")
    recall = true_merge / n_same_state if n_same_state > 0 else float("nan")

    result = {
        "n_pairs": len(pairs), "tau": tau,
        "true_merge": true_merge, "false_merge": false_merge,
        "missed_merge": missed_merge, "correct_non_merge": correct_non_merge,
        "precision": precision, "recall": recall,
    }
    if return_pairs:
        result.update(
            true_merge_pairs=true_merge_pairs, false_merge_pairs=false_merge_pairs,
            missed_merge_pairs=missed_merge_pairs, correct_non_merge_pairs=correct_non_merge_pairs,
        )
    return result


def required_n_for_power(target_rho: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Standard Fisher-z sample-size formula for detecting a correlation of
    `target_rho` at the given power/alpha (two-sided). Answers "how many
    pairs would we actually need to tell this effect size apart from zero,"
    rather than sizing follow-up pilots by ad hoc doubling.
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    z_r = np.arctanh(target_rho)
    n = ((z_alpha + z_beta) / z_r) ** 2 + 3
    return int(np.ceil(n))


def gate_passes(df: pd.DataFrame, rho_threshold: float = 0.3, false_merge_threshold: float = 0.15) -> pd.DataFrame:
    """Rows that satisfy the plan's own go/no-go gate: rho > threshold AND
    false-merge rate < threshold, for the *primary* (same_step_only) pool."""
    primary = df[df["same_step_only"] == True]  # noqa: E712
    return primary[
        (primary["spearman_rho"] > rho_threshold)
        & (primary["false_merge_rate"] < false_merge_threshold)
    ]
