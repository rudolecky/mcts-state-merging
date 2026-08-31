"""Turns the evaluation-only LOPO probe machinery in probe.py into a
deployable, frozen artifact: one Ridge projection fit on ALL given records
(never a held-out fold), usable inside a live search loop.

Stored as plain numpy arrays, not a pickled sklearn estimator -- avoids
coupling a long-lived artifact to a specific sklearn version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import SnapshotRecord, build_pairs
from .probe import _fit_ridge


@dataclass(frozen=True)
class FrozenProjection:
    layer: str
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray
    intercept: float
    alpha: float
    clip: tuple[float, float] = (0.0, 1.0)


def fit_final_projection(
    records: list[SnapshotRecord],
    layer: str,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0),
) -> FrozenProjection:
    """Fit on every given record -- this is the deployed model, never a LOPO
    fold. Reuses probe._fit_ridge so the recipe can't drift from run_probe's.
    """
    vectors = np.stack([r.hidden[layer] for r in records])
    targets = np.array([r.v_hat for r in records])
    mean, std, ridge = _fit_ridge(vectors, targets, alphas)
    return FrozenProjection(
        layer=layer, mean=mean, std=std,
        coef=np.asarray(ridge.coef_, dtype=float), intercept=float(ridge.intercept_),
        alpha=float(ridge.alpha_),
    )


def apply_projection(vec: np.ndarray, fp: FrozenProjection) -> float:
    """Pure function of a hidden vector and a frozen projection -- no model
    or torch dependency, trivially unit-testable.
    """
    z = (vec - fp.mean) / fp.std
    raw = float(np.dot(z, fp.coef) + fp.intercept)
    lo, hi = fp.clip
    return max(lo, min(hi, raw))


def calibrate_tau(
    records: list[SnapshotRecord],
    oof_projected: np.ndarray,
    step_tolerance: int = 0,
    percentiles: tuple[float, ...] = (10, 20, 30, 40, 50),
    false_merge_threshold: float = 0.3,
) -> list[dict]:
    """Reuses geometry.build_pairs to report, at each requested percentile of
    same-depth projected-distance, the resulting cutoff and false-merge rate
    -- promotes the one-off threshold analysis into reusable code so a merge
    threshold is picked from real data, not guessed.
    """
    pairs = build_pairs(records, same_step_only=True, step_tolerance=step_tolerance)
    distances = np.array([abs(oof_projected[i] - oof_projected[j]) for i, j in pairs])
    abs_dv = np.array([abs(records[i].v_hat - records[j].v_hat) for i, j in pairs])

    rows = []
    for p in percentiles:
        cutoff = float(np.percentile(distances, p))
        bucket = distances <= cutoff
        n_bucket = int(bucket.sum())
        fmr = float((abs_dv[bucket] > false_merge_threshold).mean()) if n_bucket > 0 else float("nan")
        rows.append({"percentile": p, "cutoff": cutoff, "n_bucket": n_bucket, "false_merge_rate": fmr})
    return rows


def entropy_stratified_false_merge(
    records: list[SnapshotRecord],
    projected: np.ndarray,
    entropies: np.ndarray,
    tau: float,
    step_tolerance: int = 0,
    false_merge_threshold: float = 0.3,
    split_percentile: float = 50.0,
) -> dict:
    """At a fixed, already-deployed tau, split same-depth pairs into low-/
    high-entropy strata (median split on the pair's max next-token entropy)
    and report the false-merge rate within each. Closes the Phase 2 gate
    this session's H3 harness never checked: is the merge criterion
    specifically worse on high-entropy (superposed/ambiguous) states, not
    just worse on average.

    `projected` and `entropies` are plain arrays aligned by index with
    `records` -- callers compute them however they like (a fitted
    FrozenProjection, a next-token-entropy readout from the model's own
    lm_head), keeping this function itself model-free and unit-testable.

    `split_percentile` sets where the low/high entropy cut falls (default
    50 = median split); pass e.g. 75 to isolate just the highest-entropy
    tail, useful when the entropy distribution is heavily skewed toward
    zero (as it is for step-boundary snapshots in practice) and a plain
    median split would be degenerate.
    """
    pairs = build_pairs(records, same_step_only=True, step_tolerance=step_tolerance)
    if not pairs:
        return {"n_pairs": 0, "tau": tau}

    pair_entropy = np.array([max(entropies[i], entropies[j]) for i, j in pairs])
    distances = np.array([abs(projected[i] - projected[j]) for i, j in pairs])
    abs_dv = np.array([abs(records[i].v_hat - records[j].v_hat) for i, j in pairs])
    split_value = float(np.percentile(pair_entropy, split_percentile))

    strata = {}
    for name, in_stratum in [("low_entropy", pair_entropy <= split_value), ("high_entropy", pair_entropy > split_value)]:
        would_merge = in_stratum & (distances < tau)
        n_would_merge = int(would_merge.sum())
        fmr = float((abs_dv[would_merge] > false_merge_threshold).mean()) if n_would_merge > 0 else float("nan")
        strata[name] = {
            "n_pairs_in_stratum": int(in_stratum.sum()),
            "n_would_merge": n_would_merge,
            "false_merge_rate": fmr,
        }
    return {"n_pairs": len(pairs), "tau": tau, "entropy_split_value": split_value, "strata": strata}


def save_projection(fp: FrozenProjection, path: Path) -> None:
    payload = {
        "layer": fp.layer, "mean": fp.mean.tolist(), "std": fp.std.tolist(),
        "coef": fp.coef.tolist(), "intercept": fp.intercept, "alpha": fp.alpha, "clip": list(fp.clip),
    }
    Path(path).write_text(json.dumps(payload))


def load_projection(path: Path) -> FrozenProjection:
    payload = json.loads(Path(path).read_text())
    return FrozenProjection(
        layer=payload["layer"],
        mean=np.array(payload["mean"]), std=np.array(payload["std"]),
        coef=np.array(payload["coef"]), intercept=payload["intercept"],
        alpha=payload["alpha"], clip=tuple(payload["clip"]),
    )
