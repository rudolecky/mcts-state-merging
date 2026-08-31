"""Linear-probe test: does a *trained* linear readout of a single hidden
state predict its own value, independent of any pairwise-distance question?

This tests a different, weaker claim than geometry.py's H1 regression. The
literature's strongest positive evidence (linear probes recovering outcome
correctness at AUC 0.7-0.93) is exactly this claim, not "distance predicts
value difference." A probe can find a thin, high-signal subspace that raw
distance never sees, swamped by irrelevant high-variance dimensions -- so
this can show real signal even when geometry.py's regression is flat, and
that would mean "wrong lens," not "no signal."

Leave-one-problem-out CV throughout: random k-fold would leak, since states
from the same problem are highly correlated (shared problem-level context).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV

from .geometry import SnapshotRecord


def _fit_ridge(train_vectors, train_targets, alphas):
    # No PCA: z-scoring equalizes every dimension to unit variance, which
    # would strip PCA of the variance-based cue it needs to find an
    # informative direction whenever informativeness happens to correlate
    # with raw variance (confirmed empirically: an earlier PCA-then-Ridge
    # version scored WORSE on a planted linear signal than on pure noise).
    # Ridge's regularized normal equations use X^T y directly, so it doesn't
    # depend on variance to find the target-correlated direction, and its
    # closed form is well-defined even with far more dimensions than samples
    # -- this is also what the literature's actual linear-probe papers do.
    mean = train_vectors.mean(axis=0)
    std = train_vectors.std(axis=0)
    std = np.maximum(std, 1e-8)
    train_z = (train_vectors - mean) / std

    ridge = RidgeCV(alphas=alphas)
    ridge.fit(train_z, train_targets)
    return mean, std, ridge


def _fit_predict_fold(train_vectors, train_targets, test_vectors, alphas):
    mean, std, ridge = _fit_ridge(train_vectors, train_targets, alphas)
    test_z = (test_vectors - mean) / std
    return ridge.predict(test_z)


def run_probe(
    records: list[SnapshotRecord],
    layer: str,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0),
    n_permutations: int = 0,
    rng_seed: int = 0,
) -> dict:
    """Leave-one-problem-out cross-validated linear probe: RidgeCV predicting
    v_hat from the (z-scored) hidden vector. Returns out-of-fold Spearman
    correlation between predicted and actual v_hat, plus an optional
    permutation p-value.
    """
    problem_ids = sorted({r.problem_id for r in records})
    if len(problem_ids) < 3:
        return {"layer": layer, "n_records": len(records), "n_problems": len(problem_ids),
                "spearman_rho": float("nan"), "p_value": float("nan")}

    vectors = np.stack([r.hidden[layer] for r in records])
    targets = np.array([r.v_hat for r in records])
    problem_arr = np.array([r.problem_id for r in records])

    def oof_predictions(y: np.ndarray) -> np.ndarray:
        preds = np.zeros_like(y, dtype=float)
        for pid in problem_ids:
            test_mask = problem_arr == pid
            train_mask = ~test_mask
            if train_mask.sum() < 5:
                preds[test_mask] = y[train_mask].mean() if train_mask.any() else 0.5
                continue
            preds[test_mask] = _fit_predict_fold(
                vectors[train_mask], y[train_mask], vectors[test_mask], alphas
            )
        return preds

    oof_pred = oof_predictions(targets)
    rho, _ = spearmanr(oof_pred, targets)
    rho = float(rho) if np.isfinite(rho) else float("nan")

    # Decompose into between-problem (does the probe just recognize which
    # problem this is, via a difficulty-correlated confound like entity-name
    # features?) vs within-problem (does it track state-level quality within
    # a fixed problem -- the thing that actually matters for merge-relevant
    # value estimation) signal.
    problem_mean_target = {pid: targets[problem_arr == pid].mean() for pid in problem_ids}
    problem_mean_pred = {pid: oof_pred[problem_arr == pid].mean() for pid in problem_ids}
    between_rho = float("nan")
    if len(problem_ids) >= 3:
        bt = np.array([problem_mean_target[p] for p in problem_ids])
        bp = np.array([problem_mean_pred[p] for p in problem_ids])
        r, _ = spearmanr(bp, bt)
        between_rho = float(r) if np.isfinite(r) else float("nan")

    centered_target = np.array([targets[i] - problem_mean_target[pid] for i, pid in enumerate(problem_arr)])
    centered_pred = np.array([oof_pred[i] - problem_mean_pred[pid] for i, pid in enumerate(problem_arr)])
    within_rho = float("nan")
    if np.std(centered_target) > 1e-9 and np.std(centered_pred) > 1e-9:
        r, _ = spearmanr(centered_pred, centered_target)
        within_rho = float(r) if np.isfinite(r) else float("nan")

    p_value = float("nan")
    within_p_value = float("nan")
    if n_permutations > 0 and np.isfinite(rho):
        rng = np.random.default_rng(rng_seed)
        null_rhos = []
        null_within_rhos = []
        for _ in range(n_permutations):
            permuted = rng.permutation(targets)
            perm_pred = oof_predictions(permuted)
            r, _ = spearmanr(perm_pred, permuted)
            if np.isfinite(r):
                null_rhos.append(r)

            # within-problem-only null: shuffle v_hat labels *inside* each
            # problem's own records, leaving between-problem means untouched.
            # This isolates significance of the within-problem signal alone,
            # rather than conflating it with the (often much larger, and
            # possibly confounded) between-problem effect.
            permuted_within = targets.copy()
            for pid in problem_ids:
                mask = problem_arr == pid
                permuted_within[mask] = rng.permutation(targets[mask])
            perm_pred_within = oof_predictions(permuted_within)
            perm_problem_mean_target = {p: permuted_within[problem_arr == p].mean() for p in problem_ids}
            perm_problem_mean_pred = {p: perm_pred_within[problem_arr == p].mean() for p in problem_ids}
            perm_centered_target = np.array(
                [permuted_within[i] - perm_problem_mean_target[pid] for i, pid in enumerate(problem_arr)]
            )
            perm_centered_pred = np.array(
                [perm_pred_within[i] - perm_problem_mean_pred[pid] for i, pid in enumerate(problem_arr)]
            )
            if np.std(perm_centered_target) > 1e-9 and np.std(perm_centered_pred) > 1e-9:
                r_w, _ = spearmanr(perm_centered_pred, perm_centered_target)
                if np.isfinite(r_w):
                    null_within_rhos.append(r_w)

        null_rhos = np.array(null_rhos)
        p_value = float((np.abs(null_rhos) >= abs(rho)).mean()) if len(null_rhos) else float("nan")
        null_within_rhos = np.array(null_within_rhos)
        if len(null_within_rhos) and np.isfinite(within_rho):
            within_p_value = float((np.abs(null_within_rhos) >= abs(within_rho)).mean())

    return {
        "layer": layer,
        "n_records": len(records),
        "n_problems": len(problem_ids),
        "spearman_rho": rho,
        "p_value": p_value,
        "n_permutations": n_permutations,
        "between_problem_rho": between_rho,
        "within_problem_rho": within_rho,
        "within_p_value": within_p_value,
        "oof_predictions": oof_pred,
    }


def _fit_predict_fold_pls(train_vectors, train_targets, test_vectors, n_components):
    mean = train_vectors.mean(axis=0)
    std = train_vectors.std(axis=0)
    std = np.maximum(std, 1e-8)
    train_z = (train_vectors - mean) / std
    test_z = (test_vectors - mean) / std

    k = min(n_components, train_z.shape[0] - 2, train_z.shape[1])
    k = max(k, 1)
    pls = PLSRegression(n_components=k)
    pls.fit(train_z, train_targets)
    pred_scalar = pls.predict(test_z).ravel()
    scores_kdim = pls.transform(test_z)
    return pred_scalar, scores_kdim


def run_pls_probe(
    records: list[SnapshotRecord],
    layer: str,
    n_components: int = 2,
    n_permutations: int = 0,
    rng_seed: int = 0,
) -> dict:
    """Same leave-one-problem-out structure as run_probe, but with a
    PLSRegression projection instead of Ridge's single direction. Unlike
    plain PCA, PLS finds its k components *supervised* by the target, so it
    doesn't share PCA-after-z-scoring's blind spot (needing informativeness
    to coincide with raw variance). Returns both a scalar prediction (for
    the same between/within decomposition run_probe computes) and the full
    k-dimensional out-of-fold projection (for analyze_projected_distance,
    which can use Euclidean distance in that richer space instead of
    collapsing everything to one scalar).
    """
    problem_ids = sorted({r.problem_id for r in records})
    if len(problem_ids) < 3:
        return {"layer": layer, "n_records": len(records), "n_problems": len(problem_ids),
                "spearman_rho": float("nan")}

    vectors = np.stack([r.hidden[layer] for r in records])
    targets = np.array([r.v_hat for r in records])
    problem_arr = np.array([r.problem_id for r in records])

    oof_pred = np.zeros_like(targets, dtype=float)
    oof_proj = np.zeros((len(records), n_components), dtype=float)
    for pid in problem_ids:
        test_mask = problem_arr == pid
        train_mask = ~test_mask
        if train_mask.sum() < 5:
            oof_pred[test_mask] = targets[train_mask].mean() if train_mask.any() else 0.5
            continue
        pred, scores = _fit_predict_fold_pls(vectors[train_mask], targets[train_mask], vectors[test_mask], n_components)
        oof_pred[test_mask] = pred
        oof_proj[test_mask, : scores.shape[1]] = scores

    rho, _ = spearmanr(oof_pred, targets)
    rho = float(rho) if np.isfinite(rho) else float("nan")

    problem_mean_target = {pid: targets[problem_arr == pid].mean() for pid in problem_ids}
    problem_mean_pred = {pid: oof_pred[problem_arr == pid].mean() for pid in problem_ids}
    bt = np.array([problem_mean_target[p] for p in problem_ids])
    bp = np.array([problem_mean_pred[p] for p in problem_ids])
    r, _ = spearmanr(bp, bt)
    between_rho = float(r) if np.isfinite(r) else float("nan")

    centered_target = np.array([targets[i] - problem_mean_target[pid] for i, pid in enumerate(problem_arr)])
    centered_pred = np.array([oof_pred[i] - problem_mean_pred[pid] for i, pid in enumerate(problem_arr)])
    within_rho = float("nan")
    if np.std(centered_target) > 1e-9 and np.std(centered_pred) > 1e-9:
        r, _ = spearmanr(centered_pred, centered_target)
        within_rho = float(r) if np.isfinite(r) else float("nan")

    return {
        "layer": layer, "n_records": len(records), "n_problems": len(problem_ids),
        "n_components": n_components, "spearman_rho": rho,
        "between_problem_rho": between_rho, "within_problem_rho": within_rho,
        "oof_predictions": oof_pred, "oof_projections": oof_proj,
    }


def analyze_projected_distance(
    records: list[SnapshotRecord],
    oof_projections: np.ndarray,
    same_step_only: bool = True,
    step_tolerance: int = 1,
    false_merge_threshold: float = 0.3,
) -> dict:
    """Re-run geometry.py's H1 pair regression, but with distance measured
    in a *learned* projection instead of raw/z-scored hidden-vector cosine
    or L2 distance. `oof_projections` may be 1D (a scalar score, e.g. from
    run_probe's Ridge direction) or 2D (n_records, k) for a richer
    multi-component projection (e.g. from run_pls_probe) -- distance is
    Euclidean in whatever space is given, which reduces to |a-b| when k=1.

    Every projection value must come from a model that held out the whole
    problem it's scoring (never trained on it), so this is not circular: it
    tests whether the learned direction(s) -- fit on OTHER problems --
    generalize to predicting that THIS problem's states have similar values
    when they're close in that space, exactly mirroring what
    geometry.analyze_layer tests for raw distance.
    """
    from .geometry import build_pairs

    oof_projections = np.asarray(oof_projections)
    if oof_projections.ndim == 1:
        oof_projections = oof_projections.reshape(-1, 1)

    pairs = build_pairs(records, same_step_only=same_step_only, step_tolerance=step_tolerance)
    if len(pairs) < 5:
        return {"n_pairs": len(pairs), "spearman_rho": float("nan"), "spearman_pvalue": float("nan"),
                "false_merge_rate": float("nan"), "bottom_decile_cutoff": float("nan")}

    proj_dist = np.array([np.linalg.norm(oof_projections[i] - oof_projections[j]) for i, j in pairs])
    abs_dv = np.array([abs(records[i].v_hat - records[j].v_hat) for i, j in pairs])

    rho, pvalue = spearmanr(proj_dist, abs_dv)
    cutoff = float(np.quantile(proj_dist, 0.10))
    bottom_mask = proj_dist <= cutoff
    n_bottom = int(bottom_mask.sum())
    false_merge_rate = float((abs_dv[bottom_mask] > false_merge_threshold).mean()) if n_bottom > 0 else float("nan")

    return {
        "n_pairs": len(pairs), "n_bottom_decile": n_bottom,
        "spearman_rho": float(rho) if np.isfinite(rho) else float("nan"),
        "spearman_pvalue": float(pvalue) if np.isfinite(pvalue) else float("nan"),
        "false_merge_rate": false_merge_rate, "bottom_decile_cutoff": cutoff,
    }
