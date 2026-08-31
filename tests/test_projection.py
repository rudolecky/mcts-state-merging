"""Validate the frozen, deployable projection: fit-on-all-data matches the
same recipe as probe.py's LOPO folds, apply_projection is a pure function,
save/load round-trips, and calibrate_tau reproduces sane percentile stats.
"""

import numpy as np

from mcts_phase0.geometry import SnapshotRecord
from mcts_phase0.probe import run_probe
from mcts_phase0.projection import (
    apply_projection,
    calibrate_tau,
    entropy_stratified_false_merge,
    fit_final_projection,
    load_projection,
    save_projection,
)


def _make_records(seed: int, n_problems: int, n_per_problem: int, dim: int = 16,
                   noise_scale: float = 0.05) -> list[SnapshotRecord]:
    rng = np.random.default_rng(seed)
    records = []
    for p in range(n_problems):
        for s in range(n_per_problem):
            x = rng.uniform(0, 1)
            vec = rng.normal(scale=noise_scale, size=dim)
            vec[0] += x
            v_hat = float(np.clip(x + rng.normal(scale=0.05), 0, 1))
            records.append(
                SnapshotRecord(dataset="synthetic", problem_id=f"p{p}", trace_idx=s, step_idx=(s % 3) + 1,
                               v_hat=v_hat, hidden={"layer0": vec})
            )
    return records


def test_fit_final_projection_recovers_planted_signal():
    records = _make_records(seed=0, n_problems=20, n_per_problem=6)
    fp = fit_final_projection(records, "layer0")
    preds = [apply_projection(r.hidden["layer0"], fp) for r in records]
    actual = [r.v_hat for r in records]
    from scipy.stats import spearmanr
    rho, _ = spearmanr(preds, actual)
    assert rho > 0.7  # fit-on-all-data should recover the signal at least as well as LOPO


def test_apply_projection_is_pure_and_matches_hand_computation():
    from mcts_phase0.projection import FrozenProjection
    fp = FrozenProjection(layer="l", mean=np.array([1.0, 2.0]), std=np.array([2.0, 4.0]),
                           coef=np.array([1.0, 1.0]), intercept=0.1, alpha=1.0)
    vec = np.array([3.0, 6.0])
    z = (vec - fp.mean) / fp.std  # [1.0, 1.0]
    expected = min(1.0, max(0.0, float(np.dot(z, fp.coef) + fp.intercept)))
    assert apply_projection(vec, fp) == expected


def test_apply_projection_clips_to_range():
    from mcts_phase0.projection import FrozenProjection
    fp = FrozenProjection(layer="l", mean=np.array([0.0]), std=np.array([1.0]),
                           coef=np.array([100.0]), intercept=0.0, alpha=1.0)
    assert apply_projection(np.array([10.0]), fp) == 1.0
    assert apply_projection(np.array([-10.0]), fp) == 0.0


def test_save_and_load_projection_round_trips(tmp_path):
    records = _make_records(seed=1, n_problems=10, n_per_problem=4)
    fp = fit_final_projection(records, "layer0")
    path = tmp_path / "projection.json"
    save_projection(fp, path)
    loaded = load_projection(path)

    assert loaded.layer == fp.layer
    assert np.allclose(loaded.mean, fp.mean)
    assert np.allclose(loaded.std, fp.std)
    assert np.allclose(loaded.coef, fp.coef)
    assert loaded.intercept == fp.intercept
    for r in records[:5]:
        assert apply_projection(r.hidden["layer0"], loaded) == apply_projection(r.hidden["layer0"], fp)


def test_calibrate_tau_reports_increasing_cutoffs_and_bucket_sizes():
    records = _make_records(seed=2, n_problems=20, n_per_problem=6)
    result = run_probe(records, "layer0", n_permutations=0)
    rows = calibrate_tau(records, result["oof_predictions"], step_tolerance=0)

    cutoffs = [row["cutoff"] for row in rows]
    bucket_sizes = [row["n_bucket"] for row in rows]
    assert cutoffs == sorted(cutoffs)  # higher percentile -> higher (or equal) cutoff
    assert bucket_sizes == sorted(bucket_sizes)
    assert all(0.0 <= row["false_merge_rate"] <= 1.0 for row in rows if row["n_bucket"] > 0)


def test_entropy_stratified_false_merge_detects_planted_high_entropy_effect():
    # Two problems, kept separate so build_pairs never mixes their pairs:
    # "lowE" carries low entropy + small, harmless projected-distance/value
    # gaps (true merges); "highE" carries high entropy + the same small
    # projected distance but large value gaps (false merges). A correct
    # implementation must report false_merge_rate == 0.0 for the low stratum
    # and == 1.0 for the high stratum, not blend them together.
    records = [
        SnapshotRecord(dataset="d", problem_id="lowE", trace_idx=0, step_idx=1, v_hat=0.50, hidden={}),
        SnapshotRecord(dataset="d", problem_id="lowE", trace_idx=1, step_idx=1, v_hat=0.52, hidden={}),
        SnapshotRecord(dataset="d", problem_id="lowE", trace_idx=2, step_idx=1, v_hat=0.51, hidden={}),
        SnapshotRecord(dataset="d", problem_id="highE", trace_idx=0, step_idx=1, v_hat=0.10, hidden={}),
        SnapshotRecord(dataset="d", problem_id="highE", trace_idx=1, step_idx=1, v_hat=0.90, hidden={}),
        SnapshotRecord(dataset="d", problem_id="highE", trace_idx=2, step_idx=1, v_hat=0.50, hidden={}),
    ]
    projected = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.11])
    entropies = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])

    result = entropy_stratified_false_merge(records, projected, entropies, tau=0.05, step_tolerance=0)

    assert result["entropy_split_value"] == 0.5
    assert result["strata"]["low_entropy"]["n_would_merge"] == 3
    assert result["strata"]["low_entropy"]["false_merge_rate"] == 0.0
    assert result["strata"]["high_entropy"]["n_would_merge"] == 3
    assert result["strata"]["high_entropy"]["false_merge_rate"] == 1.0
