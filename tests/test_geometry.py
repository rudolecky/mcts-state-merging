"""Validate the geometry/statistics module against synthetic data with a
known planted relationship -- this is the correctness check for the actual
regression machinery Phase 0's go/no-go call depends on.
"""

import numpy as np

from mcts_phase0.geometry import (
    SnapshotRecord,
    analyze_layer,
    apply_zscore,
    bootstrap_ci_rho,
    build_pairs,
    cosine_distance,
    filter_consistent_boundary_kind,
    fit_zscore,
    gate_passes,
    ground_truth_merge_confusion,
    l2_distance,
    required_n_for_power,
)


def _make_planted_records(seed: int, n_problems: int, n_states_per_problem: int, dim: int = 16,
                           noise_scale: float = 0.02, shuffle_value: bool = False) -> list[SnapshotRecord]:
    rng = np.random.default_rng(seed)
    records = []
    for p in range(n_problems):
        xs = rng.uniform(0, 1, size=n_states_per_problem)
        v_hats = xs.copy()
        if shuffle_value:
            rng.shuffle(v_hats)
        for s, (x, v) in enumerate(zip(xs, v_hats)):
            # encode x as an ANGLE (not just magnitude along a fixed axis) so
            # both cosine and L2 distance are sensitive to it -- a vector
            # that's mostly "x * fixed_direction + noise" has cosine
            # similarity ~1 for any x>0 (scaling doesn't change angle),
            # which would make cosine spuriously signal-free regardless of
            # whether analyze_layer is correct.
            angle = x * (np.pi / 2)
            vec = np.zeros(dim)
            vec[0] = np.cos(angle)
            vec[1] = np.sin(angle)
            vec[2:] = rng.normal(scale=noise_scale, size=dim - 2)  # irrelevant noise dims
            records.append(
                SnapshotRecord(
                    dataset="synthetic",
                    problem_id=f"p{p}",
                    trace_idx=s // 2,
                    step_idx=(s % 3) + 1,
                    v_hat=float(np.clip(v + rng.normal(scale=noise_scale), 0, 1)),
                    hidden={"layer0": vec},
                )
            )
    return records


def test_zscore_fit_and_apply_normalizes_to_unit_variance():
    rng = np.random.default_rng(0)
    vectors = rng.normal(loc=5.0, scale=3.0, size=(200, 8))
    mean, std = fit_zscore(vectors)
    z = apply_zscore(vectors, mean, std)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(z.std(axis=0), 1, atol=1e-8)


def test_zscore_floors_zero_variance_dimensions():
    vectors = np.ones((50, 3))  # every dimension constant -> std would be 0
    mean, std = fit_zscore(vectors)
    assert np.all(std >= 1e-8)
    z = apply_zscore(vectors, mean, std)
    assert np.all(np.isfinite(z))


def test_cosine_and_l2_distance_basic_properties():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_distance(a, a)) < 1e-9
    assert abs(cosine_distance(a, b) - 1.0) < 1e-9
    assert abs(l2_distance(a, a)) < 1e-9
    assert abs(l2_distance(a, b) - np.sqrt(2)) < 1e-9


def test_cosine_distance_handles_zero_vector_without_error():
    a = np.zeros(4)
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_distance(a, b) == 1.0


def test_build_pairs_stays_within_problem_and_respects_step_tolerance():
    records = _make_planted_records(seed=1, n_problems=3, n_states_per_problem=6)
    all_pairs = build_pairs(records, same_step_only=False)
    same_step_pairs = build_pairs(records, same_step_only=True, step_tolerance=0)

    for i, j in all_pairs:
        assert records[i].problem_id == records[j].problem_id
    for i, j in same_step_pairs:
        assert records[i].problem_id == records[j].problem_id
        assert records[i].step_idx == records[j].step_idx

    assert len(same_step_pairs) < len(all_pairs)


def test_analyze_layer_recovers_strong_signal_on_planted_relationship():
    records = _make_planted_records(seed=2, n_problems=25, n_states_per_problem=8, noise_scale=0.01)
    result = analyze_layer(records, layer="layer0", metric="l2", z_scored=False, same_step_only=False)
    assert result["n_pairs"] > 100
    assert result["spearman_rho"] > 0.8
    assert result["false_merge_rate"] < 0.1


def test_analyze_layer_finds_no_signal_when_value_is_shuffled():
    records = _make_planted_records(seed=3, n_problems=25, n_states_per_problem=8, shuffle_value=True)
    result = analyze_layer(records, layer="layer0", metric="l2", z_scored=False, same_step_only=False)
    assert abs(result["spearman_rho"]) < 0.4


def test_analyze_layer_zscoring_preserves_signal_direction():
    records = _make_planted_records(seed=4, n_problems=25, n_states_per_problem=8, noise_scale=0.01)
    raw = analyze_layer(records, layer="layer0", metric="cosine", z_scored=False, same_step_only=False)
    z = analyze_layer(records, layer="layer0", metric="cosine", z_scored=True, same_step_only=False)
    assert raw["spearman_rho"] > 0.5
    # z-scoring standardizes every dimension to unit variance, including the
    # 14 noise dims here vs. only 2 signal dims -- noise legitimately gets
    # more combined weight in the dot product after standardization, so a
    # weaker (but still clearly positive, not destroyed) correlation is the
    # realistic expectation, not a bug to chase away.
    assert z["spearman_rho"] > 0.2


def test_analyze_layer_too_few_pairs_returns_nan_not_error():
    records = _make_planted_records(seed=5, n_problems=1, n_states_per_problem=2)
    result = analyze_layer(records, layer="layer0", metric="l2", z_scored=False, same_step_only=False)
    assert result["n_pairs"] <= 4
    import math
    assert math.isnan(result["spearman_rho"])


def test_filter_consistent_boundary_kind_keeps_majority_per_problem():
    mk = lambda pid, kind, step: SnapshotRecord(  # noqa: E731
        dataset="d", problem_id=pid, trace_idx=0, step_idx=step, v_hat=0.5,
        hidden={"l": np.zeros(4)}, boundary_kind=kind,
    )
    records = [
        mk("p1", "position", 1), mk("p1", "position", 2), mk("p1", "step", 80),
        mk("p2", "step", 1), mk("p2", "step", 2),
    ]
    filtered = filter_consistent_boundary_kind(records)
    p1 = [r for r in filtered if r.problem_id == "p1"]
    p2 = [r for r in filtered if r.problem_id == "p2"]
    assert len(p1) == 2 and all(r.boundary_kind == "position" for r in p1)
    assert len(p2) == 2 and all(r.boundary_kind == "step" for r in p2)


def test_required_n_for_power_matches_known_reference_value():
    # standard power-analysis reference table: r=0.5, power=0.8, alpha=.05
    # two-tailed -> n=29 is the commonly cited (nearest-rounded) value; the
    # raw formula gives 29.01, and we ceil (not round) since under-sizing a
    # sample-size requirement is the wrong direction to round -- 30 is the
    # correct conservative answer, one above the table's rounded figure.
    assert required_n_for_power(0.5, power=0.8, alpha=0.05) == 30


def test_required_n_for_power_decreases_with_larger_effect():
    small = required_n_for_power(0.15)
    large = required_n_for_power(0.5)
    assert small > large  # smaller effects need more data to detect


def test_bootstrap_ci_excludes_zero_for_strong_signal():
    records = _make_planted_records(seed=10, n_problems=25, n_states_per_problem=8, noise_scale=0.01)
    result = bootstrap_ci_rho(records, "layer0", "l2", z_scored=False, same_step_only=False, n_boot=300, seed=0)
    assert result["ci_low"] > 0  # strong signal: CI shouldn't straddle zero


def test_bootstrap_ci_includes_zero_for_shuffled_data():
    records = _make_planted_records(seed=11, n_problems=25, n_states_per_problem=8, shuffle_value=True)
    result = bootstrap_ci_rho(records, "layer0", "l2", z_scored=False, same_step_only=False, n_boot=300, seed=0)
    assert result["ci_low"] < 0 < result["ci_high"]


def test_snapshot_record_old_style_construction_still_works():
    # Regression: adding ground_truth_key (default None) must not break
    # existing positional/keyword construction used by countdown/prosqa
    # collection and by every pre-existing pickle.
    r = SnapshotRecord(
        dataset="countdown", problem_id="p1", trace_idx=0, step_idx=1,
        v_hat=0.5, hidden={"layer0": np.zeros(4)},
    )
    assert r.ground_truth_key is None
    assert r.step_bodies is None


def test_ground_truth_merge_confusion_matches_hand_built_matrix_and_skips_none_keys():
    records = [
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=0, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key="A"),
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=1, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key="A"),
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=2, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key="B"),
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=3, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key="B"),
        # a countdown/prosqa-style record with no ground truth -- every pair
        # touching this one must be skipped, not miscounted.
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=4, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key=None),
    ]
    projected = np.array([0.10, 0.10, 0.10, 0.90, 0.10])

    result = ground_truth_merge_confusion(records, projected, tau=0.05, step_tolerance=0)

    assert result["n_pairs"] == 6  # 10 raw pairs among 5 records, 4 drop for a None key
    assert result["true_merge"] == 1       # (0,1): same key, indistinguishable projection
    assert result["false_merge"] == 2      # (0,2),(1,2): different key, indistinguishable projection
    assert result["missed_merge"] == 1     # (2,3): same key, but projection distinguishes them
    assert result["correct_non_merge"] == 2  # (0,3),(1,3): different key, projection distinguishes
    assert result["precision"] == 1 / 3
    assert result["recall"] == 1 / 2


def test_ground_truth_merge_confusion_return_pairs_gives_exact_indices():
    # Same fixture as the matrix test above -- return_pairs=True must yield
    # the exact same categorization, just with (i, j) identities attached
    # instead of only counts, so a caller can go back to `records` and
    # inspect e.g. step_bodies for a false-merge trajectory audit.
    records = [
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=0, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key="A"),
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=1, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key="A"),
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=2, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key="B"),
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=3, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key="B"),
        SnapshotRecord(dataset="d", problem_id="p1", trace_idx=4, step_idx=1, v_hat=0.5,
                       hidden={}, ground_truth_key=None),
    ]
    projected = np.array([0.10, 0.10, 0.10, 0.90, 0.10])

    result = ground_truth_merge_confusion(records, projected, tau=0.05, step_tolerance=0, return_pairs=True)

    assert result["true_merge_pairs"] == [(0, 1)]
    assert result["false_merge_pairs"] == [(0, 2), (1, 2)]
    assert result["missed_merge_pairs"] == [(2, 3)]
    assert result["correct_non_merge_pairs"] == [(0, 3), (1, 3)]
    # counts stay identical to the non-pairs call -- purely additive
    assert result["true_merge"] == 1
    assert result["false_merge"] == 2
    assert result["missed_merge"] == 1
    assert result["correct_non_merge"] == 2


def test_gate_passes_filters_correctly():
    import pandas as pd

    df = pd.DataFrame([
        {"layer": "mid", "same_step_only": True, "spearman_rho": 0.5, "false_merge_rate": 0.1},
        {"layer": "mid", "same_step_only": True, "spearman_rho": 0.1, "false_merge_rate": 0.5},
        {"layer": "final", "same_step_only": False, "spearman_rho": 0.9, "false_merge_rate": 0.0},
    ])
    passing = gate_passes(df)
    assert len(passing) == 1
    assert passing.iloc[0]["layer"] == "mid"
