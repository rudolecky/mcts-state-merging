import numpy as np

from mcts_phase0.geometry import SnapshotRecord
from mcts_phase0.run_prosqa_ground_truth_validation import _path_length_mismatch_stats


def _rec(trace_idx, step_idx):
    return SnapshotRecord(
        dataset="prosqa", problem_id="p1", trace_idx=trace_idx, step_idx=step_idx,
        v_hat=0.5, hidden={},
    )


def test_path_length_mismatch_stats_empty_pairs():
    assert _path_length_mismatch_stats([], []) == {"n": 0}


def test_path_length_mismatch_stats_hand_computed():
    records = [_rec(0, 1), _rec(1, 3), _rec(0, 4)]
    # pair (0,1): different trace, |1-3|=2 ; pair (0,2): same trace, |1-4|=3
    pairs = [(0, 1), (0, 2)]
    stats = _path_length_mismatch_stats(pairs, records)
    assert stats["n"] == 2
    assert stats["step_idx_diff_mean"] == 2.5
    assert stats["step_idx_diff_median"] == 2.5
    assert stats["step_idx_diff_max"] == 3
    assert stats["same_trace_fraction"] == 0.5  # only pair (0,2) is same-trace


def test_path_length_mismatch_stats_all_same_trace():
    records = [_rec(0, 1), _rec(0, 2)]
    stats = _path_length_mismatch_stats([(0, 1)], records)
    assert stats["same_trace_fraction"] == 1.0
    assert stats["step_idx_diff_mean"] == 1.0
