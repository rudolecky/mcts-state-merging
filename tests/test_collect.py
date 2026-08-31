"""Verification for collect.py's pure snapshot-index-selection logic."""

from mcts_phase0.collect import _select_snapshot_indices


def test_returns_all_indices_when_under_cap():
    assert _select_snapshot_indices(2, cap=3) == [0, 1]


def test_even_spacing_spans_the_full_range():
    assert _select_snapshot_indices(10, cap=3, selection="even") == [0, 4, 9]


def test_first_takes_the_earliest_boundaries_regardless_of_total_length():
    assert _select_snapshot_indices(17, cap=3, selection="first") == [0, 1, 2]
    assert _select_snapshot_indices(3, cap=3, selection="first") == [0, 1, 2]


def test_first_and_even_agree_when_boundaries_fit_within_cap():
    assert _select_snapshot_indices(2, cap=3, selection="first") == [0, 1]
