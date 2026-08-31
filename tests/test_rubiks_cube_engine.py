"""Verification gate for rubiks_cube_engine.py, in the order specified by
the approved plan (cheerful-jumping-moler.md): rotation group construction,
hand-checked move mechanics, canonicalization invariant, then (marked slow,
excluded from the default fast suite) the ground-truth full-BFS state-count
cross-check against the well-known 3,674,160 figure.
"""

import random

import pytest

from mcts_phase0.datasets import rubiks_cube_engine as rc


# ---------- 1. rotation group construction ----------

def test_exactly_24_distinct_rotations():
    assert len(rc.ROTATIONS_24) == 24
    assert len(set(rc.ROTATIONS_24)) == 24


def test_rotation_group_closed_under_composition():
    rotset = set(rc.ROTATIONS_24)
    for a in rc.ROTATIONS_24:
        for b in rc.ROTATIONS_24:
            assert rc.compose(a, b) in rotset


def test_inverse_cancels_both_sides():
    for a in rc.ROTATIONS_24:
        assert rc.compose(a, rc.inverse(a)) == rc.IDENTITY
        assert rc.compose(rc.inverse(a), a) == rc.IDENTITY


def test_identity_is_a_no_op():
    for c in [(1, 1, 1), (1, 1, -1), (-1, 1, 1), (-1, -1, -1)]:
        assert rc.apply_rotation_to_coord(rc.IDENTITY, c) == c


# ---------- 2. move mechanics, hand-checked ----------

def test_every_move_has_order_four():
    for move in rc.ALL_MOVES:
        state = rc.SOLVED
        for i in range(3):
            state = rc.apply_move(state, move)
            assert state != rc.SOLVED, f"move {move} solved before 4 applications (i={i})"
        state = rc.apply_move(state, move)
        assert state == rc.SOLVED


def test_move_and_its_inverse_cancel():
    for move in rc.ALL_MOVES:
        inv = rc.Move(affected=move.affected, rotation=rc.inverse(move.rotation))
        assert inv in rc.ALL_MOVES
        state = rc.apply_move(rc.SOLVED, move)
        state = rc.apply_move(state, inv)
        assert state == rc.SOLVED


def test_twelve_distinct_moves():
    assert len(rc.ALL_MOVES) == 12
    assert len(set(rc.ALL_MOVES)) == 12


# ---------- 3. canonicalization invariant ----------

def test_canonical_form_holds_after_every_move_in_a_long_random_walk():
    rng = random.Random(0)
    state = rc.SOLVED
    for _ in range(2000):
        state = rc.apply_move(state, rng.choice(rc.ALL_MOVES))
        perm, orient = state
        assert sorted(perm) == list(range(8)), "perm must stay a valid bijection"
        assert perm[0] == 0, "reference corner must always be at position 0"
        assert orient[0] == rc.IDENTITY, "reference corner must always have identity orientation"


def test_solved_is_its_own_canonical_form():
    assert rc.canonicalize(rc.SOLVED) == rc.SOLVED


def test_is_goal_true_only_at_solved():
    assert rc.is_goal(rc.SOLVED)
    rng = random.Random(1)
    scrambled = rc.scramble(depth=5, rng=rng)
    assert not rc.is_goal(scrambled)


# ---------- 4. ground-truth cross-check (slow: full 3.67M-state BFS) ----------

@pytest.mark.slow
def test_full_bfs_reaches_exactly_the_known_state_count():
    dist = rc.bfs_distances()
    assert len(dist) == 3_674_160
