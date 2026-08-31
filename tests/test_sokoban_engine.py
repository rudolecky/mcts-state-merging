"""Verification for sokoban_engine.py, no model/LLM involved:
- legal move correctness for walk vs. push at walls/edges
- apply_move correctness for both move kinds
- bfs_distances exactness on hand-verified small cases
- a genuine transposition (two different move sequences reaching the
  identical state -- here, a pure-walk round-trip that never touches the
  box, the expected common case for the domain's "walking is always
  reversible" component)
"""

from mcts_phase0.datasets.sokoban_engine import (
    GOAL,
    apply_move,
    bfs_distances,
    generate_puzzles,
    is_solved,
    legal_moves,
    make_state,
)


def test_is_solved_checks_box_on_goal_regardless_of_player():
    assert is_solved((GOAL,)) is True
    assert is_solved((GOAL - 1,)) is False


def test_legal_moves_walk_onto_free_cell():
    state = make_state(GOAL - 1, (GOAL,))  # player just left of the box
    moves = legal_moves(state)
    assert (-1, 0) in moves  # up: free floor cell, a plain walk
    assert (1, 0) in moves  # down: free floor cell, a plain walk


def test_legal_moves_push_when_beyond_cell_is_free():
    state = make_state(GOAL - 1, (GOAL,))  # player left of box, pushing right
    assert (0, 1) in legal_moves(state)  # dest has the box, beyond (GOAL+1) is free


def test_apply_move_push_moves_both_player_and_box():
    state = make_state(GOAL - 1, (GOAL,))
    new_state = apply_move(state, (0, 1))
    assert new_state == (GOAL, (GOAL + 1,))


def test_apply_move_walk_does_not_move_box():
    state = make_state(GOAL - 1, (GOAL,))
    new_state = apply_move(state, (-1, 0))
    assert new_state[1] == (GOAL,)  # box untouched by a walk


def test_bfs_distances_goal_configuration_is_zero():
    distances = bfs_distances()
    assert distances[make_state(GOAL - 1, (GOAL,))] == 0


def test_bfs_distances_hand_verified_one_push_case():
    # player=53, box=54 -- one push right solves it (54 -> 55 == GOAL).
    distances = bfs_distances()
    state = make_state(53, (54,))
    assert distances[state] == 1
    new_state = apply_move(state, (0, 1))
    assert is_solved(new_state[1])


def test_bfs_distances_hand_verified_two_push_case():
    # player=52, box=53: push right twice (53->54, then 54->55==GOAL).
    distances = bfs_distances()
    state = make_state(52, (53,))
    assert distances[state] == 2
    s1 = apply_move(state, (0, 1))
    assert s1 == (53, (54,))
    s2 = apply_move(s1, (0, 1))
    assert is_solved(s2[1])


def test_two_distinct_same_length_walk_paths_reach_identical_state():
    # Found by exhaustive search from (52,(53,)): the pure-walk sequences
    # ((-1,0),(-1,0),(1,0)) and ((-1,0),(0,-1),(0,1)) both land on
    # (42,(53,)) with the box never touched -- a genuine transposition.
    start = make_state(52, (53,))

    def _play(moves):
        state = start
        for m in moves:
            state = apply_move(state, m)
        return state

    result_a = _play(((-1, 0), (-1, 0), (1, 0)))
    result_b = _play(((-1, 0), (0, -1), (0, 1)))
    assert result_a == result_b == (42, (53,))


def test_generate_puzzles_returns_states_at_exact_target_distance():
    distances = bfs_distances()
    instances = generate_puzzles(n=4, seed=0, target_distance=2, distances=distances)
    assert len(instances) == 4
    ids = {inst.id for inst in instances}
    assert len(ids) == 4
    for inst in instances:
        assert distances[inst.start_state] == 2
        assert inst.target_distance == 2
