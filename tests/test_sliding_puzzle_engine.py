"""Verification for sliding_puzzle_engine.py, no model/LLM involved:
- blank movement / legal-move bookkeeping at edges and corners
- apply_move swap correctness
- bfs_distances exactness on hand-verified small cases
- a genuine transposition: two different same-length move sequences from
  the same start reaching the identical state (the premise the whole
  second-domain experiment turns on)
"""

from mcts_phase0.datasets.sliding_puzzle_engine import (
    apply_move,
    bfs_distances,
    blank_index,
    generate_puzzles,
    goal_state,
    legal_moves,
)


def test_goal_state_shape():
    assert goal_state(3, 3) == (1, 2, 3, 4, 5, 6, 7, 8, 0)
    assert goal_state(3, 2) == (1, 2, 3, 4, 5, 0)


def test_blank_index_finds_zero():
    assert blank_index((1, 2, 3, 4, 5, 0)) == 5
    assert blank_index((0, 1, 2, 3, 4, 5)) == 0


def test_legal_moves_corner_has_two_options():
    # blank at bottom-right corner of a 3x2 board (idx5): can only go up or left
    state = goal_state(3, 2)
    assert sorted(legal_moves(state, width=3, height=2)) == [2, 4]  # up=2, left=4


def test_legal_moves_center_has_up_to_four_options():
    # blank in the middle of a 3x3 board (idx4): up/down/left/right all in bounds
    state = (1, 2, 3, 4, 0, 5, 6, 7, 8)
    assert sorted(legal_moves(state, width=3, height=3)) == [1, 3, 5, 7]


def test_apply_move_swaps_blank_and_target():
    state = goal_state(3, 2)  # (1,2,3,4,5,0), blank at idx5
    new_state = apply_move(state, 4)  # swap with idx4 (value 5)
    assert new_state == (1, 2, 3, 4, 0, 5)
    assert blank_index(new_state) == 4


def test_bfs_distances_goal_is_zero():
    distances = bfs_distances(3, 2)
    assert distances[goal_state(3, 2)] == 0


def test_bfs_distances_one_move_away_is_one():
    distances = bfs_distances(3, 2)
    start = goal_state(3, 2)
    for move in legal_moves(start, width=3, height=2):
        assert distances[apply_move(start, move)] == 1


def test_bfs_distances_hand_verified_two_move_scramble():
    # goal -(move 2: up)-> (1,2,0,4,5,3) -(move 1: left)-> (1,0,2,4,5,3)
    # differs from goal in 3 positions (idx1,2,5) -- a single move only
    # ever changes 2 positions, so distance can't be 1; a valid 2-move
    # path exists, so distance is exactly 2.
    width, height = 3, 2
    distances = bfs_distances(width, height)
    start = goal_state(width, height)
    state1 = apply_move(start, 2)
    state2 = apply_move(state1, 1)
    assert state2 == (1, 0, 2, 4, 5, 3)
    assert distances[state2] == 2


def test_two_distinct_same_length_paths_reach_identical_state():
    # Found by exhaustive search from the goal on the 3x2 board: the
    # 3-move sequences (2,1,2) and (4,5,2) are different move orders that
    # both land on (1,2,0,4,5,3) -- a genuine transposition, the
    # higher-density analog of connect_four_engine's column-order test.
    width, height = 3, 2
    start = goal_state(width, height)

    def _play(moves):
        state = start
        for m in moves:
            state = apply_move(state, m)
        return state

    state_a = _play((2, 1, 2))
    state_b = _play((4, 5, 2))
    assert state_a == state_b == (1, 2, 0, 4, 5, 3)


def test_generate_puzzles_returns_states_at_exact_target_distance():
    width, height = 3, 2
    distances = bfs_distances(width, height)
    instances = generate_puzzles(n=5, seed=0, width=width, height=height, target_distance=3, distances=distances)
    assert len(instances) == 5
    ids = {inst.id for inst in instances}
    assert len(ids) == 5  # distinct ids
    for inst in instances:
        assert distances[inst.start_state] == 3
        assert inst.target_distance == 3
