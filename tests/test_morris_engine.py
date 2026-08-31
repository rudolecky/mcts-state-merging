"""Verification for morris_engine.py, no model/LLM involved:
- win-line / adjacency correctness
- replay_placements rejecting an occupied cell and a premature win
- a genuine movement-phase transposition (two different move sequences
  reaching the identical state -- the premise this domain's whole
  experiment turns on, same as the 8-puzzle's)
- solve_forced_win correctness on a hand-verified forced-win position AND
  a negative control, mirroring test_connect_four_engine.py's pairing
"""

from mcts_phase0.datasets.morris_engine import (
    O,
    X,
    apply_move,
    apply_placement,
    check_win,
    generate_puzzles,
    is_legal_placement,
    legal_moves,
    make_empty_board,
    opponent,
    replay_placements,
    shortest_forced_win,
    solve_forced_win,
)


def test_opponent_helper():
    assert opponent(X) == O
    assert opponent(O) == X


def test_check_win_row():
    board = (X, X, X, None, None, None, None, None, None)
    assert check_win(board, X) is True
    assert check_win(board, O) is False


def test_check_win_diagonal():
    board = (X, None, None, None, X, None, None, None, X)
    assert check_win(board, X) is True


def test_check_win_false_on_two_in_a_row():
    board = (X, X, None, None, None, None, None, None, None)
    assert check_win(board, X) is False


def test_replay_placements_rejects_occupied_cell():
    try:
        replay_placements((0, 0, 1, 2, 3, 4))
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_replay_placements_rejects_premature_win():
    # X placements at 0,1,2 complete a row on X's 3rd placement (index 4: 0,1,2,X,Y -- placements alternate X,O,X,O,X,O)
    try:
        replay_placements((0, 3, 1, 4, 2, 5))  # X: 0,1,2 -- wins on placing cell 2
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_replay_placements_produces_expected_board():
    # placements alternate X,O,X,O,X,O -> cell0=X,cell4=O,cell8=X,cell2=O,cell6=X,cell5=O
    board = replay_placements((0, 4, 8, 2, 6, 5))
    assert board == (X, None, O, None, O, O, X, None, X)


def test_legal_moves_only_to_empty_adjacent_cells():
    board = (X, O, O, None, None, None, O, X, X)
    moves = legal_moves(board, X)
    # X pieces at 0,7,8; empty cells are 3,4,5
    assert sorted(moves) == [(0, 3), (0, 4), (7, 3), (7, 4), (7, 5), (8, 4), (8, 5)]


def test_apply_move_slides_piece():
    board = (X, O, O, None, None, None, O, X, X)
    new_board = apply_move(board, 7, 4, X)
    assert new_board == (X, O, O, None, X, None, O, None, X)


def test_two_distinct_same_length_paths_reach_identical_state():
    # Found by exhaustive search from a fixed placement: the 3-move
    # sequences ((0,1),(4,0),(1,4)) and ((0,3),(4,0),(3,4)) both land on
    # (O,None,O,None,X,O,X,None,X) with O to move -- a genuine
    # transposition, no intermediate state a win for either side.
    start = replay_placements((0, 4, 8, 2, 6, 5))

    def _play(moves):
        board = start
        mover = X
        for from_cell, to_cell in moves:
            board = apply_move(board, from_cell, to_cell, mover)
            mover = opponent(mover)
        return board, mover

    result_a = _play(((0, 1), (4, 0), (1, 4)))
    result_b = _play(((0, 3), (4, 0), (3, 4)))
    assert result_a == result_b == ((O, None, O, None, X, O, X, None, X), O)


def test_solve_forced_win_true_on_hand_verified_position():
    # X pieces at 0,7,8; moving piece7 -> 4 completes the (0,4,8) diagonal.
    board = (X, O, O, None, None, None, O, X, X)
    assert solve_forced_win(board, X, X, 1, cache={}) is True
    assert shortest_forced_win(board, X, max_plies=3) == 1


def test_solve_forced_win_false_negative_control():
    # No X piece can reach a winning cell in a single move from this board.
    board = replay_placements((0, 4, 8, 2, 6, 5))
    assert solve_forced_win(board, X, X, 1, cache={}) is False


def test_generate_puzzles_returns_exact_k_plies_instances():
    instances = generate_puzzles(n=2, seed=0, k_plies=1)
    assert len(instances) == 2
    ids = {inst.id for inst in instances}
    assert len(ids) == 2
    for inst in instances:
        assert inst.k_plies == 1
        board = replay_placements(inst.pre_moves)
        assert shortest_forced_win(board, inst.to_move, max_plies=1) == 1
