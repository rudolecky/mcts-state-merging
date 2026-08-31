"""Verification for connect_four_engine.py, no model/LLM involved:
- gravity / legal-move bookkeeping
- check_win in all 4 directions
- canonicalization: two different move orders that transpose to the same
  board must produce identical (board, to_move) keys
- solve_forced_win correctness on a hand-built forced-win position AND a
  hand-built negative control (a solver that says "yes" to everything would
  still pass every positive test but fail this one)
"""

from mcts_phase0.datasets.connect_four_engine import (
    O,
    X,
    apply_move,
    canonical_state,
    check_win,
    is_legal_move,
    make_empty_board,
    opponent,
    preserving_first_moves,
    replay,
    shortest_forced_win,
    solve_forced_win,
)


def _board_from_moves(moves_by_column: dict[int, str], width: int) -> tuple:
    board = list(make_empty_board(width))
    for col, stack in moves_by_column.items():
        board[col] = tuple(stack)
    return tuple(board)


def test_gravity_stacks_bottom_up_and_rejects_full_column():
    board = make_empty_board(3)
    board = apply_move(board, 0, X)
    board = apply_move(board, 0, O)
    assert board[0] == (X, O)
    assert is_legal_move(board, 0, height=2) is False  # column full at height 2
    assert is_legal_move(board, 1, height=2) is True


def test_check_win_horizontal():
    board = _board_from_moves({0: [X], 1: [X], 2: [X], 3: [X]}, width=5)
    assert check_win(board, 3, 0, X) is True
    assert check_win(board, 2, 0, X) is True  # any of the 4 cells detects it


def test_check_win_vertical():
    board = _board_from_moves({0: [X, X, X, X]}, width=3)
    assert check_win(board, 0, 3, X) is True


def test_check_win_diagonal_up():
    # "/" diagonal: (0,0) (1,1) (2,2) (3,3)
    board = _board_from_moves(
        {0: [X], 1: [O, X], 2: [O, O, X], 3: [O, O, O, X]}, width=4
    )
    assert check_win(board, 3, 3, X) is True


def test_check_win_diagonal_down():
    # "\" diagonal: (0,3) (1,2) (2,1) (3,0)
    board = _board_from_moves(
        {0: [O, O, O, X], 1: [O, O, X], 2: [O, X], 3: [X]}, width=4
    )
    assert check_win(board, 3, 0, X) is True


def test_check_win_false_on_three_in_a_row():
    board = _board_from_moves({0: [X], 1: [X], 2: [X]}, width=5)
    assert check_win(board, 2, 0, X) is False


def test_transposed_move_orders_reach_identical_canonical_state():
    # Independent columns (0/1 vs 5), same-player moves reordered relative
    # to each other -- a genuine transposition, not just "different game."
    width, height = 6, 2
    board_a = replay((0, 5, 1, 5), width, height)
    board_b = replay((1, 5, 0, 5), width, height)
    assert board_a == board_b
    assert canonical_state(board_a, X) == canonical_state(board_b, X)


def test_different_side_to_move_is_not_the_same_state():
    board = replay((0, 1), width=4, height=2)
    assert canonical_state(board, X) != canonical_state(board, O)


def test_solve_forced_win_true_on_hand_built_double_threat_setup():
    # row0: _ X X _ _  (cols 1,2 already X; playing col3 opens a double
    # threat at col0/col4 that O can only block one side of)
    width, height = 5, 2
    board = _board_from_moves({1: [X], 2: [X]}, width=width)
    assert solve_forced_win(board, X, X, 3, height, cache={}) is True
    # X cannot win in exactly 1 ply -- the threat doesn't exist yet, it has
    # to be created first
    assert solve_forced_win(board, X, X, 1, height, cache={}) is False
    assert shortest_forced_win(board, X, max_plies=3, height=height) == 3


def test_solve_forced_win_false_negative_control_on_empty_board():
    board = make_empty_board(5)
    assert solve_forced_win(board, X, X, 1, height=4, cache={}) is False
    assert shortest_forced_win(board, X, max_plies=3, height=4) is None


def test_preserving_first_moves_isolates_the_real_setup_move():
    width, height = 5, 2
    board = _board_from_moves({1: [X], 2: [X]}, width=width)
    preserving = preserving_first_moves(board, X, plies_left=3, height=height, cache={})
    assert 3 in preserving  # col3 creates the double threat
    assert 0 not in preserving  # doesn't build toward anything on its own


def test_opponent_helper():
    assert opponent(X) == O
    assert opponent(O) == X
