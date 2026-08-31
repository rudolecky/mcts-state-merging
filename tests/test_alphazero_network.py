"""Verification for alphazero/network.py -- no training involved, just
checks a randomly-initialized network produces well-formed outputs and
that board encoding/policy masking are correct on hand-built positions.
"""

import torch

from mcts_phase0.alphazero.network import (
    HEIGHT,
    WIDTH,
    ConnectFourNet,
    encode_board,
    evaluate,
    legal_policy,
)
from mcts_phase0.datasets.connect_four_engine import X, O, apply_move, make_empty_board


def test_encode_board_marks_current_mover_on_plane_zero():
    board = make_empty_board(WIDTH)
    board = apply_move(board, 0, X)
    board = apply_move(board, 0, O)
    planes = encode_board(board, to_move=X)
    # column 0 has X at row 0, O at row 1 -- to_move is X, so plane0 marks row0, plane1 marks row1
    assert planes[0, 0, 0].item() == 1.0
    assert planes[1, 0, 0].item() == 0.0
    assert planes[0, 1, 0].item() == 0.0
    assert planes[1, 1, 0].item() == 1.0
    assert planes.sum().item() == 2.0  # exactly the two pieces placed


def test_encode_board_swaps_planes_when_perspective_flips():
    board = make_empty_board(WIDTH)
    board = apply_move(board, 0, X)
    planes_as_x = encode_board(board, to_move=X)
    planes_as_o = encode_board(board, to_move=O)
    assert planes_as_x[0, 0, 0].item() == 1.0  # X's own piece, X's perspective -> plane0
    assert planes_as_o[1, 0, 0].item() == 1.0  # same piece, O's perspective -> plane1 ("opponent's")


def test_legal_policy_masks_illegal_columns_to_zero():
    board = make_empty_board(WIDTH)
    for _ in range(HEIGHT):
        board = apply_move(board, 0, X)  # fill column 0 completely
    logits = torch.zeros(WIDTH)  # uniform logits -- would be uniform probability without masking
    policy = legal_policy(logits, board)
    assert policy[0].item() == 0.0  # column 0 is full, illegal
    assert torch.isclose(policy.sum(), torch.tensor(1.0), atol=1e-5)
    assert all(policy[c].item() > 0 for c in range(1, WIDTH))


def test_network_forward_produces_well_formed_outputs():
    torch.manual_seed(0)
    net = ConnectFourNet(channels=8, num_blocks=1)  # tiny, just checking shapes/ranges
    board = make_empty_board(WIDTH)
    policy, value = evaluate(net, board, to_move=X)
    assert policy.shape == (WIDTH,)
    assert torch.isclose(policy.sum(), torch.tensor(1.0), atol=1e-4)
    assert (policy >= 0).all()
    assert 0.0 <= value <= 1.0
