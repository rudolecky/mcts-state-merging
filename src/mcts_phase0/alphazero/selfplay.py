"""One self-play game: PUCT-guided moves from the empty board (the real
AlphaZero convention -- not the k_plies-controlled puzzle positions used
everywhere else in this project, which are reserved for the eventual
merge-vs-tree evaluation stage), recording a (board, to_move, MCTS visit
policy) example at every ply, back-filled with the game's actual outcome
once it ends. Root exploration noise (Dirichlet) is applied fresh at
*every* real move's search, not just once at the start of the game --
the standard AlphaZero convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..datasets.connect_four_engine import X, apply_move, check_win, is_full, legal_moves, make_empty_board, opponent
from .puct import HEIGHT, PUCTConfig, run_search, visit_policy

WIDTH = 5


@dataclass
class SelfPlayExample:
    board: tuple
    to_move: str
    policy: dict[int, float]  # MCTS visit-count target for the policy head
    outcome: float | None = None  # filled in after the game ends, from this example's own to_move's perspective


def _backfill(examples: list[SelfPlayExample], winner: str | None) -> None:
    for ex in examples:
        if winner is None:
            ex.outcome = 0.5  # draw
        else:
            ex.outcome = 1.0 if ex.to_move == winner else 0.0


def play_game(
    evaluate_fn, budget: int, c_puct: float, temperature_moves: int,
    dirichlet_alpha: float, dirichlet_epsilon: float, rng: np.random.Generator,
    merge_enabled: bool = True,
) -> list[SelfPlayExample]:
    board = make_empty_board(WIDTH)
    to_move = X
    pre_moves: tuple[int, ...] = ()
    examples: list[SelfPlayExample] = []
    ply = 0

    while True:
        config = PUCTConfig(
            merge_enabled=merge_enabled, c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
        )
        graph = run_search(pre_moves, to_move, WIDTH, config, budget, evaluate_fn, rng)
        policy = visit_policy(graph, WIDTH)
        examples.append(SelfPlayExample(board=board, to_move=to_move, policy=policy))

        cols = list(policy.keys())
        probs = np.array([policy[c] for c in cols], dtype=float)
        probs = probs / probs.sum()
        if ply < temperature_moves:
            col = int(rng.choice(cols, p=probs))  # temperature-1 sampling, proportional to visit counts
        else:
            col = cols[int(np.argmax(probs))]  # greedy once the game is far enough along

        new_board = apply_move(board, col, to_move)
        row = len(new_board[col]) - 1
        won = check_win(new_board, col, row, to_move)
        board = new_board
        pre_moves = pre_moves + (col,)
        ply += 1

        if won:
            _backfill(examples, winner=to_move)
            return examples
        if is_full(board, HEIGHT):
            _backfill(examples, winner=None)
            return examples
        to_move = opponent(to_move)


def play_vs_random(evaluate_fn, budget: int, c_puct: float, rng: np.random.Generator, network_plays: str) -> float:
    """One game, network (via greedy PUCT, no temperature/noise) vs. a
    uniformly-random opponent. Returns 1.0/0.5/0.0 from the network's own
    perspective -- the cheap win-rate signal Stage 1 gates on."""
    board = make_empty_board(WIDTH)
    to_move = X
    pre_moves: tuple[int, ...] = ()

    while True:
        moves = legal_moves(board, HEIGHT)
        if to_move == network_plays:
            config = PUCTConfig(merge_enabled=True, c_puct=c_puct, dirichlet_alpha=None)
            graph = run_search(pre_moves, to_move, WIDTH, config, budget, evaluate_fn, rng)
            policy = visit_policy(graph, WIDTH)
            col = max(policy, key=policy.get)
        else:
            col = int(rng.choice(moves))

        new_board = apply_move(board, col, to_move)
        row = len(new_board[col]) - 1
        won = check_win(new_board, col, row, to_move)
        board = new_board
        pre_moves = pre_moves + (col,)

        if won:
            return 1.0 if to_move == network_plays else 0.0
        if is_full(board, HEIGHT):
            return 0.5
        to_move = opponent(to_move)
