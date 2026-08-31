"""N-puzzle (sliding-tile) rules engine + exact BFS distance oracle.

State representation: a flat tuple of length `width*height`, values
`1..width*height-1` for tiles and `0` for the blank -- so the state tuple
itself is already the canonical encoding of a position, exactly like
connect_four_engine.py's board tuple: any two move sequences that reach the
same arrangement produce the byte-identical tuple, no separate
normalization step.

`bfs_distances` answers "what is the exact optimal-solution length from
every reachable state" via a single reverse BFS from the goal over the
whole (solvable) state graph -- computed once per board size rather than
per query, the direct analog of connect_four_engine's
solve_forced_win/shortest_forced_win but for a domain small enough to
enumerate exhaustively instead of needing per-position game-tree search.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


def goal_state(width: int, height: int) -> tuple[int, ...]:
    return tuple(range(1, width * height)) + (0,)


def blank_index(state: tuple[int, ...]) -> int:
    return state.index(0)


def legal_moves(state: tuple[int, ...], width: int, height: int) -> list[int]:
    """Flat indices the blank can swap with (its in-bounds up/down/left/right neighbors)."""
    idx = blank_index(state)
    row, col = divmod(idx, width)
    moves = []
    if row > 0:
        moves.append(idx - width)
    if row < height - 1:
        moves.append(idx + width)
    if col > 0:
        moves.append(idx - 1)
    if col < width - 1:
        moves.append(idx + 1)
    return moves


def apply_move(state: tuple[int, ...], target_idx: int) -> tuple[int, ...]:
    idx = blank_index(state)
    new_state = list(state)
    new_state[idx], new_state[target_idx] = new_state[target_idx], new_state[idx]
    return tuple(new_state)


def bfs_distances(width: int, height: int) -> dict[tuple[int, ...], int]:
    """Exact optimal-solution length from every solvable state, via one
    reverse BFS from the goal (sliding-puzzle moves are their own inverse,
    so a forward BFS from the goal already gives distances-to-goal for
    every other state)."""
    start = goal_state(width, height)
    distances = {start: 0}
    frontier = deque([start])
    while frontier:
        state = frontier.popleft()
        d = distances[state]
        for idx in legal_moves(state, width, height):
            neighbor = apply_move(state, idx)
            if neighbor not in distances:
                distances[neighbor] = d + 1
                frontier.append(neighbor)
    return distances


@dataclass(frozen=True)
class PuzzleInstance:
    id: str
    start_state: tuple[int, ...]
    width: int
    height: int
    target_distance: int


def generate_puzzles(
    n: int, seed: int, width: int, height: int, target_distance: int,
    distances: dict[tuple[int, ...], int] | None = None,
) -> list[PuzzleInstance]:
    """Seeded sample of `n` states whose exact optimal-solution length is
    `target_distance`. Pass a precomputed `distances` map when sweeping
    several target distances on the same board size, to avoid recomputing
    the BFS per call."""
    if distances is None:
        distances = bfs_distances(width, height)
    candidates = sorted(s for s, d in distances.items() if d == target_distance)
    rng = random.Random(seed)
    chosen = rng.sample(candidates, n)
    return [
        PuzzleInstance(
            id=f"sp_{width}x{height}_d{target_distance}_{i}",
            start_state=state, width=width, height=height, target_distance=target_distance,
        )
        for i, state in enumerate(chosen)
    ]
