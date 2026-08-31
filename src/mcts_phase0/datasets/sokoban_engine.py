"""Sokoban rules engine + exact BFS distance oracle, one fixed small level.

Level: a 10x10 grid (row-major flat indices, width=10), an outer wall
border, an 8x8 open interior room (64 floor cells), one box, one goal at
the room's center. Like morris_engine.py's fixed 3x3 board, this is one
specific hand-designed level, not a generic level-loader -- there's only
one level to test. Sized (checked directly, not guessed) so the
pull-reachable state space is rich enough for a real budget sweep: a 6x6
grid (16 floor cells) only reaches 60 states total before saturating,
while 10x10 (64 floor cells) reaches 2,268 states at a max optimal
distance of 21 -- comparable order of magnitude to the other domains'
state spaces.

State: `(player_cell, tuple(sorted(box_cells)))` -- already canonical by
construction (sorted tuple), same principle as every other engine here.
Boxes are kept as a tuple rather than hardcoded to exactly one, since
that's the natural Sokoban state shape regardless of box count -- this
level still ships with exactly one.

The real game only ever PUSHES (legal_moves/apply_move below): walking
onto a box's cell pushes it one further step in the same direction, if
that cell is free; walking onto any other free floor cell just walks.
Pushing is directional and not freely reversible -- undoing a push needs
walking around to the opposite side and pushing back, which isn't always
possible (a box against a wall in the wrong orientation can't be pulled
back at all). This is the one property distinguishing this domain from
the 8-puzzle (every move trivially reversible) and Morris (adversarial,
low transposition density) -- partial, position-dependent reversibility.

`_pull_moves`/`_apply_pull` are generation-only, private, and never used by
the search module: a pull is the literal inverse of a push (player steps
to a free cell; if a box was directly behind it, opposite the step
direction, that box gets dragged one cell closer to where the player used
to be). `bfs_distances` runs a single multi-source BFS from every "solved"
state (box on the goal, player at any other free cell -- real Sokoban's
win condition ignores player position) via pull-moves, giving the exact
optimal PUSH-distance for every state reachable by pulling, the same
"moves are invertible so BFS-from-goal gives true distances" trick
sliding_puzzle_engine.bfs_distances uses, just with pull instead of
blank-swap as the traversal function.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

WIDTH = 10
HEIGHT = 10
GOAL = 55  # row 5, col 5 -- the room's center

_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right


def _is_wall(cell: int) -> bool:
    row, col = divmod(cell, WIDTH)
    return row == 0 or row == HEIGHT - 1 or col == 0 or col == WIDTH - 1


FLOOR = tuple(c for c in range(WIDTH * HEIGHT) if not _is_wall(c))


def _step(cell: int, direction: tuple[int, int]) -> int | None:
    row, col = divmod(cell, WIDTH)
    new_row, new_col = row + direction[0], col + direction[1]
    if new_row < 0 or new_row >= HEIGHT or new_col < 0 or new_col >= WIDTH:
        return None
    return new_row * WIDTH + new_col


def make_state(player: int, boxes: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    return (player, tuple(sorted(boxes)))


def is_solved(boxes: tuple[int, ...]) -> bool:
    return tuple(sorted(boxes)) == (GOAL,)


def legal_moves(state: tuple[int, tuple[int, ...]]) -> list[tuple[int, int]]:
    player, boxes = state
    box_set = set(boxes)
    moves = []
    for direction in _DIRECTIONS:
        dest = _step(player, direction)
        if dest is None or _is_wall(dest):
            continue
        if dest not in box_set:
            moves.append(direction)  # walk
            continue
        beyond = _step(dest, direction)
        if beyond is not None and not _is_wall(beyond) and beyond not in box_set:
            moves.append(direction)  # push
    return moves


def apply_move(state: tuple[int, tuple[int, ...]], direction: tuple[int, int]) -> tuple[int, tuple[int, ...]]:
    player, boxes = state
    box_set = set(boxes)
    dest = _step(player, direction)
    if dest in box_set:
        beyond = _step(dest, direction)
        box_set.remove(dest)
        box_set.add(beyond)
    return make_state(dest, tuple(box_set))


def _pull_moves(state: tuple[int, tuple[int, ...]]) -> list[tuple[int, int]]:
    player, boxes = state
    box_set = set(boxes)
    moves = []
    for direction in _DIRECTIONS:
        dest = _step(player, direction)
        if dest is None or _is_wall(dest) or dest in box_set:
            continue
        moves.append(direction)  # always a legal walk; may also pull, see _apply_pull
    return moves


def _apply_pull(state: tuple[int, tuple[int, ...]], direction: tuple[int, int]) -> tuple[int, tuple[int, ...]]:
    player, boxes = state
    box_set = set(boxes)
    dest = _step(player, direction)
    behind = _step(player, (-direction[0], -direction[1]))
    if behind is not None and behind in box_set:
        box_set.remove(behind)
        box_set.add(player)
    return make_state(dest, tuple(box_set))


def bfs_distances() -> dict[tuple[int, tuple[int, ...]], int]:
    distances = {}
    frontier = deque()
    for player in FLOOR:
        if player == GOAL:
            continue
        start = make_state(player, (GOAL,))
        distances[start] = 0
        frontier.append(start)
    while frontier:
        state = frontier.popleft()
        d = distances[state]
        for direction in _pull_moves(state):
            neighbor = _apply_pull(state, direction)
            if neighbor not in distances:
                distances[neighbor] = d + 1
                frontier.append(neighbor)
    return distances


@dataclass(frozen=True)
class SokobanInstance:
    id: str
    start_state: tuple[int, tuple[int, ...]]
    target_distance: int


def generate_puzzles(
    n: int, seed: int, target_distance: int,
    distances: dict[tuple[int, tuple[int, ...]], int] | None = None,
) -> list[SokobanInstance]:
    import random

    if distances is None:
        distances = bfs_distances()
    candidates = sorted(s for s, d in distances.items() if d == target_distance)
    rng = random.Random(seed)
    chosen = rng.sample(candidates, n)
    return [
        SokobanInstance(id=f"sok_d{target_distance}_{i}", start_state=state, target_distance=target_distance)
        for i, state in enumerate(chosen)
    ]
