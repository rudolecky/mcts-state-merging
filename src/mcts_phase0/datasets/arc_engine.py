"""A small, hand-built grid-transform DSL for a subset of real ARC-AGI-1
tasks -- not `michaelhodel/arc-dsl` (deliberately: this pilot's question is
"does merging show anything here at all," not "is our DSL state-of-the-art
expressive"; see the plan file for the tradeoff).

A "state" is a tuple of grids -- one per training input, all transformed by
the same program so far -- exactly mirroring how a Blocksworld state is the
current world-configuration, not the move history. This is the thing that
gets keyed/merged: two different primitive orderings that reach the same
grids-tuple are a genuine transposition, the direct ARC analogue of
Blocksworld's independent-block-pair transpositions.

Structural primitives (rot90/180/270, flip_h, flip_v, transpose) alone form
a tiny, quickly-exhausted group (at most 8 reachable grids from any start) --
uninformative, the same failure mode Blocksworld hit at num_blocks=4. The
`recolor(c1, c2)` family (grounded per-state, one action per ordered pair of
colors actually present -- no STRIPS-style precondition needed beyond that)
adds real combinatorial richness: recoloring color A then color B reaches
the same grids as B then A, mirroring Blocksworld's own commuting-
independent-relocations structure, and the reachable space grows with how
many colors are in play rather than staying a fixed tiny group.

Grids are tuples of tuples of ints -- immutable, hashable, directly usable
as dict keys with no packaging step, the same "the state representation
already is the key" principle every engine in this project follows.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

Grid = tuple  # tuple[tuple[int, ...], ...]
State = tuple  # tuple[Grid, ...] -- one grid per training example

_STRUCTURAL_MOVES = ("rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose")


def _rot90(grid: Grid) -> Grid:
    return tuple(zip(*grid[::-1]))


def _apply_grid(grid: Grid, move) -> Grid:
    if move == "rot90":
        return _rot90(grid)
    if move == "rot180":
        return _rot90(_rot90(grid))
    if move == "rot270":
        return _rot90(_rot90(_rot90(grid)))
    if move == "flip_h":
        return tuple(row[::-1] for row in grid)
    if move == "flip_v":
        return grid[::-1]
    if move == "transpose":
        return tuple(zip(*grid))
    if isinstance(move, tuple) and move[0] == "recolor":
        _, c1, c2 = move
        return tuple(tuple(c2 if v == c1 else v for v in row) for row in grid)
    raise ValueError(f"unknown move: {move!r}")


def apply_move(state: State, move) -> State:
    return tuple(_apply_grid(grid, move) for grid in state)


def legal_moves(state: State) -> list:
    colors = set()
    for grid in state:
        for row in grid:
            colors.update(row)
    moves = list(_STRUCTURAL_MOVES)
    moves.extend(("recolor", c1, c2) for c1, c2 in itertools.permutations(sorted(colors), 2))
    return moves


def is_goal(state: State, target: State) -> bool:
    return state == target


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train_inputs: State
    train_outputs: State


def load_task(task_json: dict, task_id: str = "") -> ArcTask:
    train_inputs = tuple(tuple(tuple(row) for row in pair["input"]) for pair in task_json["train"])
    train_outputs = tuple(tuple(tuple(row) for row in pair["output"]) for pair in task_json["train"])
    return ArcTask(task_id=task_id, train_inputs=train_inputs, train_outputs=train_outputs)


def same_shape_task(task: ArcTask) -> bool:
    """This primitive set never changes a grid's total cell count in a way
    that could match a differently-shaped target (no tiling/cropping/scaling
    primitives) -- tasks where some training pair's input/output shapes
    differ are structurally unreachable and filtered out before the
    brute-force oracle even looks at them."""
    return all(
        (len(i), len(i[0]) if i else 0) == (len(o), len(o[0]) if o else 0)
        for i, o in zip(task.train_inputs, task.train_outputs)
    )


def find_short_program(task: ArcTask, max_depth: int) -> list | None:
    """Brute-force exhaustive search (breadth-first over program length) for
    the shortest sequence of moves solving `task`, up to `max_depth`. This
    project's own oracle for this domain -- no off-the-shelf one exists here
    the way pyperplan's BFS served Blocksworld. Returns None if no program of
    length <= max_depth solves it (a real, expected outcome for most tasks
    given this primitive set's limited expressiveness -- not an error)."""
    start = task.train_inputs
    if is_goal(start, task.train_outputs):
        return []
    frontier = [(start, [])]
    seen = {start}
    for _ in range(max_depth):
        next_frontier = []
        for state, program in frontier:
            for move in legal_moves(state):
                new_state = apply_move(state, move)
                new_program = program + [move]
                if is_goal(new_state, task.train_outputs):
                    return new_program
                if new_state not in seen:
                    seen.add(new_state)
                    next_frontier.append((new_state, new_program))
        frontier = next_frontier
        if not frontier:
            break
    return None
