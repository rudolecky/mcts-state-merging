"""2x2x2 Rubik's Cube (Pocket Cube) engine -- 8 corners, no edges, no fixed
centers. Full-fidelity representation: state is always kept in
"reference-corner-fixed" canonical form (corner id 0 always sits at
position 0 with identity orientation), matching the standard, well-known
7! x 3^6 = 3,674,160-reachable-state count for this puzzle.

Design rationale (see cheerful-jumping-moler.md's plan): rather than
hand-deriving per-face orientation-delta rules (the standard "0/1/2
clockwise twist" convention, whose transformation under a *whole-cube*
rotation is easy to get subtly wrong), every corner's orientation is
tracked as a full element of the 24-element cube rotation group (a 3x3
signed-permutation matrix, represented here as (axis-permutation,
signs)), and every operation -- a face turn, a whole-cube rotation,
canonicalization -- is the *same* kind of operation: compose a rotation
onto the current state. Correctness reduces to group-theoretic properties
(closure, inverses, associativity) that are mechanically checkable, not to
manually-reasoned sign/parity case analysis.

Positions 0-7 are indexed by bits (r, u, f) -- pos = 4*r + 2*u + f -- and
correspond to coordinates (2r-1, 2u-1, 2f-1) in {-1,+1}^3. Corner id k's
home position is position k (so "corner 0" is the reference/anchor
corner). A state is (perm, orient): perm[p] = which corner id currently
sits at position p; orient[p] = the accumulated rotation (relative to a
freshly-placed, never-moved corner) of whatever corner currently sits at
p. Applying a move M to the corner at position p produces
compose(M, orient[p]) at the new position -- this is exactly "this piece
has now undergone one more rigid rotation," which is why treating
orientation as a full rotation-group element (not a reduced 0/1/2 int)
makes the composition rule trivial and uniform.

Canonicalization after every move: since the corner currently at position
q = perm.index(0) was carried there from position 0 by exactly the net
rotation recorded in orient[q] (by construction of the composition rule
above), inverse(orient[q]) is *always* the whole-cube rotation that sends
q back to 0 with identity orientation -- apply it to every position at
once. This keeps the state always in canonical form and is what reduces
the space from the raw 8! x 3^8 (or 8! x 3^7 restricted to reachable
combinations) down to the true 3,674,160.
"""

from __future__ import annotations

import itertools
import random
from collections import deque
from dataclasses import dataclass

Rotation = tuple[tuple[int, int, int], tuple[int, int, int]]  # (axis-perm, signs)


def _permutation_sign(perm: tuple[int, int, int]) -> int:
    inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j])
    return 1 if inversions % 2 == 0 else -1


def _all_rotations() -> list[Rotation]:
    rotations = []
    for axis in itertools.permutations((0, 1, 2)):
        for signs in itertools.product((1, -1), repeat=3):
            det = _permutation_sign(axis) * signs[0] * signs[1] * signs[2]
            if det == 1:
                rotations.append((axis, signs))
    return rotations


ROTATIONS_24: list[Rotation] = _all_rotations()
IDENTITY: Rotation = ((0, 1, 2), (1, 1, 1))
assert len(ROTATIONS_24) == 24
assert IDENTITY in ROTATIONS_24


def apply_rotation_to_coord(rot: Rotation, coord: tuple[int, int, int]) -> tuple[int, int, int]:
    axis, signs = rot
    return tuple(signs[i] * coord[axis[i]] for i in range(3))


def compose(a: Rotation, b: Rotation) -> Rotation:
    """a after b: apply_rotation_to_coord(compose(a, b), v) ==
    apply_rotation_to_coord(a, apply_rotation_to_coord(b, v))."""
    axis_a, sign_a = a
    axis_b, sign_b = b
    new_axis = tuple(axis_b[axis_a[i]] for i in range(3))
    new_sign = tuple(sign_a[i] * sign_b[axis_a[i]] for i in range(3))
    return (new_axis, new_sign)


def inverse(rot: Rotation) -> Rotation:
    axis, signs = rot
    axis_inv = [0, 0, 0]
    sign_inv = [0, 0, 0]
    for i in range(3):
        j = axis[i]
        axis_inv[j] = i
        sign_inv[j] = signs[i]
    return (tuple(axis_inv), tuple(sign_inv))


def coord(pos: int) -> tuple[int, int, int]:
    r, u, f = (pos >> 2) & 1, (pos >> 1) & 1, pos & 1
    return (2 * r - 1, 2 * u - 1, 2 * f - 1)


def pos_from_coord(c: tuple[int, int, int]) -> int:
    r, u, f = (c[0] + 1) // 2, (c[1] + 1) // 2, (c[2] + 1) // 2
    return 4 * r + 2 * u + f


def _make_quarter_turn(fixed_axis: int) -> Rotation:
    others = [ax for ax in range(3) if ax != fixed_axis]
    b, c = others
    axis = [0, 0, 0]
    signs = [0, 0, 0]
    axis[fixed_axis], signs[fixed_axis] = fixed_axis, 1
    axis[b], signs[b] = c, 1
    axis[c], signs[c] = b, -1
    return (tuple(axis), tuple(signs))


ROT90: dict[int, Rotation] = {axis: _make_quarter_turn(axis) for axis in range(3)}


@dataclass(frozen=True)
class Move:
    affected: frozenset[int]
    rotation: Rotation


def _make_move(axis: int, layer_sign: int, direction: int) -> Move:
    affected = frozenset(pos for pos in range(8) if coord(pos)[axis] == layer_sign)
    rot = ROT90[axis] if direction == 1 else inverse(ROT90[axis])
    return Move(affected=affected, rotation=rot)


ALL_MOVES: list[Move] = [
    _make_move(axis, layer_sign, direction)
    for axis in range(3)
    for layer_sign in (-1, 1)
    for direction in (1, -1)
]
assert len(ALL_MOVES) == 12


State = tuple[tuple[int, ...], tuple[Rotation, ...]]


def _raw_apply_move(state: State, move: Move) -> State:
    perm, orient = state
    new_perm = list(perm)
    new_orient = list(orient)
    for pos in move.affected:
        new_pos = pos_from_coord(apply_rotation_to_coord(move.rotation, coord(pos)))
        new_perm[new_pos] = perm[pos]
        new_orient[new_pos] = compose(move.rotation, orient[pos])
    return (tuple(new_perm), tuple(new_orient))


def canonicalize(state: State) -> State:
    perm, orient = state
    q = perm.index(0)
    m = inverse(orient[q])
    new_perm = [0] * 8
    new_orient: list[Rotation] = [IDENTITY] * 8
    for pos in range(8):
        new_pos = pos_from_coord(apply_rotation_to_coord(m, coord(pos)))
        new_perm[new_pos] = perm[pos]
        new_orient[new_pos] = compose(m, orient[pos])
    return (tuple(new_perm), tuple(new_orient))


def apply_move(state: State, move: Move) -> State:
    return canonicalize(_raw_apply_move(state, move))


SOLVED: State = canonicalize((tuple(range(8)), tuple([IDENTITY] * 8)))


def legal_moves(state: State) -> list[Move]:
    return ALL_MOVES


def is_goal(state: State) -> bool:
    return state == SOLVED


def scramble(depth: int, rng: random.Random) -> State:
    state = SOLVED
    for _ in range(depth):
        state = apply_move(state, rng.choice(ALL_MOVES))
    return state


def bfs_distances() -> dict[State, int]:
    """Exact optimal-solve distance for every one of the 3,674,160 reachable
    states, via a single BFS from SOLVED. The resulting dict's size is the
    ground-truth cross-check for the whole engine's correctness (see
    test_rubiks_cube_engine.py) -- if it isn't exactly 3,674,160, the
    rotation/canonicalization logic has a bug."""
    dist = {SOLVED: 0}
    frontier = deque([SOLVED])
    while frontier:
        state = frontier.popleft()
        d = dist[state]
        for move in ALL_MOVES:
            new_state = apply_move(state, move)
            if new_state not in dist:
                dist[new_state] = d + 1
                frontier.append(new_state)
    return dist


@dataclass(frozen=True)
class RubiksInstance:
    id: str
    start_state: State
    scramble_distance: int  # exact BFS distance from goal


def generate_puzzles(n: int, seed: int, target_distance: int, distances: dict[State, int]) -> list[RubiksInstance]:
    """Seeded rejection sampling over the precomputed exact-distance table,
    mirroring sliding_puzzle_engine.generate_puzzles /
    sokoban_engine.generate_puzzles exactly."""
    rng = random.Random(seed)
    candidates = [s for s, d in distances.items() if d == target_distance]
    rng.shuffle(candidates)
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} states at target_distance={target_distance}, need {n}")
    return [
        RubiksInstance(id=f"cube_d{target_distance}_{i}", start_state=s, scramble_distance=target_distance)
        for i, s in enumerate(candidates[:n])
    ]
