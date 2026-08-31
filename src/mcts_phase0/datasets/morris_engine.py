"""Three Men's Morris rules engine + exact forced-win solver.

Board: a 3x3 grid (indices 0-8, row-major), tic-tac-toe's own 8 win-lines.
Two phases: placement (each side alternately places 3 pieces, exactly like
Connect Four's drops -- irreversible) then movement (each side alternately
slides one of its own pieces to an adjacent empty cell, forever -- fully
reversible: sliding out and back returns the exact prior board). The board
tuple is already canonical, same principle as connect_four_engine.py's.

Adjacency is king-move on the 3x3 grid (corners: 3 neighbors, edges: 5,
center: 8). Since placement always uses exactly 6 of the 9 cells, the
movement phase always has exactly 3 empty cells, for the rest of the game.

Rule choice, stated explicitly: if a side has zero legal moves (every one
of its pieces is fully boxed in by occupied neighbors), that side loses.
This is the only way the otherwise-endless movement phase terminates
outside of a completed line -- the direct analog of Connect Four's
`is_full` board-fills case, just for a game that never structurally fills.

`solve_forced_win` mirrors connect_four_engine.py's own alternating
hero/opponent recursion exactly, over the movement phase's move generator.
"""

from __future__ import annotations

from dataclasses import dataclass

X, O = "X", "O"

_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

_ADJACENCY = {
    0: (1, 3, 4),
    1: (0, 2, 3, 4, 5),
    2: (1, 4, 5),
    3: (0, 1, 4, 6, 7),
    4: (0, 1, 2, 3, 5, 6, 7, 8),
    5: (1, 2, 4, 7, 8),
    6: (3, 4, 7),
    7: (3, 4, 5, 6, 8),
    8: (4, 5, 7),
}


def opponent(player: str) -> str:
    return O if player == X else X


def make_empty_board() -> tuple[str | None, ...]:
    return (None,) * 9


def check_win(board: tuple[str | None, ...], player: str) -> bool:
    return any(all(board[c] == player for c in line) for line in _LINES)


def is_legal_placement(board: tuple[str | None, ...], cell: int) -> bool:
    return board[cell] is None


def apply_placement(board: tuple[str | None, ...], cell: int, player: str) -> tuple[str | None, ...]:
    return board[:cell] + (player,) + board[cell + 1:]


def replay_placements(pre_moves: tuple[int, ...]) -> tuple[str | None, ...]:
    """Apply 6 alternating placements from an empty board, starting with X.
    Raises ValueError on an occupied cell or a completed line during
    placement -- Morris puzzles are about the movement phase, so placement
    must land on a not-yet-won position (mirrors connect_four_engine.replay's
    own guard against pre_moves that already contain a win)."""
    board = make_empty_board()
    player = X
    for cell in pre_moves:
        if not is_legal_placement(board, cell):
            raise ValueError(f"illegal placement: cell {cell} already occupied")
        board = apply_placement(board, cell, player)
        if check_win(board, player):
            raise ValueError("pre_moves already contains a completed win during placement")
        player = opponent(player)
    return board


def legal_moves(board: tuple[str | None, ...], player: str) -> list[tuple[int, int]]:
    """(from_cell, to_cell) pairs: each of player's pieces to each of its
    empty adjacent cells."""
    moves = []
    for cell in range(9):
        if board[cell] != player:
            continue
        for neighbor in _ADJACENCY[cell]:
            if board[neighbor] is None:
                moves.append((cell, neighbor))
    return moves


def apply_move(board: tuple[str | None, ...], from_cell: int, to_cell: int, player: str) -> tuple[str | None, ...]:
    board = board[:from_cell] + (None,) + board[from_cell + 1:]
    board = board[:to_cell] + (player,) + board[to_cell + 1:]
    return board


def canonical_state(board: tuple[str | None, ...], to_move: str) -> tuple:
    return (board, to_move)


def solve_forced_win(
    board: tuple[str | None, ...], side_to_move: str, hero: str, plies_left: int,
    cache: dict | None = None, stats: dict | None = None,
) -> bool:
    """Can `hero` force a win within exactly `plies_left` movement-phase
    plies from here, with `side_to_move` playing next? Mirrors
    connect_four_engine.solve_forced_win's alternating recursion exactly."""
    if cache is None:
        cache = {}
    if plies_left <= 0:
        return False

    key = (board, side_to_move, plies_left)
    if key in cache:
        if stats is not None:
            stats["hits"] = stats.get("hits", 0) + 1
        return cache[key]
    if stats is not None:
        stats["misses"] = stats.get("misses", 0) + 1

    moves = legal_moves(board, side_to_move)
    if not moves:
        # side_to_move is boxed in -- it loses, so this is a win for hero
        # iff side_to_move is the opponent.
        result = side_to_move != hero
        cache[key] = result
        return result

    if side_to_move == hero:
        result = False
        for from_cell, to_cell in moves:
            new_board = apply_move(board, from_cell, to_cell, hero)
            if check_win(new_board, hero):
                result = True
                break
            if solve_forced_win(new_board, opponent(hero), hero, plies_left - 1, cache, stats):
                result = True
                break
    else:
        result = True
        for from_cell, to_cell in moves:
            new_board = apply_move(board, from_cell, to_cell, side_to_move)
            if check_win(new_board, side_to_move):
                result = False  # opponent escaping into their own win defeats hero's forced win
                break
            if not solve_forced_win(new_board, hero, hero, plies_left - 1, cache, stats):
                result = False
                break

    cache[key] = result
    return result


def shortest_forced_win(
    board: tuple[str | None, ...], hero: str, max_plies: int, cache: dict | None = None,
) -> int | None:
    """Minimal plies_left for which solve_forced_win is true, or None if no
    forced win exists within max_plies. hero always moves first, so the
    answer (when it exists) is always odd."""
    if cache is None:
        cache = {}
    for p in range(1, max_plies + 1):
        if solve_forced_win(board, hero, hero, p, cache):
            return p
    return None


def preserving_first_moves(
    board: tuple[str | None, ...], hero: str, plies_left: int, cache: dict | None = None,
) -> list[tuple[int, int]]:
    """Moves hero can play right now that either win immediately or keep a
    forced win alive with one fewer ply remaining."""
    if cache is None:
        cache = {}
    result = []
    for from_cell, to_cell in legal_moves(board, hero):
        new_board = apply_move(board, from_cell, to_cell, hero)
        if check_win(new_board, hero):
            result.append((from_cell, to_cell))
            continue
        if plies_left > 1 and solve_forced_win(new_board, opponent(hero), hero, plies_left - 1, cache):
            result.append((from_cell, to_cell))
    return result


@dataclass(frozen=True)
class MorrisInstance:
    id: str
    pre_moves: tuple[int, ...]
    to_move: str
    k_plies: int


def generate_puzzles(
    n: int, seed: int, k_plies: int, max_attempts: int = 200_000,
) -> list[MorrisInstance]:
    """Seeded rejection sampling over random 6-cell placement orderings:
    reject on an illegal/early-win placement sequence (via replay_placements),
    keep those whose movement-phase start has shortest_forced_win exactly
    equal to k_plies -- the same exact-difficulty-by-construction principle
    as connect_four's own puzzle generation."""
    import random

    rng = random.Random(seed)
    found = []
    seen_pre_moves = set()
    cells = list(range(9))
    for _ in range(max_attempts):
        if len(found) >= n:
            break
        pre_moves = tuple(rng.sample(cells, 6))
        if pre_moves in seen_pre_moves:
            continue
        seen_pre_moves.add(pre_moves)
        try:
            board = replay_placements(pre_moves)
        except ValueError:
            continue
        hero = X if len(pre_moves) % 2 == 0 else O
        if shortest_forced_win(board, hero, max_plies=k_plies) == k_plies:
            found.append(MorrisInstance(
                id=f"morris_k{k_plies}_{len(found)}", pre_moves=pre_moves, to_move=hero, k_plies=k_plies,
            ))
    if len(found) < n:
        raise ValueError(f"only found {len(found)}/{n} puzzles at k_plies={k_plies} within {max_attempts} attempts")
    return found
