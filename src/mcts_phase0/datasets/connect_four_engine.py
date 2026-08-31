"""Connect Four rules engine + exact forced-win solver.

Board representation: a tuple of `width` columns, each column a tuple of
"X"/"O" stacked bottom-to-top (empty cells are simply absent, not stored) --
so the board tuple itself is already the canonical encoding of a position:
any two move orders that reach the same stack of pieces in every column
produce the byte-identical tuple, with no separate normalization step. This
is deliberate -- it's what makes `board` usable directly as a ground-truth
"same state" key for the merge-confusion analysis in geometry.py.

`solve_forced_win` answers "can `hero` force a win within `plies_left`
plies, regardless of how the opponent replies" via the standard alternating
game-tree recursion (the same structure as a chess "mate in N" solver):
on hero's own turn, hero needs only ONE move that wins immediately or leads
to a forced win one ply shorter; on the opponent's turn, EVERY legal reply
must still leave hero with a forced win one ply shorter, or the position is
not forced. Memoized on (board, side_to_move, plies_left) via a
caller-supplied cache dict, so puzzle generation and per-move verification
share transposition hits within one call.
"""

from __future__ import annotations

X, O = "X", "O"


def opponent(player: str) -> str:
    return O if player == X else X


def make_empty_board(width: int) -> tuple[tuple[str, ...], ...]:
    return tuple(() for _ in range(width))


def cell(board: tuple[tuple[str, ...], ...], col: int, row: int) -> str | None:
    if col < 0 or col >= len(board):
        return None
    column = board[col]
    if row < 0 or row >= len(column):
        return None
    return column[row]


def is_legal_move(board: tuple[tuple[str, ...], ...], col: int, height: int) -> bool:
    return 0 <= col < len(board) and len(board[col]) < height


def legal_moves(board: tuple[tuple[str, ...], ...], height: int) -> list[int]:
    return [c for c in range(len(board)) if is_legal_move(board, c, height)]


def is_full(board: tuple[tuple[str, ...], ...], height: int) -> bool:
    return not legal_moves(board, height)


def apply_move(board: tuple[tuple[str, ...], ...], col: int, player: str) -> tuple[tuple[str, ...], ...]:
    return tuple(column + (player,) if c == col else column for c, column in enumerate(board))


_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def check_win(board: tuple[tuple[str, ...], ...], col: int, row: int, player: str) -> bool:
    """Does the just-played piece at (col, row) complete 4-in-a-row for player?"""
    for dc, dr in _DIRECTIONS:
        count = 1
        c, r = col + dc, row + dr
        while cell(board, c, r) == player:
            count += 1
            c, r = c + dc, r + dr
        c, r = col - dc, row - dr
        while cell(board, c, r) == player:
            count += 1
            c, r = c - dc, r - dr
        if count >= 4:
            return True
    return False


def canonical_state(board: tuple[tuple[str, ...], ...], to_move: str) -> tuple:
    """Ground-truth state key: the board tuple already is canonical (see
    module docstring), so this just packages it with whose turn is next --
    the same stones with a different side to move are a different
    game-theoretic state and must not collide as a key.
    """
    return (board, to_move)


def board_key(board: tuple[tuple[str, ...], ...], to_move: str) -> str:
    """Compact, stable string form of canonical_state, for use in instance
    ids / exclude_ids sets (which need a hashable str, not a nested tuple)."""
    return "|".join("".join(column) for column in board) + f"#{to_move}"


def replay(pre_moves: tuple[int, ...], width: int, height: int) -> tuple[tuple[str, ...], ...]:
    """Apply pre_moves from an empty board, alternating starting with X.
    Raises ValueError on an illegal move or a pre_moves sequence that
    already contains a win -- pre_moves are meant to set up an *unresolved*
    puzzle position, never to replay a finished game.
    """
    board = make_empty_board(width)
    player = X
    for col in pre_moves:
        if not is_legal_move(board, col, height):
            raise ValueError(f"illegal pre_move: column {col}")
        board = apply_move(board, col, player)
        row = len(board[col]) - 1
        if check_win(board, col, row, player):
            raise ValueError("pre_moves already contains a completed win")
        player = opponent(player)
    return board


def solve_forced_win(
    board: tuple[tuple[str, ...], ...], side_to_move: str, hero: str, plies_left: int,
    height: int, cache: dict | None = None, stats: dict | None = None,
) -> bool:
    """Can `hero` force a win within exactly `plies_left` plies from here,
    with `side_to_move` playing next? See module docstring for the
    alternating-turn recursion.
    """
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

    moves = legal_moves(board, height)
    if not moves:
        cache[key] = False
        return False

    if side_to_move == hero:
        result = False
        for col in moves:
            new_board = apply_move(board, col, hero)
            row = len(new_board[col]) - 1
            if check_win(new_board, col, row, hero):
                result = True
                break
            if solve_forced_win(new_board, opponent(hero), hero, plies_left - 1, height, cache, stats):
                result = True
                break
    else:
        result = True
        for col in moves:
            new_board = apply_move(board, col, side_to_move)
            row = len(new_board[col]) - 1
            if check_win(new_board, col, row, side_to_move):
                result = False  # opponent escaping into their own win defeats hero's forced win
                break
            if not solve_forced_win(new_board, hero, hero, plies_left - 1, height, cache, stats):
                result = False
                break

    cache[key] = result
    return result


def shortest_forced_win(
    board: tuple[tuple[str, ...], ...], hero: str, max_plies: int, height: int, cache: dict | None = None,
) -> int | None:
    """Minimal plies_left for which solve_forced_win is true, or None if no
    forced win exists within max_plies. hero always moves first in this
    search, so the answer (when it exists) is always odd -- the win must
    land on one of hero's own moves.
    """
    if cache is None:
        cache = {}
    for p in range(1, max_plies + 1):
        if solve_forced_win(board, hero, hero, p, height, cache):
            return p
    return None


def preserving_first_moves(
    board: tuple[tuple[str, ...], ...], hero: str, plies_left: int, height: int, cache: dict | None = None,
) -> list[int]:
    """Columns hero can play right now that either win immediately or keep
    a forced win alive with one fewer ply remaining. Used both for
    ConnectFourInstance.path_count (root) and per-move verification
    (any position along a claimed line).
    """
    if cache is None:
        cache = {}
    result = []
    for col in legal_moves(board, height):
        new_board = apply_move(board, col, hero)
        row = len(new_board[col]) - 1
        if check_win(new_board, col, row, hero):
            result.append(col)
            continue
        if plies_left > 1 and solve_forced_win(new_board, opponent(hero), hero, plies_left - 1, height, cache):
            result.append(col)
    return result


def render_ascii(board: tuple[tuple[str, ...], ...], height: int) -> str:
    """Top row first, 0-indexed column header -- read-only rendering of a
    fixed position for the prompt; never re-parsed from model output.
    """
    lines = []
    for row in range(height - 1, -1, -1):
        lines.append(" ".join(cell(board, c, row) or "." for c in range(len(board))))
    lines.append(" ".join(str(c) for c in range(len(board))))
    return "\n".join(lines)
