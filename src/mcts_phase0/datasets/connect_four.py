"""Connect Four tactical puzzles: given a partially-played board and whose
turn it is, find the forced win within instance.k_plies plies -- verified
by the exact solver in connect_four_engine.py, not by the LLM's own claims.

Framed as a puzzle-plus-independent-verifier (like countdown's brute-force
derivation check and prosqa's DAG path solver), not an open-ended self-play
game: the point of this dataset is not "does merging help in a board game"
(transposition tables already solve exact state-merging in game engines --
not a new question) but whether the learned hidden-state projection's merge
decisions agree with a REAL, ground-truth notion of state identity (an
actual transposition: two move orders reaching the byte-identical board),
which countdown/prosqa cannot provide.

`path_count` here is scoped narrower than countdown's full-derivation
enumeration: it counts distinct FIRST moves for the side to move that
preserve the forced-win invariant, not all complete lines (full-line
enumeration is exponential in the opponent's replies and isn't the object
of interest -- first-move multiplicity is what's relevant to how much
search-redundancy exists at the depth merging first has a chance to act on).

`exclude_ids` is implemented (unlike prosqa, like countdown): at these
board sizes, positions with an exact-K forced win are comparatively rare
among random walks, so independent generation runs tend to rediscover the
same small set of tactical shapes -- and because this domain is *built
around* transpositions, the same board is also reachable via multiple
`pre_moves` orders. Both effects push toward cross-seed recurrence more
than countdown's already-bounded integer range, and are the opposite of
prosqa's fresh-per-call entity-name space that can't plausibly collide.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .connect_four_engine import (
    apply_move,
    board_key,
    check_win,
    is_legal_move,
    make_empty_board,
    opponent,
    preserving_first_moves,
    render_ascii,
    replay,
    shortest_forced_win,
    X,
)

_MOVE_RE = re.compile(r"^\s*drop in column\s+(\d+)\s*\.?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ConnectFourInstance:
    pre_moves: tuple[int, ...]
    width: int
    height: int
    k_plies: int
    path_count: int

    @property
    def to_move(self) -> str:
        return X if len(self.pre_moves) % 2 == 0 else opponent(X)

    @property
    def id(self) -> str:
        board = replay(self.pre_moves, self.width, self.height)
        return f"c4_{self.width}x{self.height}_k{self.k_plies}_{board_key(board, self.to_move)}"


def generate_puzzles(
    n_low: int,
    n_high: int,
    seed: int = 0,
    width: int = 6,
    height: int = 5,
    k_plies: int = 3,
    low_threshold: int = 2,
    max_pre_moves: int = 8,
    exclude_ids: set[str] | None = None,
) -> list[ConnectFourInstance]:
    """Generate instances stratified into low-path-count (< low_threshold)
    and high-path-count (>= low_threshold) buckets, n_low and n_high of
    each, via random-walk-then-solver-filter (mirrors countdown's
    generate-then-filter precedent): draw a short random legal opening,
    keep it only if it has an exact-k_plies forced win for the side to
    move, discard otherwise.

    `max_pre_moves` default of 8 (not e.g. 3-4) is an empirical Stage-0
    finding, not a guess: at width=5/height=4, k_plies=3 exact forced-win
    positions essentially never appear among openings shorter than ~5
    plies (0/3000 random draws), and stay rare even past that (~0.4% of
    draws at max_pre_moves=8-9) -- confirms the module docstring's
    reasoning for needing `exclude_ids` (genuinely tactical positions are
    rare enough that independent runs will rediscover the same few).
    """
    exclude_ids = exclude_ids or set()
    rng = random.Random(seed)
    seen: set[str] = set()
    low_bucket: list[ConnectFourInstance] = []
    high_bucket: list[ConnectFourInstance] = []
    attempts = 0
    max_attempts = 500 * (n_low + n_high) + 2000
    while (len(low_bucket) < n_low or len(high_bucket) < n_high) and attempts < max_attempts:
        attempts += 1
        pre_moves = _random_opening(rng, width, height, max_pre_moves)
        if pre_moves is None:
            continue
        board = replay(pre_moves, width, height)
        hero = X if len(pre_moves) % 2 == 0 else opponent(X)

        cache: dict = {}
        shortest = shortest_forced_win(board, hero, max_plies=k_plies, height=height, cache=cache)
        if shortest != k_plies:
            continue

        path_count = len(preserving_first_moves(board, hero, k_plies, height, cache))
        key = board_key(board, hero)
        if key in seen:
            continue

        inst = ConnectFourInstance(
            pre_moves=tuple(pre_moves), width=width, height=height, k_plies=k_plies, path_count=path_count,
        )
        if inst.id in exclude_ids:
            continue
        seen.add(key)

        if path_count < low_threshold and len(low_bucket) < n_low:
            low_bucket.append(inst)
        elif path_count >= low_threshold and len(high_bucket) < n_high:
            high_bucket.append(inst)

    return low_bucket + high_bucket


def _random_opening(rng: random.Random, width: int, height: int, max_pre_moves: int) -> tuple[int, ...] | None:
    """A short random legal opening with no completed win along the way --
    puzzles start from an unresolved position, never a finished game."""
    pre_len = rng.randint(0, max_pre_moves)
    board = make_empty_board(width)
    player = X
    moves: list[int] = []
    for _ in range(pre_len):
        legal = [c for c in range(width) if is_legal_move(board, c, height)]
        if not legal:
            return None
        col = rng.choice(legal)
        board = apply_move(board, col, player)
        row = len(board[col]) - 1
        if check_win(board, col, row, player):
            return None  # opening must not already contain a win
        moves.append(col)
        player = opponent(player)
    return tuple(moves)


def build_prompt(instance: ConnectFourInstance, encourage_scratch_board: bool = False) -> str:
    example = (
        "Example (5 columns, X to move, forced win):\n"
        ". . . . .\n"
        ". X X . .\n"
        "0 1 2 3 4\n"
        "X has two in a row. Playing column 3 makes three in a row with "
        "both ends (columns 0 and 4) open -- O can only block one end, so "
        "X completes four in a row at the other end no matter what O plays.\n"
        "Step 1: drop in column 3\n"
        "Step 2: drop in column 0\n"
        "Step 3: drop in column 4\n"
        "Answer: win\n\n"
    )
    board = replay(instance.pre_moves, instance.width, instance.height)
    grid = render_ascii(board, instance.height)
    body = (
        f"{instance.width} columns (0-{instance.width - 1}), {instance.to_move} to move. "
        "You have a forced win no matter how the other side replies -- find it.\n"
        f"{grid}\n"
        "Write one move per line as\n"
        "Step N: drop in column C\n"
        "(moves alternate sides automatically starting with the side to move above). "
        "When your move completes four in a row, stop and write\n"
        "Answer: win\n"
    )
    if encourage_scratch_board:
        body += (
            "You may sketch the board after each move as a plain comment line before "
            "the Step line, if that helps you track the position -- only the "
            "'Step N:' and 'Answer:' lines are graded.\n"
        )
    return example + body


def parse_and_verify(
    instance: ConnectFourInstance, step_bodies: list[str], answer_body: str | None
) -> tuple[bool, dict]:
    """Replay claimed moves on a fresh internal board (there is no
    model-produced board text to distrust, by construction). At every one
    of the hero's own turns, the chosen column must be one of the columns
    the solver certifies as forcing-win-preserving right now -- checking
    only that the line eventually reaches a win would validate one lucky
    branch against a possibly-suboptimal self-scripted opponent, not that
    the win was actually forced. Opponent moves are checked for legality
    only, never optimality.
    """
    board = replay(instance.pre_moves, instance.width, instance.height)
    hero = instance.to_move
    side = hero
    remaining = instance.k_plies
    cache: dict = {}
    won = False

    for i, body in enumerate(step_bodies):
        m = _MOVE_RE.match(body)
        if not m:
            return False, {"well_formed": False, "reason": f"unparseable move: {body!r}"}
        col = int(m.group(1))

        if remaining <= 0:
            return False, {"well_formed": False, "reason": "exceeds ply budget"}
        if not is_legal_move(board, col, instance.height):
            return False, {"well_formed": False, "reason": f"illegal move: column {col}"}
        if side == hero:
            preserving = preserving_first_moves(board, hero, remaining, instance.height, cache)
            if col not in preserving:
                return False, {"well_formed": False, "reason": f"column {col} is not a forcing move"}

        board = apply_move(board, col, side)
        row = len(board[col]) - 1
        just_won = check_win(board, col, row, side)
        if just_won and side != hero:
            return False, {"well_formed": False, "reason": "opponent won -- position was not actually forced"}

        remaining -= 1
        if just_won and side == hero:
            won = True
            if i != len(step_bodies) - 1:
                return False, {"well_formed": False, "reason": "extra moves claimed after the winning move"}
            break
        side = opponent(side)

    if not won:
        return False, {"well_formed": False, "reason": "did not reach a win"}

    answer_norm = (answer_body or "").strip().rstrip(".").lower()
    if answer_norm != "win":
        return False, {"well_formed": True, "reason": f"wrong final answer: {answer_body!r}"}
    return True, {"well_formed": True, "reason": "ok"}


def canonical_state_at(instance: ConnectFourInstance, step_bodies_so_far: list[str]) -> str | None:
    """Ground-truth "same state" key for a snapshot taken after some prefix
    of moves within one trace -- passed to collect.py as a
    `ground_truth_key_fn`. Replays pre_moves, then each parsed move in
    order (reusing the same move grammar as parse_and_verify); returns None
    on any illegal/unparseable move or a claimed move after the game is
    already decided (a partial/malformed trace has no well-defined board),
    which is exactly what makes such snapshots skip cleanly in
    geometry.ground_truth_merge_confusion rather than poison the analysis
    with a bogus key.
    """
    board = replay(instance.pre_moves, instance.width, instance.height)
    side = instance.to_move
    for body in step_bodies_so_far:
        m = _MOVE_RE.match(body)
        if not m:
            return None
        col = int(m.group(1))
        if not is_legal_move(board, col, instance.height):
            return None
        board = apply_move(board, col, side)
        side = opponent(side)
    return board_key(board, side)
