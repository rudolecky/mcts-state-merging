"""Classical Monte Carlo Tree Search on Connect Four -- no LLM, no learned
metric. Merging is exact board-state equality (a real transposition table),
and the value estimate is a plain random-rollout (the original 2006 MCTS
recipe). This isolates the one question every LLM-based experiment in this
project entangled with "does the learned metric work": does node-merging
in a search tree help at all, in its purest, cheapest-to-verify form.

Node keying IS the merge switch, structurally (mirrors search.py's
merge_enabled flag): treatment keys nodes by
connect_four_engine.canonical_state(board, to_move) -- identical boards
reached via different move orders always resolve to the same dict entry.
Baseline keys nodes by the literal move-path from the root, which can never
collide across branches by construction -- a plain tree. One implementation,
one key function swapped.

Selection uses classical UCB1, not PUCT -- deliberate: PUCT's prior term
came from LLM sample proportions; there's no policy network here to supply
one, so plain UCB1 (unvisited edges get infinite priority) is the
appropriate classical algorithm, not a downgrade.

Cycles are structurally impossible for the same reason as search.py's LLM
harness: every edge goes from one ply to the next (a column drop always
advances the game by exactly one ply), so no path can revisit an ancestor.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets.connect_four_engine import (
    apply_move,
    canonical_state,
    check_win,
    is_full,
    legal_moves,
    opponent,
    replay,
)


@dataclass
class MCTSEdge:
    child_key: tuple
    n_edge: int = 0


@dataclass
class MCTSNode:
    board: tuple
    to_move: str
    n_visits: int = 0
    w_value: float = 0.0
    children: dict[int, MCTSEdge] = field(default_factory=dict)
    untried_moves: list[int] = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float | None = None  # 1.0 hero wins, 0.0 draw/opponent wins


@dataclass
class MCTSGraph:
    nodes: dict[tuple, MCTSNode]
    root_key: tuple


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    c: float = 1.4  # ~sqrt(2), the standard UCB1 exploration constant
    guidance_depth_cap: int | None = None  # None = honest full rollout (today's behavior)


def _make_node(board: tuple, to_move: str, hero: str, height: int, just_won: bool) -> MCTSNode:
    """`to_move` is whoever moves NEXT (after the move that produced this
    board); the side whose move just landed is therefore opponent(to_move).
    """
    is_terminal = just_won or is_full(board, height)
    terminal_value = None
    if is_terminal:
        mover = opponent(to_move)
        terminal_value = 1.0 if (just_won and mover == hero) else 0.0
    return MCTSNode(
        board=board, to_move=to_move,
        untried_moves=[] if is_terminal else legal_moves(board, height),
        is_terminal=is_terminal, terminal_value=terminal_value,
    )


def create_root(pre_moves: tuple[int, ...], to_move: str, width: int, height: int, merge_enabled: bool) -> MCTSGraph:
    board = replay(pre_moves, width, height)
    root_key = canonical_state(board, to_move) if merge_enabled else ()
    root = MCTSNode(board=board, to_move=to_move, untried_moves=legal_moves(board, height))
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(parent.n_visits) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list[tuple], tuple[int, ...]]:
    """Descend via UCB1 through fully-expanded, non-terminal nodes. Stops at
    the first node with an untried move remaining, or at a terminal node.
    Returns (path of node keys, move-path from root -- only meaningful for
    baseline's key computation).
    """
    path = [graph.root_key]
    path_moves: tuple[int, ...] = ()
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        best_col = max(
            node.children,
            key=lambda c: _ucb1_score(node, node.children[c], graph.nodes[node.children[c].child_key], config.c),
        )
        edge = node.children[best_col]
        path.append(edge.child_key)
        path_moves = path_moves + (best_col,)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def expand(
    graph: MCTSGraph, leaf_key: tuple, path_moves: tuple[int, ...],
    hero: str, height: int, config: ClassicalMCTSConfig, rng: random.Random,
) -> tuple:
    """Add one untried child of the leaf (or reuse an existing node with the
    same canonical state, in merge mode -- a real transposition hit)."""
    node = graph.nodes[leaf_key]
    col = rng.choice(node.untried_moves)
    node.untried_moves.remove(col)
    new_board = apply_move(node.board, col, node.to_move)
    row = len(new_board[col]) - 1
    won = check_win(new_board, col, row, node.to_move)
    new_to_move = opponent(node.to_move)

    child_key = canonical_state(new_board, new_to_move) if config.merge_enabled else path_moves + (col,)
    if child_key not in graph.nodes:
        graph.nodes[child_key] = _make_node(new_board, new_to_move, hero, height, won)
    node.children[col] = MCTSEdge(child_key=child_key)
    return child_key


def simulate(
    board: tuple, to_move: str, hero: str, width: int, height: int, rng: random.Random,
    guidance_depth_cap: int | None = None,
) -> float:
    """Uniformly random legal moves for both sides until terminal or the
    board fills. 1.0 iff hero completes a win during the rollout, else 0.0
    (draw or opponent win) -- the same fixed-perspective binary framing
    search.py's own backup uses, not a symmetric two-player value.

    `guidance_depth_cap`, when set, caps the rollout at that many plies
    instead of playing to the board-fill terminal: a decisive win found
    within the cap still returns the honest 1.0/0.0, but exhausting the
    cap without one returns a neutral 0.5 instead of continuing -- a
    cheap, less-discriminating value estimate, the classical analog of a
    noisy shared value proxy (H3_MERGE_SEARCH_FINDINGS.md's C6 ablation).
    None (default) reproduces today's honest-rollout behavior exactly.
    """
    natural_max_depth = width * height + 1  # defensive cap; the board fills before this in practice
    max_depth = natural_max_depth if guidance_depth_cap is None else min(guidance_depth_cap, natural_max_depth)
    for _ in range(max_depth):
        moves = legal_moves(board, height)
        if not moves:
            return 0.0
        col = rng.choice(moves)
        board = apply_move(board, col, to_move)
        row = len(board[col]) - 1
        if check_win(board, col, row, to_move):
            return 1.0 if to_move == hero else 0.0
        to_move = opponent(to_move)
    return 0.5 if guidance_depth_cap is not None else 0.0


def backup(graph: MCTSGraph, path: list[tuple], value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            col = next(c for c, e in parent.children.items() if e.child_key == key)
            parent.children[col].n_edge += 1


def run_search(
    pre_moves: tuple[int, ...], to_move: str, width: int, height: int,
    config: ClassicalMCTSConfig, budget: int, rng: random.Random,
) -> MCTSGraph:
    """budget counts EXPANSIONS (new-node attempts), not simulations that
    land on an already-terminal node -- those are free re-visits, matching
    search.py's own budget accounting exactly.
    """
    graph = create_root(pre_moves, to_move, width, height, config.merge_enabled)
    hero = to_move
    expansions_used = 0
    iterations = 0
    max_iterations = 20 * budget
    while expansions_used < budget and iterations < max_iterations:
        iterations += 1
        path, path_moves = select(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.is_terminal:
            backup(graph, path, leaf.terminal_value)
            continue
        child_key = expand(graph, path[-1], path_moves, hero, height, config, rng)
        expansions_used += 1
        path.append(child_key)
        child = graph.nodes[child_key]
        value = child.terminal_value if child.is_terminal else simulate(
            child.board, child.to_move, hero, width, height, rng, config.guidance_depth_cap
        )
        backup(graph, path, value)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal and n.terminal_value == 1.0 for n in graph.nodes.values())


def run_random_search(board: tuple, to_move: str, hero: str, width: int, height: int, budget: int, rng: random.Random) -> bool:
    """The "no MCTS at all" ablation (see RANDOM_VS_MCTS_FINDINGS.md):
    `budget` independent random rollouts from the start position, no tree,
    no UCB1, nothing carried between attempts -- reuses this module's own
    `simulate()`. Takes a board directly (not pre_moves) since not every
    useful test position is reachable via alternating replay."""
    for _ in range(budget):
        if simulate(board, to_move, hero, width, height, rng) == 1.0:
            return True
    return False
