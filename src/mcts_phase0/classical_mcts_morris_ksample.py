"""K-sampling classical MCTS on Three Men's Morris -- no LLM, no learned
metric. Combines three properties established separately elsewhere in this
project but never together: Morris's two-player hero/opponent node shape
(classical_mcts_morris.py), cycle-safe select()/run_search (movement-phase
slides are reversible, same as classical_mcts_morris.py), and
K-independent-draws-with-replacement expansion with a per-draw-unique
baseline key (classical_mcts_ksample.py's core pattern).

Tests whether Connect Four's finding -- guidance-independent null flips to
guidance-dependent once the expansion mechanism loses its structural
anti-duplicate guarantee -- generalizes past Connect Four, or whether
Morris's OWN null (adversarial interleaving makes two independently-chosen
move sequences rarely land on the same joint state, confirmed in
MORRIS_FINDINGS.md) survives even without any structural guarantee at all.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets.morris_engine import apply_move, check_win, legal_moves, opponent


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
    expanded: bool = False
    is_terminal: bool = False
    terminal_value: float | None = None


@dataclass
class MCTSGraph:
    nodes: dict[tuple, MCTSNode]
    root_key: tuple


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    K: int = 4
    c: float = 1.4
    rollout_depth: int = 20  # underlying honest random-walk length
    guidance_depth_cap: int | None = None  # None = use rollout_depth honestly (0.0 on exhaustion)


def _make_node(board: tuple, to_move: str, hero: str, just_won_by: str | None) -> MCTSNode:
    moves = legal_moves(board, to_move)
    is_terminal = just_won_by is not None or not moves
    terminal_value = None
    if is_terminal:
        if just_won_by is not None:
            terminal_value = 1.0 if just_won_by == hero else 0.0
        else:
            terminal_value = 0.0 if to_move == hero else 1.0
    return MCTSNode(board=board, to_move=to_move, is_terminal=is_terminal, terminal_value=terminal_value)


def create_root(board: tuple, to_move: str, merge_enabled: bool) -> MCTSGraph:
    root_key = (board, to_move) if merge_enabled else ()
    root = MCTSNode(board=board, to_move=to_move)
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(parent.n_visits) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list[tuple], list[int]]:
    """Cycle-safe: restricts UCB1's argmax to children not already on this
    path (Morris's movement phase is reversible, so an unrestricted argmax
    could deterministically prefer a cycling child forever). Returns
    (path, edge_draws) -- edge_draws[i] is the draw index used to step from
    path[i] to path[i+1], threaded explicitly because two different draws
    from the same parent can point at the same merged child."""
    path = [graph.root_key]
    edge_draws = []
    node = graph.nodes[graph.root_key]
    while node.expanded and node.children and not node.is_terminal:
        candidates = {d: e for d, e in node.children.items() if e.child_key not in path}
        if not candidates:
            break
        best_draw = max(
            candidates,
            key=lambda d: _ucb1_score(node, candidates[d], graph.nodes[candidates[d].child_key], config.c),
        )
        edge = candidates[best_draw]
        path.append(edge.child_key)
        edge_draws.append(best_draw)
        node = graph.nodes[edge.child_key]
    return path, edge_draws


_baseline_serial_counter = [0]


def _next_baseline_id() -> tuple:
    _baseline_serial_counter[0] += 1
    return ("baseline", _baseline_serial_counter[0])


def expand_batch(
    graph: MCTSGraph, leaf_key: tuple, hero: str, config: ClassicalMCTSConfig, rng: random.Random,
) -> list[tuple]:
    node = graph.nodes[leaf_key]
    moves = legal_moves(node.board, node.to_move)
    child_keys = []
    for draw_index in range(config.K):
        from_cell, to_cell = rng.choice(moves)
        new_board = apply_move(node.board, from_cell, to_cell, node.to_move)
        won = check_win(new_board, node.to_move)
        new_to_move = opponent(node.to_move)

        if config.merge_enabled:
            child_key = (new_board, new_to_move)
        else:
            child_key = _next_baseline_id()  # fresh id per draw -- never merges, even exact repeats

        if child_key not in graph.nodes:
            graph.nodes[child_key] = _make_node(new_board, new_to_move, hero, node.to_move if won else None)
        node.children[draw_index] = MCTSEdge(child_key=child_key)
        child_keys.append(child_key)
    node.expanded = True
    return child_keys


def simulate(board: tuple, to_move: str, hero: str, config: ClassicalMCTSConfig, rng: random.Random) -> float:
    """Uniformly random legal moves for both sides. `guidance_depth_cap`,
    when set, cuts the rollout short and returns a neutral 0.5 instead of
    continuing to `rollout_depth` -- the same honest/degraded distinction
    classical_mcts.py's Connect Four ablation uses. None (default) plays
    the full `rollout_depth` and returns 0.0 on exhaustion, matching
    classical_mcts_morris.py's original unconditional behavior."""
    cap = config.rollout_depth if config.guidance_depth_cap is None else min(config.guidance_depth_cap, config.rollout_depth)
    for _ in range(cap):
        moves = legal_moves(board, to_move)
        if not moves:
            return 0.0 if to_move == hero else 1.0
        from_cell, to_cell = rng.choice(moves)
        board = apply_move(board, from_cell, to_cell, to_move)
        if check_win(board, to_move):
            return 1.0 if to_move == hero else 0.0
        to_move = opponent(to_move)
    return 0.5 if config.guidance_depth_cap is not None else 0.0


def backup(graph: MCTSGraph, path: list[tuple], edge_draws: list[int], value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            parent.children[edge_draws[i - 1]].n_edge += 1


def run_search(board: tuple, to_move: str, config: ClassicalMCTSConfig, budget: int, rng: random.Random) -> MCTSGraph:
    """budget counts EXPANSION EVENTS (K-draw batches)."""
    graph = create_root(board, to_move, config.merge_enabled)
    hero = to_move
    expansions_used = 0
    iterations = 0
    max_iterations = 20 * budget
    while expansions_used < budget and iterations < max_iterations:
        iterations += 1
        path, edge_draws = select(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.is_terminal:
            backup(graph, path, edge_draws, leaf.terminal_value)
            continue
        if leaf.expanded:
            # select() gave up on a cycle: leaf is already expanded (has K
            # children) but every one of them would revisit the path.
            # Refine its own estimate with one more rollout instead of
            # re-expanding -- free, like the terminal case, and avoids
            # silently overwriting the existing K children's edges.
            value = simulate(leaf.board, leaf.to_move, hero, config, rng)
            backup(graph, path, edge_draws, value)
            continue
        child_keys = expand_batch(graph, path[-1], hero, config, rng)
        expansions_used += 1
        for draw_index, child_key in enumerate(child_keys):
            child = graph.nodes[child_key]
            value = child.terminal_value if child.is_terminal else simulate(child.board, child.to_move, hero, config, rng)
            backup(graph, path + [child_key], edge_draws + [draw_index], value)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal and n.terminal_value == 1.0 for n in graph.nodes.values())
