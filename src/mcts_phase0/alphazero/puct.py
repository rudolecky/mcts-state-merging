"""Real PUCT search for Connect Four: a trained network's own policy as
the prior (no uniform-1/K stand-in), leaf evaluation via the network
directly (no rollout at all -- the actual AlphaZero mechanism, distinct
from every rollout-based module elsewhere in this project).

Value convention (load-bearing, read before touching `backup`): every
node's `w_value`/`n_visits` is from *that node's own* `to_move`'s
perspective, in [0, 1] (matching `network.py`'s value head). Walking a
path back up to the root, each ancestor is the *other* player, so the
value the parent should credit is the *complement* `1 - value`, not the
child's own value and not a negation (this project's usual value range is
[0, 1], not AlphaZero's original [-1, +1], so the perspective-flip
operation is complement, not negation). `_puct_score` and `backup` both
depend on this being applied at exactly one point each -- get it wrong and
the network trains to prefer losing, a well-known AlphaZero-clone bug
class this project's own test suite exists to catch before any real
training runs.

Unlike every other classical module here, a node's children are ALL
created the moment the node is first expanded (one network call gives a
policy over every legal move at once), not one per expansion the way
`untried_moves`-tracked UCB1 or K-sampling does -- PUCT's own prior
already tells selection which untried moves are worth trying, so there's
no need to force trying each one before ever repeating.

Connect Four's moves are irreversible (a column drop can never be
replayed at a node), so -- as in `classical_mcts.py` -- no cycle-safety
is needed in `select()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..datasets.connect_four_engine import (
    apply_move,
    canonical_state,
    check_win,
    is_full,
    legal_moves,
    opponent,
    replay,
)

HEIGHT = 4


@dataclass
class MCTSEdge:
    child_key: tuple
    prior: float
    n_edge: int = 0


@dataclass
class MCTSNode:
    board: tuple
    to_move: str
    n_visits: int = 0
    w_value: float = 0.0  # from this node's OWN to_move's perspective
    children: dict[int, MCTSEdge] = field(default_factory=dict)
    expanded: bool = False
    is_terminal: bool = False
    terminal_value: float | None = None  # also from this node's own to_move's perspective


@dataclass
class MCTSGraph:
    nodes: dict[tuple, MCTSNode]
    root_key: tuple


@dataclass
class PUCTConfig:
    merge_enabled: bool
    c_puct: float = 1.5
    dirichlet_alpha: float | None = None  # None = no root noise (self-play only, never in evaluation)
    dirichlet_epsilon: float = 0.25


def _make_node(board: tuple, to_move: str, just_won: bool) -> MCTSNode:
    full = is_full(board, HEIGHT)
    is_terminal = just_won or full
    if just_won:
        terminal_value = 0.0  # the opponent just completed 4-in-a-row: to_move just lost
    elif full:
        terminal_value = 0.5  # draw
    else:
        terminal_value = None
    return MCTSNode(board=board, to_move=to_move, is_terminal=is_terminal, terminal_value=terminal_value)


def create_root(pre_moves: tuple[int, ...], to_move: str, width: int, merge_enabled: bool) -> MCTSGraph:
    board = replay(pre_moves, width, HEIGHT)
    root_key = canonical_state(board, to_move) if merge_enabled else ()
    root = MCTSNode(board=board, to_move=to_move)
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _puct_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c_puct: float) -> float:
    q = 0.0 if child.n_visits == 0 else 1.0 - (child.w_value / child.n_visits)  # complement: parent's perspective
    exploration = c_puct * edge.prior * math.sqrt(parent.n_visits) / (1 + edge.n_edge)
    return q + exploration


def select(graph: MCTSGraph, config: PUCTConfig) -> tuple[list[tuple], list[int], tuple[int, ...]]:
    """Descend via PUCT through expanded, non-terminal nodes -- every
    legal move already has an edge (with a prior) the moment its parent is
    expanded, so there's no separate untried/expanded distinction inside a
    node the way UCB1-based modules need."""
    path = [graph.root_key]
    edge_draws: list[int] = []
    path_moves: tuple[int, ...] = ()
    node = graph.nodes[graph.root_key]
    while node.expanded and not node.is_terminal:
        best_col = max(
            node.children,
            key=lambda c: _puct_score(node, node.children[c], graph.nodes[node.children[c].child_key], config.c_puct),
        )
        edge = node.children[best_col]
        path.append(edge.child_key)
        edge_draws.append(best_col)
        path_moves = path_moves + (best_col,)
        node = graph.nodes[edge.child_key]
    return path, edge_draws, path_moves


def expand_and_evaluate(
    graph: MCTSGraph, node_key: tuple, path_moves: tuple[int, ...], width: int,
    config: PUCTConfig, evaluate_fn, rng: np.random.Generator | None = None,
) -> float:
    """One network call for `node_key`; creates edges (with priors) AND
    child nodes for every legal move in one pass (cheap game logic per
    child -- apply_move/check_win/is_full, no extra network calls).
    Returns this node's own value, from its own to_move's perspective, for
    the caller to back up."""
    node = graph.nodes[node_key]
    policy, value = evaluate_fn(node.board, node.to_move)
    moves = legal_moves(node.board, HEIGHT)

    priors = {m: float(policy[m]) for m in moves}
    if config.dirichlet_alpha is not None and node_key == graph.root_key:
        noise = rng.dirichlet([config.dirichlet_alpha] * len(moves))
        eps = config.dirichlet_epsilon
        priors = {m: (1 - eps) * priors[m] + eps * float(n) for m, n in zip(moves, noise)}

    for col in moves:
        new_board = apply_move(node.board, col, node.to_move)
        row = len(new_board[col]) - 1
        won = check_win(new_board, col, row, node.to_move)
        new_to_move = opponent(node.to_move)
        child_key = canonical_state(new_board, new_to_move) if config.merge_enabled else path_moves + (col,)
        if child_key not in graph.nodes:
            graph.nodes[child_key] = _make_node(new_board, new_to_move, won)
        node.children[col] = MCTSEdge(child_key=child_key, prior=priors[col])
    node.expanded = True
    return value


def backup(graph: MCTSGraph, path: list[tuple], edge_draws: list[int], leaf_value: float) -> None:
    """Walks the path root-ward, flipping perspective (complement, not
    negation -- see module docstring) exactly once per ply."""
    value = leaf_value
    for i in reversed(range(len(path))):
        node = graph.nodes[path[i]]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            parent.children[edge_draws[i - 1]].n_edge += 1
        value = 1.0 - value


def run_search(
    pre_moves: tuple[int, ...], to_move: str, width: int,
    config: PUCTConfig, budget: int, evaluate_fn, rng: np.random.Generator | None = None,
) -> MCTSGraph:
    """budget counts simulations (select -> expand-and-evaluate -> backup),
    the standard AlphaZero unit -- not "expansion events" the way the
    K-sampling modules count budget, since PUCT never batches K candidates
    per visit."""
    graph = create_root(pre_moves, to_move, width, config.merge_enabled)
    for _ in range(budget):
        path, edge_draws, path_moves = select(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.is_terminal:
            backup(graph, path, edge_draws, leaf.terminal_value)
            continue
        value = expand_and_evaluate(graph, path[-1], path_moves, width, config, evaluate_fn, rng)
        backup(graph, path, edge_draws, value)
    return graph


def visit_policy(graph: MCTSGraph, width: int) -> dict[int, float]:
    """Normalized visit-count distribution over the root's children -- the
    self-play training target for the policy head (not the prior; the
    prior was the network's own guess *before* search, this is what search
    actually concluded)."""
    root = graph.nodes[graph.root_key]
    total = sum(graph.nodes[e.child_key].n_visits for e in root.children.values())
    if total == 0:
        return {col: 1.0 / len(root.children) for col in root.children}
    return {col: graph.nodes[e.child_key].n_visits / total for col, e in root.children.items()}


def is_solved(graph: MCTSGraph, hero: str) -> bool:
    """True iff the tree contains a terminal node, anywhere, representing
    hero's opponent having just lost -- i.e. hero delivered a win at the
    end of some explored line, not necessarily in one move. A terminal
    node's own to_move always just lost (`_make_node`'s convention:
    terminal_value=0.0 means "this node's to_move just lost"), and
    to_move alternates strictly with depth, so "to_move != hero" here is
    exactly "this loss belongs to hero's opponent, at whatever depth" --
    no separate depth-tracking needed. Mirrors every other module's
    `is_solved(graph)` in this project, adapted for this module's
    terminal_value convention (0.0/0.5, not 0.0/1.0)."""
    return any(n.is_terminal and n.terminal_value == 0.0 and n.to_move != hero for n in graph.nodes.values())
