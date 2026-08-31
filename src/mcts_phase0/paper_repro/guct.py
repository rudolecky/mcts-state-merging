"""Plain GreedyUCT (GUCT), not GUCT-Uniform -- arXiv:2405.18248's Definitions
2 & 4: NEC f_GUCT(n) = h_GUCT(n) - c*sqrt(2*log|L(p)| / |L(n)|), where
|L(*)| are leaf counts (numerically identical to visit counts here, since
this is a lazy, no-rollout MCTS: every backup routes through exactly one
leaf sample), and h_GUCT is Monte Carlo backup (Definition 4) -- the plain
running *mean* of every leaf heuristic value seen in a node's subtree, not
GUCT-Uniform's Full Bellman minimum. Framed as minimization (planning is a
cost), so this is structurally the same shape as classical UCB1 (see
`classical_mcts_blocksworld.py`) with a sign flip -- and because it's a mean
rather than a min/max spread, it has the same self-limiting exploration
term UCB1 has (shrinks as a node accumulates more samples), unlike
GUCT-Uniform's LCB1-Uniform. So the simple "reuse a free extra sample on
cycle-block" pattern is expected to be safe here (confirmed by this
module's own test), unlike GUCT-Uniform's confirmed divergence risk.

`c` is a free hyperparameter in the paper with no stated default in the
visible text; this project's other UCB1 modules all use c=1.4 (~sqrt(2),
the textbook constant), reused here as a documented default rather than a
hidden assumption.

Generic: works on any pyperplan `Task`, not Blocksworld-specific -- used
here against real IPC domains via `pddl_loader.load_task`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from pyperplan.search.searchspace import make_root_node


def _legal_moves(state, task):
    return [op for op in task.operators if op.applicable(state)]


def _apply_move(state, op):
    return op.apply(state)


def _is_goal(state, task):
    return task.goal_reached(state)


@dataclass
class MCTSEdge:
    child_key: object
    n_edge: int = 0


@dataclass
class MCTSNode:
    state: frozenset
    own_h: float = 0.0  # this state's own heuristic value, fixed at creation
    n_visits: int = 0
    sum_h: float = 0.0  # running mean h_GUCT = sum_h / n_visits
    children: dict = field(default_factory=dict)  # Operator -> MCTSEdge
    untried_moves: list = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class MCTSGraph:
    nodes: dict
    root_key: object
    next_id: int = 0  # baseline-mode node-id counter; see expand()


@dataclass
class GUCTConfig:
    merge_enabled: bool
    c: float = 1.4  # ~sqrt(2), this project's standard UCB1 exploration constant


def _make_node(state: frozenset, task, heuristic) -> MCTSNode:
    solved = _is_goal(state, task)
    own_h = 0.0 if solved else float(heuristic(make_root_node(state)))
    return MCTSNode(
        state=state,
        own_h=own_h,
        untried_moves=[] if solved else _legal_moves(state, task),
        is_terminal=solved,
    )


def create_root(start_state: frozenset, task, merge_enabled: bool, heuristic) -> MCTSGraph:
    root = _make_node(start_state, task, heuristic)
    if merge_enabled:
        return MCTSGraph(nodes={start_state: root}, root_key=start_state)
    return MCTSGraph(nodes={0: root}, root_key=0, next_id=1)


def _guct_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    """f_GUCT(n) = h_GUCT(n) - c*sqrt(2*log|L(p)| / |L(n)|), minimized."""
    if child.n_visits == 0:
        return float("-inf")
    h_mean = child.sum_h / child.n_visits
    return h_mean - c * math.sqrt(2.0 * math.log(parent.n_visits) / child.n_visits)


def select(graph: MCTSGraph, config: GUCTConfig) -> tuple[list[object], list]:
    """Cycle-safe: restricts the argmin to children not already on this
    path, same reasoning as every reversible-domain module in this project."""
    path = [graph.root_key]
    path_set = {graph.root_key}
    path_moves: list = []
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        candidates = {
            m: e for m, e in node.children.items() if e.child_key not in path_set
        }
        if not candidates:
            break
        best_move = min(
            candidates,
            key=lambda m: _guct_score(node, candidates[m], graph.nodes[candidates[m].child_key], config.c),
        )
        edge = candidates[best_move]
        path.append(edge.child_key)
        path_set.add(edge.child_key)
        path_moves.append(best_move)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def expand(
    graph: MCTSGraph, leaf_key: object, task, config: GUCTConfig, rng: random.Random, heuristic,
) -> tuple[object, object]:
    node = graph.nodes[leaf_key]
    op = rng.choice(node.untried_moves)
    node.untried_moves.remove(op)
    new_state = _apply_move(node.state, op)

    if config.merge_enabled:
        child_key = new_state
    else:
        child_key = graph.next_id
        graph.next_id += 1
    if child_key not in graph.nodes:
        graph.nodes[child_key] = _make_node(new_state, task, heuristic)
    node.children[op] = MCTSEdge(child_key=child_key)
    return child_key, op


def backup(graph: MCTSGraph, path: list, path_moves: list, value: float) -> None:
    """Monte Carlo backup (Definition 4): each node's h_GUCT is the running
    *mean* of every leaf value ever backed up through it, not a minimum."""
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.sum_h += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            parent.children[path_moves[i - 1]].n_edge += 1


def run_search(
    start_state: frozenset, task, config: GUCTConfig, budget: int, rng: random.Random, heuristic,
) -> MCTSGraph:
    """budget counts node evaluations (one heuristic call per expansion),
    matching this project's usual accounting. No rollout -- evaluation is
    one heuristic call, same as GUCT-Uniform."""
    graph = create_root(start_state, task, config.merge_enabled, heuristic)
    expansions_used = 0
    iterations = 0
    max_iterations = 20 * budget
    while expansions_used < budget and iterations < max_iterations:
        iterations += 1
        path, path_moves = select(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.is_terminal:
            backup(graph, path, path_moves, 0.0)
            continue
        if not leaf.untried_moves:
            # Cycle-blocked: unlike GUCT-Uniform, Monte Carlo backup's mean
            # is self-limiting (more samples shrink the exploration term
            # instead of growing it), so reusing this leaf's own fixed
            # heuristic value as one more free sample is safe here.
            backup(graph, path, path_moves, leaf.own_h)
            continue
        child_key, op = expand(graph, path[-1], task, config, rng, heuristic)
        expansions_used += 1
        path.append(child_key)
        path_moves.append(op)
        child = graph.nodes[child_key]
        backup(graph, path, path_moves, child.own_h)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal for n in graph.nodes.values())
