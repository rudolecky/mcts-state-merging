"""Classical Monte Carlo Tree Search on Sokoban -- no LLM, no learned
metric. Fourth no-LLM merge-vs-tree domain: single-agent (like the
8-puzzle), but only *some* moves are reversible -- walking is fully
reversible, pushing a box is directional and only reversible when the
board geometry allows walking back around to the opposite side. A genuine
intermediate point on the redundancy-density axis between the 8-puzzle
(every move trivially reversible) and Morris (adversarial, mostly none).

Cycles are possible here too (a walk-only round trip, or a push undone by
walking around and pushing back), so this reuses the same cycle-safe
select()/run_search fix as classical_mcts_puzzle.py and
classical_mcts_morris.py -- the fourth call site of this small,
deliberately-duplicated core (see cheerful-jumping-moler.md's design
section: all three prior modules are closed, already-reported results).

No deadlock detection anywhere: an unsolvable-from-here state (box wedged
against a wall with no way to push it further) simply never gets solved
within rollout_depth/budget, exactly like any other hard state -- the same
convention every domain in this project already uses for "didn't solve in
time," not a gap that needs filling.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets.sokoban_engine import GOAL, WIDTH, apply_move, is_solved as _boxes_solved, legal_moves


@dataclass
class MCTSEdge:
    child_key: tuple
    n_edge: int = 0


@dataclass
class MCTSNode:
    state: tuple  # (player_cell, (box_cells,))
    n_visits: int = 0
    w_value: float = 0.0
    children: dict[tuple, MCTSEdge] = field(default_factory=dict)
    untried_moves: list[tuple] = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float | None = None  # 1.0 iff solved


@dataclass
class MCTSGraph:
    nodes: dict[tuple, MCTSNode]
    root_key: tuple


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    c: float = 1.4  # ~sqrt(2), the standard UCB1 exploration constant
    value_source: str = "rollout"  # "rollout" | "heuristic" -- see _box_goal_heuristic


def _make_node(state: tuple) -> MCTSNode:
    solved = _boxes_solved(state[1])
    return MCTSNode(
        state=state,
        untried_moves=[] if solved else legal_moves(state),
        is_terminal=solved,
        terminal_value=1.0 if solved else None,
    )


def create_root(start_state: tuple, merge_enabled: bool) -> MCTSGraph:
    root_key = start_state if merge_enabled else ()
    root = MCTSNode(state=start_state, untried_moves=legal_moves(start_state))
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(parent.n_visits) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list[tuple], tuple]:
    path = [graph.root_key]
    path_moves: tuple = ()
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        candidates = {m: e for m, e in node.children.items() if e.child_key not in path}
        if not candidates:
            break
        best_move = max(
            candidates,
            key=lambda m: _ucb1_score(node, candidates[m], graph.nodes[candidates[m].child_key], config.c),
        )
        edge = candidates[best_move]
        path.append(edge.child_key)
        path_moves = path_moves + (best_move,)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def expand(graph: MCTSGraph, leaf_key: tuple, path_moves: tuple, config: ClassicalMCTSConfig, rng: random.Random) -> tuple:
    node = graph.nodes[leaf_key]
    move = rng.choice(node.untried_moves)
    node.untried_moves.remove(move)
    new_state = apply_move(node.state, move)

    child_key = new_state if config.merge_enabled else path_moves + (move,)
    if child_key not in graph.nodes:
        graph.nodes[child_key] = _make_node(new_state)
    node.children[move] = MCTSEdge(child_key=child_key)
    return child_key


def simulate(state: tuple, rollout_depth: int, rng: random.Random) -> float:
    """Uniformly random legal moves (walk or push, whichever a given
    direction happens to be) for up to `rollout_depth` steps. 1.0 iff
    solved during the walk, else 0.0."""
    if _boxes_solved(state[1]):
        return 1.0
    for _ in range(rollout_depth):
        moves = legal_moves(state)
        state = apply_move(state, rng.choice(moves))
        if _boxes_solved(state[1]):
            return 1.0
    return 0.0


def _box_goal_heuristic(state: tuple) -> int:
    """Sum of box-to-goal Manhattan distances, ignoring walls. This level's
    interior has no walls (only the outer border, per sokoban_engine.py's
    own docstring), so ignoring walls doesn't overestimate here."""
    goal_r, goal_c = divmod(GOAL, WIDTH)
    total = 0
    for box in state[1]:
        r, c = divmod(box, WIDTH)
        total += abs(r - goal_r) + abs(c - goal_c)
    return total


def _player_box_heuristic(state: tuple) -> int:
    """Player's Manhattan distance to its nearest box. Box-goal distance
    alone is unchanged by any walk-only move (the box doesn't move), which
    gives UCB1 zero discrimination among most of a state's legal moves
    (confirmed empirically: pure box-goal heuristic solved 0/50 across a
    (c, budget) calibration sweep). Adding this term gives walking moves a
    real gradient too, breaking that degeneracy -- a standard, simple
    addition to a box-distance-only Sokoban heuristic."""
    player = state[0]
    boxes = state[1]
    if not boxes:
        return 0
    pr, pc = divmod(player, WIDTH)
    return min(abs(pr - divmod(b, WIDTH)[0]) + abs(pc - divmod(b, WIDTH)[1]) for b in boxes)


def _heuristic_value(state: tuple) -> float:
    h = _box_goal_heuristic(state) + _player_box_heuristic(state)
    return 1.0 / (1.0 + h)


def _leaf_value(state: tuple, rollout_depth: int, config: ClassicalMCTSConfig, rng: random.Random) -> float:
    if config.value_source == "heuristic":
        return _heuristic_value(state)
    return simulate(state, rollout_depth, rng)


def backup(graph: MCTSGraph, path: list[tuple], value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            move = next(m for m, e in parent.children.items() if e.child_key == key)
            parent.children[move].n_edge += 1


def run_search(start_state: tuple, config: ClassicalMCTSConfig, budget: int, rollout_depth: int, rng: random.Random) -> MCTSGraph:
    graph = create_root(start_state, config.merge_enabled)
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
        if not leaf.untried_moves:
            value = _leaf_value(leaf.state, rollout_depth, config, rng)
            backup(graph, path, value)
            continue
        child_key = expand(graph, path[-1], path_moves, config, rng)
        expansions_used += 1
        path.append(child_key)
        child = graph.nodes[child_key]
        value = child.terminal_value if child.is_terminal else _leaf_value(child.state, rollout_depth, config, rng)
        backup(graph, path, value)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal and n.terminal_value == 1.0 for n in graph.nodes.values())


def run_random_search(start_state: tuple, budget: int, rollout_depth: int, rng: random.Random) -> bool:
    """The "no MCTS at all" ablation (see RANDOM_VS_MCTS_FINDINGS.md):
    `budget` independent random rollouts from the start state, no tree, no
    UCB1, reusing this module's own `simulate()`."""
    for _ in range(budget):
        if simulate(start_state, rollout_depth, rng) == 1.0:
            return True
    return False
