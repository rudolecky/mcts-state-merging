"""Classical Monte Carlo Tree Search on the sliding-tile (N-)puzzle -- no
LLM, no learned metric. Second no-LLM merge-vs-tree domain after Connect
Four (classical_mcts.py), chosen specifically to raise transposition
density: sliding-puzzle moves commute far more than Connect Four's
column-drops, so genuine transpositions should be much more common here.

Single-agent, unlike Connect Four: no opponent, no hero-relative
win/draw/loss -- a state is either the goal or it isn't. Node keying is
still the merge switch, structurally identical to classical_mcts.py:
treatment keys nodes by the state tuple itself (already canonical, see
sliding_puzzle_engine.py), baseline keys by the literal move-path from the
root.

Selection uses classical UCB1, not PUCT, for the same reason as
classical_mcts.py: no policy network exists here to supply a prior.

Unlike Connect Four, moves here are reversible (undoing a move returns the
prior state), so the merged graph can contain genuine cycles -- a fully
expanded node can have some children that are ancestors already in the
current select() path. `select()` restricts UCB1's argmax to children NOT
already in the path (an unrestricted argmax could deterministically prefer
a cycling child forever -- an infinite loop, not just a bad choice), and
only gives up on a node once *every* child would cycle back (rare: the
underlying state graph has no dead ends, so this means "temporarily
nothing new reachable from here without backtracking," not "stuck for
good"). `run_search` treats that give-up case like a terminal re-visit --
one more rollout from the node's own state, backed up without spending
expansion budget, rather than a crash trying to `expand()` a node with no
untried moves left.

The UCB1/select/backup logic below is intentionally duplicated from
classical_mcts.py rather than factored into a shared module -- see
cheerful-jumping-moler.md's design section for why (classical_mcts.py is a
closed, already-reported result; refactoring it to serve a second call
site isn't worth re-verifying that finding for a ~35-line saving).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets.sliding_puzzle_engine import apply_move, goal_state, legal_moves


@dataclass
class MCTSEdge:
    child_key: tuple
    n_edge: int = 0


@dataclass
class MCTSNode:
    state: tuple[int, ...]
    n_visits: int = 0
    w_value: float = 0.0
    children: dict[int, MCTSEdge] = field(default_factory=dict)
    untried_moves: list[int] = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float | None = None  # 1.0 iff this state is the goal


@dataclass
class MCTSGraph:
    nodes: dict[tuple, MCTSNode]
    root_key: tuple


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    c: float = 1.4  # ~sqrt(2), the standard UCB1 exploration constant
    value_source: str = "rollout"  # "rollout" | "heuristic" -- see _manhattan_heuristic


def _make_node(state: tuple[int, ...], width: int, height: int) -> MCTSNode:
    is_terminal = state == goal_state(width, height)
    return MCTSNode(
        state=state,
        untried_moves=[] if is_terminal else legal_moves(state, width, height),
        is_terminal=is_terminal,
        terminal_value=1.0 if is_terminal else None,
    )


def create_root(start_state: tuple[int, ...], width: int, height: int, merge_enabled: bool) -> MCTSGraph:
    root_key = start_state if merge_enabled else ()
    root = MCTSNode(state=start_state, untried_moves=legal_moves(start_state, width, height))
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(parent.n_visits) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list[tuple], tuple[int, ...]]:
    path = [graph.root_key]
    path_moves: tuple[int, ...] = ()
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        candidates = {m: e for m, e in node.children.items() if e.child_key not in path}
        if not candidates:
            break  # every child would revisit an ancestor on this path -- genuinely stuck, not just suboptimal
        best_move = max(
            candidates,
            key=lambda m: _ucb1_score(node, candidates[m], graph.nodes[candidates[m].child_key], config.c),
        )
        edge = candidates[best_move]
        path.append(edge.child_key)
        path_moves = path_moves + (best_move,)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def expand(
    graph: MCTSGraph, leaf_key: tuple, path_moves: tuple[int, ...],
    width: int, height: int, config: ClassicalMCTSConfig, rng: random.Random,
) -> tuple:
    """Add one untried child of the leaf (or reuse an existing node with the
    same state, in merge mode -- a real transposition hit)."""
    node = graph.nodes[leaf_key]
    move = rng.choice(node.untried_moves)
    node.untried_moves.remove(move)
    new_state = apply_move(node.state, move)

    child_key = new_state if config.merge_enabled else path_moves + (move,)
    if child_key not in graph.nodes:
        graph.nodes[child_key] = _make_node(new_state, width, height)
    node.children[move] = MCTSEdge(child_key=child_key)
    return child_key


def simulate(state: tuple[int, ...], width: int, height: int, rollout_depth: int, rng: random.Random) -> float:
    """Uniformly random legal moves for up to `rollout_depth` steps. 1.0 iff
    the goal is reached during the walk, else 0.0. Unlike Connect Four
    there's no natural "board fills" terminal bound, so the depth cap is an
    explicit parameter."""
    goal = goal_state(width, height)
    if state == goal:
        return 1.0
    for _ in range(rollout_depth):
        moves = legal_moves(state, width, height)
        state = apply_move(state, rng.choice(moves))
        if state == goal:
            return 1.0
    return 0.0


def _manhattan_heuristic(state: tuple[int, ...], width: int) -> int:
    """Sum of each tile's Manhattan distance from its goal cell. Goal
    position of tile value v is always index v-1 (see goal_state), so no
    lookup table is needed. Standard, admissible."""
    total = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        goal_r, goal_c = divmod(v - 1, width)
        r, c = divmod(i, width)
        total += abs(r - goal_r) + abs(c - goal_c)
    return total


def _heuristic_value(state: tuple[int, ...], width: int) -> float:
    return 1.0 / (1.0 + _manhattan_heuristic(state, width))


def _leaf_value(
    state: tuple[int, ...], width: int, height: int, rollout_depth: int,
    config: ClassicalMCTSConfig, rng: random.Random,
) -> float:
    if config.value_source == "heuristic":
        return _heuristic_value(state, width)
    return simulate(state, width, height, rollout_depth, rng)


def backup(graph: MCTSGraph, path: list[tuple], value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            move = next(m for m, e in parent.children.items() if e.child_key == key)
            parent.children[move].n_edge += 1


def run_search(
    start_state: tuple[int, ...], width: int, height: int,
    config: ClassicalMCTSConfig, budget: int, rollout_depth: int, rng: random.Random,
) -> MCTSGraph:
    """budget counts EXPANSIONS (new-node attempts), matching
    classical_mcts.py's own accounting exactly."""
    graph = create_root(start_state, width, height, config.merge_enabled)
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
            # select() broke out of a cycle: leaf is fully expanded with
            # nowhere new to go. Refine its own estimate with one more
            # rollout instead of expanding -- free, like the terminal case.
            value = _leaf_value(leaf.state, width, height, rollout_depth, config, rng)
            backup(graph, path, value)
            continue
        child_key = expand(graph, path[-1], path_moves, width, height, config, rng)
        expansions_used += 1
        path.append(child_key)
        child = graph.nodes[child_key]
        value = child.terminal_value if child.is_terminal else _leaf_value(child.state, width, height, rollout_depth, config, rng)
        backup(graph, path, value)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal and n.terminal_value == 1.0 for n in graph.nodes.values())


def run_random_search(start_state: tuple[int, ...], width: int, height: int, budget: int, rollout_depth: int, rng: random.Random) -> bool:
    """The "no MCTS at all" ablation (see RANDOM_VS_MCTS_FINDINGS.md):
    `budget` independent random rollouts from the start state, no tree, no
    UCB1, reusing this module's own `simulate()`."""
    for _ in range(budget):
        if simulate(start_state, width, height, rollout_depth, rng) == 1.0:
            return True
    return False
