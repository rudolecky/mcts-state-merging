"""Classical Monte Carlo Tree Search on the 2x2x2 Rubik's Cube -- no LLM,
no learned metric. Mirrors classical_mcts_puzzle.py's structure exactly
(single-agent, UCB1, MC-average backup, cycle-safe select() since every
move is invertible). Moves are labeled by index into
rubiks_cube_engine.ALL_MOVES (0-11), matching how classical_mcts_puzzle.py
labels moves by blank-swap index.

Two value-guidance arms, both meaningful because rubiks_cube_engine.py's
full-fidelity design gives an exact BFS-distance oracle for free (unlike
the 8-puzzle/Sokoban/Blocksworld's realistic-heuristic-only builds):
"heuristic" (count of corners not in solved position+orientation, the
cube analogue of "misplaced tiles") and "oracle" (the exact BFS distance
itself, mirroring search_oracle.py's ProsQA ablation). Both map through
the same 1/(1+h) convention used by the classical-domain heuristic sweep
in REAL_HEURISTIC_MERGE_FINDINGS.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets import rubiks_cube_engine as rc


@dataclass
class MCTSEdge:
    child_key: object
    n_edge: int = 0


@dataclass
class MCTSNode:
    state: rc.State
    n_visits: int = 0
    w_value: float = 0.0
    children: dict[int, MCTSEdge] = field(default_factory=dict)
    untried_moves: list[int] = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float | None = None  # 1.0 iff this state is the goal
    parent_count: int = 0  # distinct parents merged into this node -- see merge_parent_cap


@dataclass
class MCTSGraph:
    nodes: dict[object, MCTSNode]
    root_key: object


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    c: float = 1.4  # ~sqrt(2), the standard UCB1 exploration constant
    value_source: str = "rollout"  # "rollout" | "heuristic" | "oracle" -- see _leaf_value
    merge_parent_cap: int | None = None
    """None = uncapped merging (original behavior). An int caps how many
    distinct parents a single merged node may absorb -- see
    RUBIKS_CUBE_FINDINGS.md's mechanism section: uncapped merging let a
    small hub of heavily-revisited nodes near the start absorb 55-60% of
    the *entire* expansion budget at every difficulty/budget tested,
    without the search ever escaping toward the goal. Once a node hits the
    cap, the next expansion that would merge into it instead falls back to
    ordinary path-keying (a guaranteed-fresh node, exactly like
    merge_enabled=False for that one expansion) -- bounding how much
    budget any single hub can absorb while still letting less-popular
    transpositions merge normally. Found to be a cliff, not a dial:
    cap=1 recovers full performance only because it prevents any real
    pooling from ever happening (no second parent can attach); cap>=2
    reproduces the full failure at undiminished severity even though it
    measurably cuts wasted budget -- see merge_visit_cap for the
    follow-up hypothesis this motivated."""
    merge_visit_cap: int | None = None
    """None = uncapped. An int caps how many visits (n_visits, not
    distinct parents) a node may accumulate before further merges into it
    are refused, same fallback mechanism as merge_parent_cap. Tests the
    hypothesis that merge_parent_cap's cliff (cap=1 works only because it
    blocks pooling entirely; cap>=2 fails just as badly as uncapped) is
    really driven by UCB1's own visit-count dynamics -- a pooled node
    accumulates visits faster than any single-parent alternative, shrinks
    its own exploration bonus, and gets exploited disproportionately
    regardless of true quality, independent of how many distinct parents
    fed it those visits."""


def _make_node(state: rc.State) -> MCTSNode:
    is_terminal = rc.is_goal(state)
    return MCTSNode(
        state=state,
        untried_moves=[] if is_terminal else list(range(len(rc.ALL_MOVES))),
        is_terminal=is_terminal,
        terminal_value=1.0 if is_terminal else None,
    )


def create_root(start_state: rc.State, merge_enabled: bool) -> MCTSGraph:
    root_key = start_state if merge_enabled else ()
    root = MCTSNode(state=start_state, untried_moves=list(range(len(rc.ALL_MOVES))))
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(parent.n_visits) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list[object], tuple[int, ...]]:
    path = [graph.root_key]
    path_moves: tuple[int, ...] = ()
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        candidates = {m: e for m, e in node.children.items() if e.child_key not in path}
        if not candidates:
            break  # every child would revisit an ancestor on this path
        best_move = max(
            candidates,
            key=lambda m: _ucb1_score(node, candidates[m], graph.nodes[candidates[m].child_key], config.c),
        )
        edge = candidates[best_move]
        path.append(edge.child_key)
        path_moves = path_moves + (best_move,)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def _find_or_add_edge(parent: MCTSNode, move: int, child_key: object) -> None:
    parent.children[move] = MCTSEdge(child_key=child_key)


def expand(graph: MCTSGraph, leaf_key: object, path_moves: tuple[int, ...], config: ClassicalMCTSConfig, rng: random.Random) -> object:
    node = graph.nodes[leaf_key]
    move = rng.choice(node.untried_moves)
    node.untried_moves.remove(move)
    new_state = rc.apply_move(node.state, rc.ALL_MOVES[move])

    use_merge = config.merge_enabled
    if use_merge and new_state in graph.nodes:
        target = graph.nodes[new_state]
        if config.merge_parent_cap is not None and target.parent_count >= config.merge_parent_cap:
            use_merge = False  # this merge target is "full" (parent count) -- fresh path-keyed node
        elif config.merge_visit_cap is not None and target.n_visits >= config.merge_visit_cap:
            use_merge = False  # this merge target is "full" (visit count) -- fresh path-keyed node

    child_key = new_state if use_merge else path_moves + (move,)
    if child_key not in graph.nodes:
        graph.nodes[child_key] = _make_node(new_state)
    graph.nodes[child_key].parent_count += 1
    _find_or_add_edge(node, move, child_key)
    return child_key


def simulate(state: rc.State, rollout_depth: int, rng: random.Random) -> float:
    """Uniformly random legal moves for up to `rollout_depth` steps. 1.0 iff
    the goal is reached during the walk, else 0.0."""
    if rc.is_goal(state):
        return 1.0
    for _ in range(rollout_depth):
        state = rc.apply_move(state, rng.choice(rc.ALL_MOVES))
        if rc.is_goal(state):
            return 1.0
    return 0.0


def _misplaced_corners_heuristic(state: rc.State) -> int:
    perm, orient = state
    return sum(1 for p in range(8) if perm[p] != p or orient[p] != rc.IDENTITY)


def _value_from_h(h: float) -> float:
    return 1.0 / (1.0 + h)


def _leaf_value(state: rc.State, rollout_depth: int, config: ClassicalMCTSConfig, rng: random.Random, distances: dict | None) -> float:
    if config.value_source == "heuristic":
        return _value_from_h(_misplaced_corners_heuristic(state))
    if config.value_source == "oracle":
        return _value_from_h(distances[state])
    return simulate(state, rollout_depth, rng)


def backup(graph: MCTSGraph, path: list[object], value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            move = next(m for m, e in parent.children.items() if e.child_key == key)
            parent.children[move].n_edge += 1


def run_search(
    start_state: rc.State, config: ClassicalMCTSConfig, budget: int, rollout_depth: int, rng: random.Random,
    distances: dict | None = None, stats: dict | None = None,
) -> MCTSGraph:
    """budget counts EXPANSIONS, matching every other classical module's
    accounting exactly. distances is required when config.value_source ==
    "oracle" (the exact BFS-distance table, e.g. from
    rubiks_cube_engine.bfs_distances()). stats, if provided, gets
    "new_nodes"/"merge_hits" counters incremented -- see
    RUBIKS_CUBE_FINDINGS.md's mechanism section: this distinguishes an
    expansion that grows the tree's own distinct-node count from one that
    only adds another parent pointer to an already-known state."""
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
            value = _leaf_value(leaf.state, rollout_depth, config, rng, distances)
            backup(graph, path, value)
            continue
        nodes_before = len(graph.nodes)
        child_key = expand(graph, path[-1], path_moves, config, rng)
        expansions_used += 1
        if stats is not None:
            if len(graph.nodes) > nodes_before:
                stats["new_nodes"] = stats.get("new_nodes", 0) + 1
            else:
                stats["merge_hits"] = stats.get("merge_hits", 0) + 1
        path.append(child_key)
        child = graph.nodes[child_key]
        value = child.terminal_value if child.is_terminal else _leaf_value(child.state, rollout_depth, config, rng, distances)
        backup(graph, path, value)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal and n.terminal_value == 1.0 for n in graph.nodes.values())


def run_random_search(start_state: rc.State, budget: int, rollout_depth: int, rng: random.Random) -> bool:
    """The "no MCTS at all" ablation (see RANDOM_VS_MCTS_FINDINGS.md):
    `budget` independent random rollouts from the start state, no tree, no
    UCB1, reusing this module's own `simulate()`."""
    for _ in range(budget):
        if simulate(start_state, rollout_depth, rng) == 1.0:
            return True
    return False
