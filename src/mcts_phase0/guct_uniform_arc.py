"""GreedyUCT-Uniform (arXiv:2405.18248) on ARC-AGI program synthesis --
ports `guct_uniform_blocksworld.py`'s exact algorithm to this domain,
extending the generalization test past the 8-puzzle/Sokoban/cube (all
solve-rate-neutral under this backup rule) to a second domain in
Blocksworld's own class: extreme transposition density where blind
UCB1+MC-average shows merging is critical, and where GUCT-Uniform itself
still showed a persistent gap on Blocksworld (`BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`)
rather than the parity seen on the other three. Is Blocksworld unique, or
does ARC-AGI -- this project's other large-merge-benefit domain -- show
the same persistent gap under GUCT-Uniform too?

Reward: unlike every other GUCT-Uniform port, ARC-AGI has no exact BFS
oracle (the DSL program space is unbounded, no exhaustive reachability
table is feasible) and no pre-existing heuristic at all
(`classical_mcts_arc_program.py` uses blind 0/1 rollout only). `_arc_heuristic`
is new: for each training example, 1 minus the fraction of grid cells
where the program's *current final context value* (`ctx[-1]`, the exact
value `is_goal` itself checks) matches that example's target output grid,
summed across examples. Deliberately restricted to `ctx[-1]` only (not
scanning earlier context values for a partial match elsewhere in the
program) so that h=0.0 if and only if `is_goal` is True -- scanning every
context value risked a state where some *other*, non-final value happened
to match a target while the actual current output didn't, spuriously
zeroing the heuristic at a state `is_goal` disagrees is solved.

Node keying mirrors `classical_mcts_arc_program.py`'s own convention
exactly: `canonical_key` (order-independent set of the program's
(type, value) entries) for merge, an incrementing integer counter for
baseline (not a path-move tuple -- program states can hold large nested
values, expensive to rehash repeatedly as a growing path key, the same
reasoning `guct_uniform_blocksworld.py`'s own baseline keys already used).
Move sampling and real-execution-failure handling
(`legal_moves`/`safe_apply_move`) are reused directly from the classical
module and `arc_program_engine`, not reimplemented.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .classical_mcts_arc_program import safe_apply_move
from .datasets import arc_program_engine as engine


@dataclass
class MCTSEdge:
    child_key: object
    t: int = 0
    l_hat: float = float("inf")
    u_hat: float = float("-inf")
    closed: bool = False


@dataclass
class MCTSNode:
    program_state: engine.ProgramState
    own_h: float = 0.0
    h_gbfs: float = 0.0
    n_visits: int = 0
    children: dict = field(default_factory=dict)  # move -> MCTSEdge
    untried_moves: list = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class MCTSGraph:
    nodes: dict
    root_key: object
    next_id: int = 0


@dataclass
class GUCTUniformConfig:
    merge_enabled: bool
    moves_per_node: int = 25


def _grid_overlap_fraction(candidate, target) -> float:
    """Real bug caught during calibration: checking only that candidate[0]
    is a tuple isn't enough to confirm candidate is a Grid (Tuple[Tuple[Integer]])
    -- other DSL types are also tuples-of-tuples-looking in their first
    element (e.g. a jagged structure whose row 0 happens to be a nested
    tuple but a later "row" is a bare int), and crashed zip(r1, r2) with
    'int is not iterable' the first time a non-Grid tuple value reached
    this function. Every row is now validated as its own tuple, and every
    row's length is checked against target's corresponding row, not just
    row 0's -- any mismatch falls through to 0.0 (no partial credit)
    instead of crashing."""
    if not isinstance(candidate, tuple) or not candidate:
        return 0.0
    if not all(isinstance(row, tuple) for row in candidate):
        return 0.0
    if len(candidate) != len(target) or any(len(r1) != len(r2) for r1, r2 in zip(candidate, target)):
        return 0.0
    total = sum(len(row) for row in candidate)
    matches = sum(1 for r1, r2 in zip(candidate, target) for c1, c2 in zip(r1, r2) if c1 == c2)
    return matches / total


def _arc_heuristic(program_state: engine.ProgramState, target_outputs: tuple) -> float:
    return sum(
        1.0 - _grid_overlap_fraction(ctx[-1], target)
        for ctx, target in zip(program_state.contexts, target_outputs)
    )


def _make_node(program_state: engine.ProgramState, target_outputs: tuple, config: GUCTUniformConfig, rng: random.Random) -> MCTSNode:
    solved = engine.is_goal(program_state, target_outputs)
    own_h = 0.0 if solved else _arc_heuristic(program_state, target_outputs)
    untried = [] if solved else engine.legal_moves(program_state, rng, config.moves_per_node)
    return MCTSNode(program_state=program_state, own_h=own_h, h_gbfs=own_h, untried_moves=untried, is_terminal=solved)


def create_root(train_inputs: tuple, target_outputs: tuple, config: GUCTUniformConfig, rng: random.Random) -> MCTSGraph:
    state = engine.create_initial_state(train_inputs)
    root = _make_node(state, target_outputs, config, rng)
    if config.merge_enabled:
        root_key = engine.canonical_key(state)
        return MCTSGraph(nodes={root_key: root}, root_key=root_key)
    return MCTSGraph(nodes={0: root}, root_key=0, next_id=1)


def _lcb1_uniform_score(edge: MCTSEdge, T: int) -> float:
    if edge.t == 0:
        return float("-inf")
    mid = (edge.u_hat + edge.l_hat) / 2.0
    spread = edge.u_hat - edge.l_hat
    return mid - spread * math.sqrt(6.0 * edge.t * math.log(T))


def select(graph: MCTSGraph, config: GUCTUniformConfig) -> tuple[list, list]:
    path = [graph.root_key]
    path_set = {graph.root_key}
    path_moves: list = []
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        candidates = {
            m: e for m, e in node.children.items()
            if e.child_key not in path_set and not e.closed
        }
        if not candidates:
            break
        T = node.n_visits
        best_move = min(candidates, key=lambda m: _lcb1_uniform_score(candidates[m], T))
        edge = candidates[best_move]
        path.append(edge.child_key)
        path_set.add(edge.child_key)
        path_moves.append(best_move)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def expand(graph: MCTSGraph, leaf_key, target_outputs: tuple, config: GUCTUniformConfig, rng: random.Random):
    """Same real-execution-failure handling as classical_mcts_arc_program.py's
    own expand(): tries untried moves until one succeeds, returns None if
    every remaining one fails at real execution (safe_apply_move)."""
    node = graph.nodes[leaf_key]
    while node.untried_moves:
        move = rng.choice(node.untried_moves)
        node.untried_moves.remove(move)
        new_state = safe_apply_move(node.program_state, move)
        if new_state is None:
            continue
        if config.merge_enabled:
            child_key = engine.canonical_key(new_state)
        else:
            child_key = graph.next_id
            graph.next_id += 1
        if child_key not in graph.nodes:
            graph.nodes[child_key] = _make_node(new_state, target_outputs, config, rng)
        node.children[move] = MCTSEdge(child_key=child_key)
        return child_key, move
    return None


def backup(graph: MCTSGraph, path: list, path_moves: list, value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.h_gbfs = min(node.h_gbfs, value)
        node.n_visits += 1
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            edge = parent.children[path_moves[i - 1]]
            edge.t += 1
            edge.l_hat = min(edge.l_hat, value)
            edge.u_hat = max(edge.u_hat, value)


def run_search(
    train_inputs: tuple, target_outputs: tuple, config: GUCTUniformConfig, budget: int, rng: random.Random,
) -> MCTSGraph:
    """budget counts EXPANSIONS. No rollout_depth -- evaluation is one
    heuristic call, never a rollout. Same closed-edge divergence-prevention
    logic as every other GUCT-Uniform port in this project, plus one extra
    case specific to this domain: expand() can return None even when
    leaf.untried_moves was non-empty at entry (every remaining move failed
    at real execution) -- treated identically to "no untried moves left",
    since by the time expand() returns None the leaf's untried_moves list
    has already been fully consumed internally."""
    graph = create_root(train_inputs, target_outputs, config, rng)
    expansions_used = 0
    iterations = 0
    max_iterations = 20 * budget
    while expansions_used < budget and iterations < max_iterations:
        iterations += 1
        path, path_moves = select(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.is_terminal:
            backup(graph, path, path_moves, 0.0)
            if len(path) > 1:
                graph.nodes[path[-2]].children[path_moves[-1]].closed = True
            continue
        if not leaf.untried_moves:
            if len(path) == 1:
                break
            parent = graph.nodes[path[-2]]
            parent.children[path_moves[-1]].closed = True
            continue
        expanded = expand(graph, path[-1], target_outputs, config, rng)
        if expanded is None:
            if len(path) == 1:
                break
            parent = graph.nodes[path[-2]]
            parent.children[path_moves[-1]].closed = True
            continue
        child_key, move = expanded
        expansions_used += 1
        path.append(child_key)
        path_moves.append(move)
        child = graph.nodes[child_key]
        backup(graph, path, path_moves, child.own_h)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal for n in graph.nodes.values())
