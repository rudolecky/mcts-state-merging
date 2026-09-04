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
matching that example's target output grid, summed across examples, and
evaluated at whichever context index comes closest (via
`engine.goal_progress`).

It used to read `ctx[-1]`, the most recently produced value, which was a
genuine merge/tree asymmetry: `canonical_key` is order-independent and so
does not determine which value is last, meaning the merge arm scored a node
with whichever ordering happened to create it while the tree arm scored
each ordering separately. That matters more here than anywhere else in the
ARC work, because GUCT-Uniform is entirely h-driven -- there are no
rollouts to wash it out. Evaluating at the best index makes h a function of
the merge key and restores `h == 0.0` iff `is_goal` in both directions.

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
    """Thin alias for `engine.grid_overlap_fraction`, where this function's
    body (and the real jagged-tuple crash it was hardened against) now
    lives, so the classical module's graded reward and this heuristic
    share one implementation instead of two copies that can drift."""
    return engine.grid_overlap_fraction(candidate, target)


def _arc_heuristic(program_state: engine.ProgramState, target_outputs: tuple) -> float:
    """Per-example mismatch summed across examples, evaluated at whichever
    context index comes closest to the targets.

    Reading `ctx[-1]` instead -- the most recently produced value -- was a
    real merge/tree asymmetry, and a worse one than the `is_goal` version of
    the same mistake. `canonical_key` is order-independent, so it does not
    determine which value is last; the merge arm therefore kept whichever
    ordering created the node first while the tree arm got each ordering's
    own h. GUCT-Uniform is entirely h-driven (no rollouts), so the two arms
    were scoring different things everywhere, not just at the goal test.

    Taking the best index makes h a function of the merge key, since
    `goal_progress` depends only on the set of per-example value tuples.
    It also restores the `h == 0.0` iff `is_goal` equivalence exactly, in
    both directions, which the `ctx[-1]` form only had in one."""
    return len(program_state.contexts) * (1.0 - engine.goal_progress(program_state, target_outputs))


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
    stats: dict | None = None,
) -> MCTSGraph:
    """budget counts EXPANSIONS. No rollout_depth -- evaluation is one
    heuristic call, never a rollout. Same closed-edge divergence-prevention
    logic as every other GUCT-Uniform port in this project, plus one extra
    case specific to this domain: expand() can return None even when
    leaf.untried_moves was non-empty at entry (every remaining move failed
    at real execution) -- treated identically to "no untried moves left",
    since by the time expand() returns None the leaf's untried_moves list
    has already been fully consumed internally.

    stats, if provided, gets "new_nodes"/"merge_hits" counters incremented
    -- see RUBIKS_CUBE_FINDINGS.md's mechanism section, ported here (and to
    guct_uniform_blocksworld.py) to compare budget accounting directly
    across the domain that shows a persistent gap (Blocksworld) and one
    that shows parity (this one)."""
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
        nodes_before = len(graph.nodes)
        expanded = expand(graph, path[-1], target_outputs, config, rng)
        if expanded is None:
            if len(path) == 1:
                break
            parent = graph.nodes[path[-2]]
            parent.children[path_moves[-1]].closed = True
            continue
        child_key, move = expanded
        expansions_used += 1
        if stats is not None:
            if len(graph.nodes) > nodes_before:
                stats["new_nodes"] = stats.get("new_nodes", 0) + 1
            else:
                stats["merge_hits"] = stats.get("merge_hits", 0) + 1
        path.append(child_key)
        path_moves.append(move)
        child = graph.nodes[child_key]
        backup(graph, path, path_moves, child.own_h)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal for n in graph.nodes.values())
