"""Classical blind UCB1-MCTS over `arc_program_engine`'s program-synthesis
search space -- no heuristic exists here any more than it did for
Blocksworld's blind arc (`classical_mcts_blocksworld.py`), so this mirrors
that module's shape closely: random rollout up to a depth cap, checking for
an exact goal match after every step, reward 1.0/0.0.

Cycle-safety IS needed here after all, for a reason specific to how the
merge-mode key is computed. `canonical_key` is a *set* of a state's
(type, value) entries -- and if a new move produces a value that happens to
duplicate one already in the context (e.g. hmirror twice reaching a value
equal to something already computed another way), the resulting set is
identical to the parent's, since a set doesn't grow when you add an element
it already contains. That makes the "child" node the same node as its own
parent -- a genuine self-loop, confirmed directly (traced via a hung
`select()` call spinning through a one-node `children` dict forever) rather
than assumed away. `select()` uses the same cycle-safe path-restriction
this project's reversible domains already use; MC-average backup makes the
"reuse a free rollout on cycle-block" fallback safe here (no
GUCT-Uniform-style runaway-divergence risk -- that was specific to
LCB1-Uniform's non-self-limiting exploration term).

Node keying: `arc_program_engine.canonical_key(state)` in merge mode (order-
independent -- two different primitive orderings reaching the same
available values are the same node), a plain incrementing counter in tree
mode (same performance-motivated choice as `guct_uniform_blocksworld.py`'s
baseline keys -- program states can involve large nested values, and a
counter avoids ever re-hashing a growing structure).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets import arc_program_engine as engine


@dataclass
class MCTSEdge:
    child_key: object
    n_edge: int = 0


@dataclass
class MCTSNode:
    program_state: engine.ProgramState
    n_visits: int = 0
    w_value: float = 0.0
    children: dict = field(default_factory=dict)  # move -> MCTSEdge
    untried_moves: list = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class MCTSGraph:
    nodes: dict
    root_key: object
    next_id: int = 0


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    c: float = 1.4
    moves_per_node: int = 25


def _make_node(program_state: engine.ProgramState, target_outputs: tuple, config: ClassicalMCTSConfig, rng: random.Random) -> MCTSNode:
    solved = engine.is_goal(program_state, target_outputs)
    untried = [] if solved else engine.legal_moves(program_state, rng, config.moves_per_node)
    return MCTSNode(program_state=program_state, untried_moves=untried, is_terminal=solved)


def create_root(train_inputs: tuple, target_outputs: tuple, config: ClassicalMCTSConfig, rng: random.Random) -> MCTSGraph:
    state = engine.create_initial_state(train_inputs)
    root = _make_node(state, target_outputs, config, rng)
    if config.merge_enabled:
        root_key = engine.canonical_key(state)
        return MCTSGraph(nodes={root_key: root}, root_key=root_key)
    return MCTSGraph(nodes={0: root}, root_key=0, next_id=1)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(max(parent.n_visits, 1)) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list, list]:
    """Cycle-safe: restricts the argmax to children not already on this
    path (see this module's docstring for why a self-loop is a real,
    confirmed possibility here, not a theoretical one)."""
    path = [graph.root_key]
    path_set = {graph.root_key}
    path_moves: list = []
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        candidates = {m: e for m, e in node.children.items() if e.child_key not in path_set}
        if not candidates:
            break
        best_move = max(
            candidates,
            key=lambda m: _ucb1_score(node, candidates[m], graph.nodes[candidates[m].child_key], config.c),
        )
        edge = candidates[best_move]
        path.append(edge.child_key)
        path_set.add(edge.child_key)
        path_moves.append(best_move)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def safe_apply_move(state: engine.ProgramState, move):
    """`legal_moves` only checks that argument *types* line up -- it doesn't
    verify a bound function's arity/shape matches how a closure-consumer
    will actually call it (e.g. offering a 2-argument function where
    `argmin`'s 1-argument compfunc is expected), since that finer-grained
    check would need much more of a real type system than this pilot's
    scope calls for. Real execution catches what the type tags miss --
    returns None on any runtime failure rather than raising, exactly the
    same "generate then filter" pattern real program-synthesis systems use
    for a DSL that isn't statically guaranteed type-safe."""
    try:
        return engine.apply_move(state, move)
    except Exception:
        return None


def expand(graph: MCTSGraph, leaf_key, target_outputs: tuple, config: ClassicalMCTSConfig, rng: random.Random):
    """Tries untried moves (discarding ones that fail at real execution --
    see `safe_apply_move`) until one succeeds. Returns None if every
    remaining untried move fails -- the caller falls back to `simulate()`
    on this leaf, the same as if it had no untried moves to begin with."""
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
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            parent.children[path_moves[i - 1]].n_edge += 1


def simulate(program_state: engine.ProgramState, target_outputs: tuple, rollout_depth: int, config: ClassicalMCTSConfig, rng: random.Random) -> float:
    if engine.is_goal(program_state, target_outputs):
        return 1.0
    state = program_state
    for _ in range(rollout_depth):
        moves = engine.legal_moves(state, rng, config.moves_per_node)
        rng.shuffle(moves)
        next_state = None
        for move in moves:
            next_state = safe_apply_move(state, move)
            if next_state is not None:
                break
        if next_state is None:
            return 0.0
        state = next_state
        if engine.is_goal(state, target_outputs):
            return 1.0
    return 0.0


def run_search(train_inputs: tuple, target_outputs: tuple, config: ClassicalMCTSConfig, budget: int, rollout_depth: int, rng: random.Random) -> MCTSGraph:
    graph = create_root(train_inputs, target_outputs, config, rng)
    expansions_used = 0
    iterations = 0
    max_iterations = 20 * budget
    while expansions_used < budget and iterations < max_iterations:
        iterations += 1
        path, path_moves = select(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.is_terminal:
            backup(graph, path, path_moves, 1.0)
            continue
        if not leaf.untried_moves:
            value = simulate(leaf.program_state, target_outputs, rollout_depth, config, rng)
            backup(graph, path, path_moves, value)
            continue
        expanded = expand(graph, path[-1], target_outputs, config, rng)
        if expanded is None:
            # every untried move failed at real execution (see safe_apply_move) --
            # equivalent to the leaf having no untried moves after all.
            value = simulate(leaf.program_state, target_outputs, rollout_depth, config, rng)
            backup(graph, path, path_moves, value)
            continue
        child_key, move = expanded
        expansions_used += 1
        path.append(child_key)
        path_moves.append(move)
        child = graph.nodes[child_key]
        value = 1.0 if child.is_terminal else simulate(child.program_state, target_outputs, rollout_depth, config, rng)
        backup(graph, path, path_moves, value)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal for n in graph.nodes.values())


def run_random_search(train_inputs: tuple, target_outputs: tuple, budget: int, rollout_depth: int, rng: random.Random) -> bool:
    """The "no MCTS at all" ablation: `budget` independent random rollouts
    from the initial state, no tree, no UCB1, nothing carried between
    attempts -- isolates what the tree/bandit structure itself contributes
    (separate from merge vs. tree), by reusing the exact same `simulate()`
    used inside `run_search`'s own rollouts, called cold every time instead
    of from an accumulating, UCB1-guided tree."""
    config = ClassicalMCTSConfig(merge_enabled=False)  # merge_enabled unused by simulate() itself
    initial_state = engine.create_initial_state(train_inputs)
    for _ in range(budget):
        if simulate(initial_state, target_outputs, rollout_depth, config, rng) == 1.0:
            return True
    return False
