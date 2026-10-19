"""Classical blind UCB1-MCTS over `arc_program_engine`'s program-synthesis
search space -- no heuristic exists here any more than it did for
Blocksworld's blind arc (`classical_mcts_blocksworld.py`), so this mirrors
that module's shape closely: random rollout up to a depth cap, checking for
a goal match after every step. Reward is 1.0/0.0 by default
(`reward_mode="binary"`) or graded by `engine.goal_progress`
(`reward_mode="graded"`) -- see `ClassicalMCTSConfig`.

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
    exhausted: bool = False
    """Reject-duplicates arm only: this node has no untried moves left and
    `select()` could not descend past it, so nothing new can ever come from
    it. Without marking it, it stays the most attractive frontier node its
    parent has and gets re-selected for rollout-only iterations forever --
    which burns the iteration allowance (20x budget) without spending a
    single expansion, so the arm silently runs at a fraction of the budget
    the other arms get. Marked nodes are excluded from `select()`."""


@dataclass
class MCTSGraph:
    nodes: dict
    root_key: object
    next_id: int = 0
    seen_state_keys: set = field(default_factory=set)
    """Only populated in the reject-duplicates arm, where node keys are
    path-based (so `nodes` cannot answer "has this state been seen")."""
    new_nodes: int = 0
    """Nodes created by an expansion. Excludes the root, matching
    `classical_mcts_rubiks.py`'s pre-existing `stats["new_nodes"]`
    convention, so the counter means the same thing in every module."""
    merge_hits: int = 0
    rejected: int = 0


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    c: float = 1.4
    moves_per_node: int = 25
    reward_mode: str = "binary"
    """"binary" (original): rollouts return 1.0 iff the goal was reached,
    else 0.0. "graded": rollouts return the best `engine.goal_progress`
    seen along the walk. Merging works by pooling statistics, and pooling
    zeros carries no information -- ARC is the sparsest-reward domain in
    this repo and shows its smallest merge effect, so reward density is a
    candidate mechanism, testable by running both modes."""
    merge_key_mode: str = "typed"
    """Equivalence relation for merge keys, passed to
    `engine.canonical_key`: "typed" (original) or "values" (coarser,
    observational-equivalence flavored). Also used by the
    reject-duplicates arm to decide what counts as a duplicate."""
    reject_duplicates: bool = False
    """Third arm, only meaningful with `merge_enabled=False`: compute the
    merge key but DISCARD a child whose state was already reached anywhere,
    instead of pooling into the existing node. Keeps the tree a tree, so
    no statistics are ever aggregated and the backup rule is untouched --
    which isolates "expansions land on distinct states" from "statistics
    from different paths get pooled". Those are the two things ordinary
    merging does at once, and the Rubik's Cube result says the second one
    is where the risk lives.

    Budget accounting: a rejected move never becomes an expansion, so it
    does not count against the expansion budget -- consistent with this
    project's convention that budget counts expansions. It does cost a real
    `apply_move`, though, so this arm does strictly more work per unit of
    budget than the other two; that difference shows up only in wall-clock,
    which is why wall-clock is reported next to every arm's node count."""



    def __post_init__(self):
        if self.merge_enabled and self.reject_duplicates:
            # These two do not compose, and the way they fail to compose
            # differs per module (one silently disables merging, another
            # silently ignores the rejection), so the combination is
            # refused rather than left to mean whatever the code happens
            # to do. `reject_duplicates` is an alternative to merging, not
            # a modifier of it.
            raise ValueError("reject_duplicates is only meaningful with merge_enabled=False")

def _make_node(program_state: engine.ProgramState, target_outputs: tuple, config: ClassicalMCTSConfig, rng: random.Random) -> MCTSNode:
    solved = engine.is_goal(program_state, target_outputs)
    untried = [] if solved else engine.legal_moves(program_state, rng, config.moves_per_node)
    return MCTSNode(program_state=program_state, untried_moves=untried, is_terminal=solved)


def create_root(train_inputs: tuple, target_outputs: tuple, config: ClassicalMCTSConfig, rng: random.Random) -> MCTSGraph:
    state = engine.create_initial_state(train_inputs)
    root = _make_node(state, target_outputs, config, rng)
    if config.merge_enabled:
        root_key = engine.canonical_key(state, config.merge_key_mode)
        return MCTSGraph(nodes={root_key: root}, root_key=root_key)
    seen = set()
    if config.reject_duplicates:
        seen.add(engine.canonical_key(state, config.merge_key_mode))
    return MCTSGraph(nodes={0: root}, root_key=0, next_id=1, seen_state_keys=seen)


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
        candidates = {m: e for m, e in node.children.items()
                      if e.child_key not in path_set and not graph.nodes[e.child_key].exhausted}
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
            child_key = engine.canonical_key(new_state, config.merge_key_mode)
        else:
            if config.reject_duplicates:
                state_key = engine.canonical_key(new_state, config.merge_key_mode)
                if state_key in graph.seen_state_keys:
                    graph.rejected += 1
                    continue  # admission filter: this state exists already, try another move
                graph.seen_state_keys.add(state_key)
            child_key = graph.next_id
            graph.next_id += 1
        if child_key not in graph.nodes:
            graph.nodes[child_key] = _make_node(new_state, target_outputs, config, rng)
            graph.new_nodes += 1
        else:
            graph.merge_hits += 1
        node.children[move] = MCTSEdge(child_key=child_key)
        return child_key, move
    return None


def _mark_exhausted(graph: MCTSGraph, leaf_key, leaf: MCTSNode, config: ClassicalMCTSConfig) -> bool:
    """Reject-duplicates arm only. Marks a dead frontier node so `select()`
    stops returning it, and reports whether the whole search is finished --
    which is the case exactly when the dead node is the root, since
    `select()` can only return an exhausted node when it had no
    non-exhausted alternative anywhere above it."""
    if not config.reject_duplicates:
        return False
    if leaf.untried_moves:
        return False
    if any(not graph.nodes[e.child_key].exhausted for e in leaf.children.values()):
        # Still has a live subtree. `select()` returns nodes that have BOTH
        # untried moves and existing children, so a node whose remaining
        # untried moves were all rejected is NOT dead -- marking it here
        # would make `select()` skip it and orphan every live descendant.
        return False
    leaf.exhausted = True
    return leaf_key == graph.root_key


def backup(graph: MCTSGraph, path: list, path_moves: list, value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            parent.children[path_moves[i - 1]].n_edge += 1


def simulate(program_state: engine.ProgramState, target_outputs: tuple, rollout_depth: int, config: ClassicalMCTSConfig, rng: random.Random) -> float:
    """Returns 1.0 iff the goal is reached during the walk. Otherwise 0.0
    under `reward_mode="binary"`, or the best `engine.goal_progress` seen
    at any point on the walk (including the starting state) under
    `reward_mode="graded"`. Reaching the goal implies progress 1.0, so the
    two modes agree exactly on solved rollouts and differ only in what an
    unsolved rollout reports."""
    graded = config.reward_mode == "graded"
    if engine.is_goal(program_state, target_outputs):
        return 1.0
    best = engine.goal_progress(program_state, target_outputs) if graded else 0.0
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
            return best
        state = next_state
        # Incremental checks: the walk already established that every
        # earlier index misses, and each step appends exactly one value, so
        # only that value can change the answer. Equivalent to the full
        # scan, without its O(context length) cost on every step.
        if engine.last_value_is_goal(state, target_outputs):
            return 1.0
        if graded:
            best = max(best, engine.last_value_progress(state, target_outputs))
    return best


def run_search(train_inputs: tuple, target_outputs: tuple, config: ClassicalMCTSConfig, budget: int, rollout_depth: int, rng: random.Random,
               stop_on_solve: bool = False) -> MCTSGraph:
    """`stop_on_solve` returns as soon as an expansion first reaches a goal
    instead of running the remaining budget out. Off by default, and it
    cannot change any solve rate, since budget is spent only on expansions.

    It exists because the node-count and wall-clock columns are otherwise
    not comparable across arms. Once a terminal node exists, `select()`
    keeps returning it (its Q is maximal) and the loop re-backs-up 1.0
    without consuming expansion budget, until the 20x-budget iteration cap.
    So a solved run freezes its node count at the solve point and then
    burns cheap iterations -- meaning an arm that solves MORE often reports
    fewer nodes and different wall-clock for reasons that have nothing to
    do with its efficiency."""
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
            if _mark_exhausted(graph, path[-1], leaf, config):
                break
            continue
        expanded = expand(graph, path[-1], target_outputs, config, rng)
        if expanded is None:
            # every untried move failed at real execution (see safe_apply_move),
            # or was rejected as a duplicate -- equivalent to the leaf having
            # had no untried moves at all.
            value = simulate(leaf.program_state, target_outputs, rollout_depth, config, rng)
            backup(graph, path, path_moves, value)
            if _mark_exhausted(graph, path[-1], leaf, config):
                break
            continue
        child_key, move = expanded
        expansions_used += 1
        path.append(child_key)
        path_moves.append(move)
        child = graph.nodes[child_key]
        value = 1.0 if child.is_terminal else simulate(child.program_state, target_outputs, rollout_depth, config, rng)
        backup(graph, path, path_moves, value)
        if stop_on_solve and child.is_terminal:
            break
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
