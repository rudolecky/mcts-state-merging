"""Classical Monte Carlo Tree Search on Blocksworld -- no LLM, no learned
metric. Single-agent (like the 8-puzzle) -- no hero/opponent split. Fully
reversible (pickup/putdown and stack/unstack are literal inverse pairs),
so this needs the same cycle-safe select()/run_search fix as
classical_mcts_puzzle.py (8-puzzle) and classical_mcts_morris.py (Morris)
from the very start -- the exact bug class hit twice already this session,
built in here rather than rediscovered a third time.

Node keying: a state's own frozenset is already canonical (no packaging
needed -- simpler than every prior domain, none of which had a bare
hashable state object to key on directly).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from pyperplan.search.searchspace import make_root_node

from .datasets.blocksworld_engine import apply_move, is_goal, legal_moves


@dataclass
class MCTSEdge:
    child_key: object
    n_edge: int = 0


@dataclass
class MCTSNode:
    state: frozenset
    n_visits: int = 0
    w_value: float = 0.0
    children: dict = field(default_factory=dict)  # Operator -> MCTSEdge
    untried_moves: list = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float | None = None  # 1.0 iff solved
    exhausted: bool = False
    """Reject-duplicates arm only: this node has no untried moves left and
    `select()` could not descend past it, so nothing new can ever come from
    it. Without marking it, it stays the most attractive frontier node its
    parent has and gets re-selected for rollout-only iterations forever --
    burning the iteration allowance (20x budget) without spending a single
    expansion, so the arm silently runs at a fraction of the budget the
    other arms get. Marked nodes are excluded from `select()`."""


@dataclass
class MCTSGraph:
    nodes: dict
    root_key: object
    seen_states: set = field(default_factory=set)
    """Only populated in the reject-duplicates arm, where node keys are
    path-based, so `nodes` cannot answer "has this state been seen"."""
    new_nodes: int = 0
    """Nodes created by an expansion. Excludes the root, matching
    `classical_mcts_rubiks.py`'s pre-existing `stats["new_nodes"]`
    convention, so the counter means the same thing in every module."""
    merge_hits: int = 0
    rejected: int = 0


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    c: float = 1.4  # ~sqrt(2), the standard UCB1 exploration constant
    value_source: str = "rollout"  # "rollout" | "heuristic" -- see _heuristic_value; requires run_search's heuristic= arg
    reject_duplicates: bool = False
    """Third arm, only meaningful with `merge_enabled=False`: keep
    path-based node keys (so no statistics are ever pooled and the backup
    rule is untouched) but DISCARD an expansion that lands on a state seen
    anywhere in this search, trying another untried move instead. Splits
    what ordinary merging does into its two halves -- "expansions land on
    distinct states" and "statistics from different paths get pooled" --
    so they can be attributed separately. Blocksworld is the domain where
    merging is this project's largest win, so it is the sharpest test of
    which half that win comes from.

    Budget accounting: a rejected move never becomes an expansion, so it
    does not count against the expansion budget -- consistent with this
    project's convention that budget counts expansions. It does cost a real
    `apply_move`, though, so this arm does strictly more work per unit of
    budget than the other two; that difference shows up only in wall-clock,
    which is why wall-clock is reported next to every arm's node count."""



    charge_rejections: bool = False
    """Whether a rejected expansion counts against the budget.

    It should, for any claim of the form "the filter beats merging at
    matched budget": a rejected move still costs a real `apply_move`, so
    with this False the filter performs strictly more state-generation
    work per unit of budget than the arms it is compared against. Under
    GUCT-Uniform, charging rejections erased the filter's entire apparent
    advantage on Blocksworld and the cube -- see
    results/GUCT_UNIFORM_REJECT_CROSSING_FINDINGS.md. Default False only
    so previously-recorded runs stay reproducible."""

    def __post_init__(self):
        if self.merge_enabled and self.reject_duplicates:
            # These two do not compose, and the way they fail to compose
            # differs per module (one silently disables merging, another
            # silently ignores the rejection), so the combination is
            # refused rather than left to mean whatever the code happens
            # to do. `reject_duplicates` is an alternative to merging, not
            # a modifier of it.
            raise ValueError("reject_duplicates is only meaningful with merge_enabled=False")

def _make_node(state: frozenset, goal: frozenset, task) -> MCTSNode:
    solved = is_goal(state, goal)
    return MCTSNode(
        state=state,
        untried_moves=[] if solved else legal_moves(state, task),
        is_terminal=solved,
        terminal_value=1.0 if solved else None,
    )


def create_root(start_state: frozenset, goal: frozenset, task, merge_enabled: bool, reject_duplicates: bool = False) -> MCTSGraph:
    root_key = start_state if merge_enabled else ()
    root = MCTSNode(state=start_state, untried_moves=legal_moves(start_state, task))
    seen = {start_state} if reject_duplicates else set()
    return MCTSGraph(nodes={root_key: root}, root_key=root_key, seen_states=seen)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(parent.n_visits) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list[object], list]:
    """Cycle-safe: restricts UCB1's argmax to children not already on this
    path (Blocksworld moves are reversible, so an unrestricted argmax could
    deterministically prefer a cycling child forever -- see
    classical_mcts_puzzle.py's docstring for the full reasoning, identical
    here). Returns (path, path_moves) -- path_moves is the sequence of
    operators taken, needed for baseline's path-based keying."""
    path = [graph.root_key]
    path_moves: list = []
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        candidates = {m: e for m, e in node.children.items()
                      if e.child_key not in path and not graph.nodes[e.child_key].exhausted}
        if not candidates:
            break
        best_move = max(
            candidates,
            key=lambda m: _ucb1_score(node, candidates[m], graph.nodes[candidates[m].child_key], config.c),
        )
        edge = candidates[best_move]
        path.append(edge.child_key)
        path_moves.append(best_move)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def expand(
    graph: MCTSGraph, leaf_key: object, path_moves: list, goal: frozenset, task,
    config: ClassicalMCTSConfig, rng: random.Random,
) -> tuple[object, object] | None:
    """Returns (child_key, op) -- the caller needs `op` too, to record the
    correct edge for the final backup step. Returns None only in the
    reject-duplicates arm, when every remaining untried move led to an
    already-seen state; the caller then treats this leaf like a
    fully-expanded one (one more rollout instead of an expansion), exactly
    as it already does when `select()` gives up on a cycle."""
    node = graph.nodes[leaf_key]
    while node.untried_moves:
        op = rng.choice(node.untried_moves)
        node.untried_moves.remove(op)
        new_state = apply_move(node.state, op)

        if config.reject_duplicates:
            if new_state in graph.seen_states:
                graph.rejected += 1
                continue  # admission filter: this state exists already, try another move
            graph.seen_states.add(new_state)

        child_key = new_state if config.merge_enabled else tuple(path_moves) + (op,)
        if child_key not in graph.nodes:
            graph.nodes[child_key] = _make_node(new_state, goal, task)
            graph.new_nodes += 1
        else:
            graph.merge_hits += 1
        node.children[op] = MCTSEdge(child_key=child_key)
        return child_key, op
    return None


def simulate(state: frozenset, goal: frozenset, task, rollout_depth: int, rng: random.Random) -> float:
    """Uniformly random applicable operators for up to `rollout_depth`
    steps. 1.0 iff the goal is reached during the walk, else 0.0."""
    if is_goal(state, goal):
        return 1.0
    for _ in range(rollout_depth):
        moves = legal_moves(state, task)
        state = apply_move(state, rng.choice(moves))
        if is_goal(state, goal):
            return 1.0
    return 0.0


def _heuristic_value(state: frozenset, heuristic) -> float:
    """heuristic is a pyperplan heuristic instance (e.g. hFFHeuristic(task)),
    built once by the caller and reused across calls -- see
    guct_uniform_blocksworld.py's own `heuristic(make_root_node(state))`
    usage, which this mirrors exactly."""
    h = float(heuristic(make_root_node(state)))
    return 1.0 / (1.0 + h)


def _leaf_value(
    state: frozenset, goal: frozenset, task, rollout_depth: int,
    config: ClassicalMCTSConfig, rng: random.Random, heuristic,
) -> float:
    if config.value_source == "heuristic":
        return _heuristic_value(state, heuristic)
    return simulate(state, goal, task, rollout_depth, rng)


def _mark_exhausted(graph: MCTSGraph, leaf_key, leaf: MCTSNode, config: ClassicalMCTSConfig) -> bool:
    """Reject-duplicates arm only. Marks a dead frontier node so `select()`
    stops returning it, and reports whether the search is finished -- which
    is the case exactly when the dead node is the root, since `select()` can
    only return an exhausted node with no non-exhausted alternative above
    it."""
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


def run_search(
    start_state: frozenset, goal: frozenset, task,
    config: ClassicalMCTSConfig, budget: int, rollout_depth: int, rng: random.Random,
    heuristic=None, stop_on_solve: bool = False,
) -> MCTSGraph:
    """budget counts EXPANSIONS, matching every other classical module's
    accounting exactly. heuristic is required when config.value_source ==
    "heuristic" (e.g. hFFHeuristic(task), built once by the caller).

    `stop_on_solve` returns the graph as soon as an expansion first reaches
    the goal, instead of running the remaining budget out. Off by default
    (every published experiment measured solve rate, which this cannot
    change, since budget is spent only on expansions). It exists for
    graph-structure analysis: once a terminal node exists, the loop keeps
    re-selecting it and backing up 1.0 along the solution path without
    consuming any expansion budget, so visit counts on that path inflate by
    orders of magnitude and any post-hoc statistic involving n_visits
    becomes an artifact of how long the loop ran afterwards."""
    graph = create_root(start_state, goal, task, config.merge_enabled, config.reject_duplicates)
    expansions_used = 0
    iterations = 0
    max_iterations = 20 * budget
    while expansions_used < budget and iterations < max_iterations:
        iterations += 1
        path, path_moves = select(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.is_terminal:
            backup(graph, path, path_moves, leaf.terminal_value)
            continue
        if not leaf.untried_moves:
            # select() gave up on a cycle: leaf is fully expanded with
            # nowhere new to go. Refine its own estimate with one more
            # rollout instead of expanding -- free, like the terminal case.
            value = _leaf_value(leaf.state, goal, task, rollout_depth, config, rng, heuristic)
            backup(graph, path, path_moves, value)
            if _mark_exhausted(graph, path[-1], leaf, config):
                break
            continue
        rejected_before = graph.rejected
        expanded = expand(graph, path[-1], path_moves, goal, task, config, rng)
        if config.charge_rejections:
            expansions_used += graph.rejected - rejected_before
        if expanded is None:
            # reject-duplicates arm: every remaining untried move led
            # somewhere already seen. Same fallback as the cycle-block case.
            value = _leaf_value(leaf.state, goal, task, rollout_depth, config, rng, heuristic)
            backup(graph, path, path_moves, value)
            if _mark_exhausted(graph, path[-1], leaf, config):
                break
            continue
        child_key, op = expanded
        expansions_used += 1
        path.append(child_key)
        path_moves.append(op)
        child = graph.nodes[child_key]
        value = child.terminal_value if child.is_terminal else _leaf_value(child.state, goal, task, rollout_depth, config, rng, heuristic)
        backup(graph, path, path_moves, value)
        if stop_on_solve and child.is_terminal and child.terminal_value == 1.0:
            break
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal and n.terminal_value == 1.0 for n in graph.nodes.values())


def run_random_search(start_state, goal, task, budget: int, rollout_depth: int, rng: random.Random) -> bool:
    """The "no MCTS at all" ablation (see ARC's own `run_random_search` in
    classical_mcts_arc_program.py): `budget` independent random rollouts
    from the start state, no tree, no UCB1, nothing carried between
    attempts -- isolates what the tree/bandit structure itself contributes,
    separate from merge vs. tree, by reusing the exact same `simulate()`
    this module's own MCTS calls for its rollouts."""
    for _ in range(budget):
        if simulate(start_state, goal, task, rollout_depth, rng) == 1.0:
            return True
    return False
