"""GreedyUCT-Uniform (arXiv:2405.18248) on Sokoban -- ports
`guct_uniform_blocksworld.py`'s exact algorithm to this domain, same
generalization test as `guct_uniform_puzzle.py`: does Full-Bellman-min
backup change merge's already-*positive* result here
(`REAL_HEURISTIC_MERGE_FINDINGS.md`, UCB1+MC-average+box-goal heuristic),
or did the backup rule only matter on the Rubik's Cube because its
transposition density was extreme enough to flip merging catastrophic?

Reward is the exact BFS-oracle distance (`sokoban_engine.bfs_distances`),
not the box-goal-plus-player heuristic -- same reasoning as
guct_uniform_puzzle.py: cleanest possible backup-rule-only comparison.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets.sokoban_engine import apply_move, is_solved as _boxes_solved, legal_moves


@dataclass
class MCTSEdge:
    child_key: object
    t: int = 0
    l_hat: float = float("inf")
    u_hat: float = float("-inf")
    closed: bool = False


@dataclass
class MCTSNode:
    state: tuple
    own_h: float = 0.0
    h_gbfs: float = 0.0
    n_visits: int = 0
    children: dict[tuple, MCTSEdge] = field(default_factory=dict)
    untried_moves: list[tuple] = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class MCTSGraph:
    nodes: dict[object, MCTSNode]
    root_key: object
    seen_states: set = field(default_factory=set)
    """Reject-duplicates arm only, where node keys are path-based, so
    `nodes` cannot answer "has this state been seen"."""
    rejected: int = 0


@dataclass
class GUCTUniformConfig:
    merge_enabled: bool
    reject_duplicates: bool = False
    """Third arm, only meaningful with `merge_enabled=False`: compute the
    merge key but DISCARD an expansion landing on an already-seen state
    rather than pooling into the existing node.

    Here to falsify the explanation the Blocksworld crossing produced
    (results/GUCT_UNIFORM_REJECT_CROSSING_FINDINGS.md). There the filter
    still beat pooling at tight budgets even under this backup rule, and
    the account given was: the backup rule stops pooled statistics
    destroying the search, while the filter separately recovers the budget
    wasted re-deriving duplicates. That account predicts the crossing is
    **null on this domain**, because GUCT-Uniform already brought merge and
    tree to parity here -- there is no budget gap left for a filter to
    widen. If the filter wins here too, the account is wrong."""

    charge_rejections: bool = False
    """Whether a rejected expansion counts against the budget.

    It should, for any claim of the form "the filter beats pooling at
    matched budget". A rejected move still costs a real `apply_move` --
    the state is generated, hashed, compared and thrown away -- so with
    this False the filter does strictly more state-generation work per
    unit of budget than the arms it is compared against. Measured on the
    cube at td=6/budget=40: 41 nodes plus 53 rejections, i.e. 2.4x the
    generations merging performed at the same nominal budget.

    Left False by default only so previously-recorded runs remain
    reproducible. Every comparative claim should be read off a run with
    this True."""

    def __post_init__(self):
        if self.merge_enabled and self.reject_duplicates:
            raise ValueError("reject_duplicates is only meaningful with merge_enabled=False")


_UNSOLVABLE_H = 1e6
"""Sentinel oracle distance for a state absent from `distances` -- a real,
expected case, not a bug: `bfs_distances()` only covers states reachable
*to* the goal via pulls, and a push-based random walk can shove a box into
a genuine deadlock (reachable by pushing forward, never pull-reachable
back), which correctly has no finite distance at all."""


def _make_node(state: tuple, distances: dict) -> MCTSNode:
    solved = _boxes_solved(state[1])
    own_h = 0.0 if solved else float(distances.get(state, _UNSOLVABLE_H))
    return MCTSNode(
        state=state,
        own_h=own_h,
        h_gbfs=own_h,
        untried_moves=[] if solved else legal_moves(state),
        is_terminal=solved,
    )


def create_root(start_state: tuple, merge_enabled: bool, distances: dict,
                reject_duplicates: bool = False) -> MCTSGraph:
    root_key = start_state if merge_enabled else ()
    root = _make_node(start_state, distances)
    seen = {start_state} if reject_duplicates else set()
    return MCTSGraph(nodes={root_key: root}, root_key=root_key, seen_states=seen)


def _lcb1_uniform_score(edge: MCTSEdge, T: int) -> float:
    if edge.t == 0:
        return float("-inf")
    mid = (edge.u_hat + edge.l_hat) / 2.0
    spread = edge.u_hat - edge.l_hat
    return mid - spread * math.sqrt(6.0 * edge.t * math.log(T))


def select(graph: MCTSGraph, config: GUCTUniformConfig) -> tuple[list[object], list[tuple]]:
    path = [graph.root_key]
    path_set = {graph.root_key}
    path_moves: list[tuple] = []
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


def expand(
    graph: MCTSGraph, leaf_key: object, path_moves: list[tuple],
    config: GUCTUniformConfig, rng: random.Random, distances: dict,
) -> tuple[object, tuple] | None:
    node = graph.nodes[leaf_key]
    while node.untried_moves:
        move = rng.choice(node.untried_moves)
        node.untried_moves.remove(move)
        new_state = apply_move(node.state, move)

        if config.reject_duplicates:
            if new_state in graph.seen_states:
                graph.rejected += 1
                continue  # admission filter: already reached, try another move
            graph.seen_states.add(new_state)

        child_key = new_state if config.merge_enabled else tuple(path_moves) + (move,)
        if child_key not in graph.nodes:
            graph.nodes[child_key] = _make_node(new_state, distances)
        node.children[move] = MCTSEdge(child_key=child_key)
        return child_key, move
    return None  # reject arm only: every remaining move led somewhere seen


def backup(graph: MCTSGraph, path: list[object], path_moves: list[tuple], value: float) -> None:
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
    start_state: tuple, config: GUCTUniformConfig, budget: int, rng: random.Random, distances: dict,
) -> MCTSGraph:
    """budget counts EXPANSIONS, matching every other classical module's
    accounting exactly. No rollout_depth -- evaluation is one oracle
    lookup, never a rollout. Same closed-edge divergence-prevention logic
    as guct_uniform_blocksworld.py/guct_uniform_rubiks.py's own
    run_search."""
    graph = create_root(start_state, config.merge_enabled, distances, config.reject_duplicates)
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
        rejected_before = graph.rejected
        expanded = expand(graph, path[-1], path_moves, config, rng, distances)
        if config.charge_rejections:
            expansions_used += graph.rejected - rejected_before
        if expanded is None:
            # Reject arm: every remaining move from this leaf led somewhere
            # already reached. Retry with NO backup (re-feeding a frozen
            # sample makes LCB1-Uniform's score diverge -- see this module's
            # own cycle-block handling), and close the edge into the leaf
            # ONLY when it has no live child edge left. Closing it
            # unconditionally would prune the leaf's unexplored subtree:
            # "produced nothing new" is not "leads nowhere".
            leaf_node = graph.nodes[path[-1]]
            if any(not e.closed for e in leaf_node.children.values()):
                continue
            if len(path) > 1:
                graph.nodes[path[-2]].children[path_moves[-1]].closed = True
                continue
            break
        child_key, move = expanded
        expansions_used += 1
        path.append(child_key)
        path_moves.append(move)
        child = graph.nodes[child_key]
        backup(graph, path, path_moves, child.own_h)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal for n in graph.nodes.values())
