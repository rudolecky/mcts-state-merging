"""GreedyUCT-Uniform (arXiv:2405.18248) on the 2x2x2 Rubik's Cube -- ports
`guct_uniform_blocksworld.py`'s exact algorithm to the cube domain, not a
new design. Two real differences from `classical_mcts_rubiks.py`:

1. Reward is the exact BFS-oracle distance (`distances[state]`), never a
   random rollout or a hand-derived heuristic -- the paper's own
   Full-Bellman backup pairs naturally with any deterministic per-state
   value, and the cube's exact oracle table already exists on disk.
2. Backup is Full Bellman (Definition 3 of the paper): each node's value
   is the *minimum* distance-to-goal ever seen anywhere in its subtree,
   not a running average -- see RUBIKS_CUBE_FINDINGS.md's "Attempted fix"
   section for why plain MC-average backup (classical_mcts_rubiks.py's
   own convention) turned out to be the actual problem with merging on
   this domain, not how often merging fires.
3. Selection uses the paper's own LCB1-Uniform bandit (Theorem 6), with
   no free exploration constant, framed as cost-minimization (lower
   score = more promising) -- the reverse convention from
   classical_mcts_rubiks.py's `1/(1+h)` maximization. This module stays
   in the paper's own raw-cost convention throughout, not a hybrid.

Node keying (merge-vs-tree) matches classical_mcts_rubiks.py's own
convention (path-move-index tuples for baseline, not
guct_uniform_blocksworld.py's incrementing-integer workaround -- that
workaround exists there specifically because pyperplan's `Operator`
re-hashes expensively on every lookup; cube states are plain tuples with
no such cost, confirmed by every other cube experiment run so far).

Cycle-safety needs the same real addition as guct_uniform_blocksworld.py:
LCB1-Uniform has no self-limiting exploration term, so a cycle-blocked or
confirmed-terminal edge must be permanently excluded from future
selection (`MCTSEdge.closed`), not reused as a free extra sample the way
classical_mcts_rubiks.py's rollout-based modules safely do. Cube moves
are all invertible, exactly like Blocksworld's, so the same reasoning
applies unchanged.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets import rubiks_cube_engine as rc


@dataclass
class MCTSEdge:
    child_key: object
    t: int = 0  # visit count for this edge (arm)
    l_hat: float = float("inf")  # running min sample, this edge's own pulls only
    u_hat: float = float("-inf")  # running max sample, this edge's own pulls only
    closed: bool = False  # no more useful sampling through this edge; see run_search


@dataclass
class MCTSNode:
    state: rc.State
    own_h: float = 0.0  # this state's own oracle distance, fixed at creation
    h_gbfs: float = 0.0  # Full Bellman aggregate: min distance ever seen in this subtree
    n_visits: int = 0
    children: dict[int, MCTSEdge] = field(default_factory=dict)
    untried_moves: list[int] = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class MCTSGraph:
    nodes: dict[object, MCTSNode]
    root_key: object


@dataclass
class GUCTUniformConfig:
    merge_enabled: bool


def _make_node(state: rc.State, distances: dict) -> MCTSNode:
    solved = rc.is_goal(state)
    own_h = 0.0 if solved else float(distances[state])
    return MCTSNode(
        state=state,
        own_h=own_h,
        h_gbfs=own_h,
        untried_moves=[] if solved else list(range(len(rc.ALL_MOVES))),
        is_terminal=solved,
    )


def create_root(start_state: rc.State, merge_enabled: bool, distances: dict) -> MCTSGraph:
    root_key = start_state if merge_enabled else ()
    root = _make_node(start_state, distances)
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _lcb1_uniform_score(edge: MCTSEdge, T: int) -> float:
    """Theorem 6: LCB1-Uniform_i = (u+l)/2 - (u-l)*sqrt(6*t_i*log T).
    Minimization framing (planning cost) -- lower is more promising.
    No free exploration constant, unlike UCB1's `c`."""
    if edge.t == 0:
        return float("-inf")
    mid = (edge.u_hat + edge.l_hat) / 2.0
    spread = edge.u_hat - edge.l_hat
    return mid - spread * math.sqrt(6.0 * edge.t * math.log(T))


def select(graph: MCTSGraph, config: GUCTUniformConfig) -> tuple[list[object], list[int]]:
    """Cycle-safe: restricts the argmin to children not already on this path
    and excludes edges already marked `closed` -- see run_search's
    docstring. Path membership tracked as a set: this bandit's deliberate
    depth-first commitment produces much deeper paths than
    classical_mcts_rubiks.py's blind UCB1 ever sees, so an O(depth) list
    scan would turn select() cubic in budget."""
    path = [graph.root_key]
    path_set = {graph.root_key}
    path_moves: list[int] = []
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
    graph: MCTSGraph, leaf_key: object, path_moves: list[int],
    config: GUCTUniformConfig, rng: random.Random, distances: dict,
) -> tuple[object, int]:
    node = graph.nodes[leaf_key]
    move = rng.choice(node.untried_moves)
    node.untried_moves.remove(move)
    new_state = rc.apply_move(node.state, rc.ALL_MOVES[move])

    child_key = new_state if config.merge_enabled else tuple(path_moves) + (move,)
    if child_key not in graph.nodes:
        graph.nodes[child_key] = _make_node(new_state, distances)
    node.children[move] = MCTSEdge(child_key=child_key)
    return child_key, move


def backup(graph: MCTSGraph, path: list[object], path_moves: list[int], value: float) -> None:
    """Full Bellman: each ancestor's h_gbfs is the min distance ever backed
    up through it. Each traversed edge's own (t, l_hat, u_hat) is updated
    from the same sample -- these are what LCB1-Uniform reads."""
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
    start_state: rc.State, config: GUCTUniformConfig, budget: int, rng: random.Random, distances: dict,
) -> MCTSGraph:
    """budget counts EXPANSIONS, matching every other classical module's
    accounting exactly. No rollout_depth -- evaluation is one oracle
    lookup, never a rollout.

    Same divergence-prevention logic as guct_uniform_blocksworld.py's own
    run_search (see that module's docstring for the full derivation):
    LCB1-Uniform's score diverges toward -inf as an edge's visit count t
    grows with its (l_hat, u_hat) spread pinned, so re-feeding the same
    frozen sample through a cycle-blocked or confirmed-terminal edge would
    make it look permanently more attractive, locking the search into one
    dead lane. Both cases close their edge after exactly one backup
    instead."""
    graph = create_root(start_state, config.merge_enabled, distances)
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
                break  # root itself is fully exhausted -- nothing left to explore
            parent = graph.nodes[path[-2]]
            parent.children[path_moves[-1]].closed = True
            continue
        child_key, move = expand(graph, path[-1], path_moves, config, rng, distances)
        expansions_used += 1
        path.append(child_key)
        path_moves.append(move)
        child = graph.nodes[child_key]
        backup(graph, path, path_moves, child.own_h)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal for n in graph.nodes.values())
