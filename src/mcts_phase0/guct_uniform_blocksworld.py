"""GreedyUCT-Uniform (arXiv:2405.18248) on Blocksworld -- a faithful
reimplementation of the paper's own proposed algorithm, not our usual blind
UCB1-MCTS. Three real differences from every other classical module here:

1. Reward is a heuristic function call (`h(s)`, via pyperplan's `hFFHeuristic`
   -- the exact heuristic the paper itself uses), never a random rollout.
2. Backup is Full Bellman (Definition 3 of the paper): each node's value is
   the *minimum* heuristic ever seen anywhere in its subtree, not a running
   average. This is a genuinely different pooling rule from every other
   module's Monte Carlo backup.
3. Selection uses the paper's own bandit, LCB1-Uniform (Theorem 6), derived
   from Peaks-Over-Threshold Extreme Value Theory rather than UCB1's generic
   Hoeffding bound -- and it has no free exploration constant to tune.

Node keying (merge-vs-tree) is unchanged from `classical_mcts_blocksworld.py`.
Cycle-safety needed one real addition beyond that module's pattern: LCB1-Uniform
has no self-limiting exploration term (see run_search's docstring), so a
cycle-blocked or fully-confirmed-terminal edge must be permanently excluded
from future selection (`MCTSEdge.closed`), not just reused as a free extra
sample the way the rollout-based modules safely do.
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
    t: int = 0  # visit count for this edge (arm)
    l_hat: float = float("inf")  # running min sample, this edge's own pulls only
    u_hat: float = float("-inf")  # running max sample, this edge's own pulls only
    closed: bool = False  # no more useful sampling through this edge; see run_search


@dataclass
class MCTSNode:
    state: frozenset
    own_h: float = 0.0  # this state's own heuristic value, fixed at creation
    h_gbfs: float = 0.0  # Full Bellman aggregate: min heuristic ever seen in this subtree
    n_visits: int = 0
    children: dict = field(default_factory=dict)  # Operator -> MCTSEdge
    untried_moves: list = field(default_factory=list)
    is_terminal: bool = False


@dataclass
class MCTSGraph:
    nodes: dict
    root_key: object
    next_id: int = 0  # baseline-mode node-id counter; see expand()


@dataclass
class GUCTUniformConfig:
    merge_enabled: bool


def _make_node(state: frozenset, goal: frozenset, task, heuristic) -> MCTSNode:
    solved = is_goal(state, goal)
    own_h = 0.0 if solved else float(heuristic(make_root_node(state)))
    return MCTSNode(
        state=state,
        own_h=own_h,
        h_gbfs=own_h,
        untried_moves=[] if solved else legal_moves(state, task),
        is_terminal=solved,
    )


def create_root(start_state: frozenset, goal: frozenset, task, merge_enabled: bool, heuristic) -> MCTSGraph:
    root = _make_node(start_state, goal, task, heuristic)
    if merge_enabled:
        return MCTSGraph(nodes={start_state: root}, root_key=start_state)
    return MCTSGraph(nodes={0: root}, root_key=0, next_id=1)


def _lcb1_uniform_score(edge: MCTSEdge, T: int) -> float:
    """Theorem 6: LCB1-Uniform_i = (u+l)/2 - (u-l)*sqrt(6*t_i*log T).
    Minimization framing (planning cost) -- lower is more promising.
    No free exploration constant, unlike UCB1's `c`."""
    if edge.t == 0:
        return float("-inf")
    mid = (edge.u_hat + edge.l_hat) / 2.0
    spread = edge.u_hat - edge.l_hat
    return mid - spread * math.sqrt(6.0 * edge.t * math.log(T))


def select(graph: MCTSGraph, config: GUCTUniformConfig) -> tuple[list[object], list]:
    """Cycle-safe: restricts the argmin to children not already on this path
    (Blocksworld stays reversible regardless of which bandit drives selection
    -- identical reasoning to classical_mcts_blocksworld.py's select()), and
    also excludes edges already marked `closed` -- either a prior cycle-block
    or a confirmed terminal whose value is already fully known. See
    run_search's docstring for why this bandit specifically needs both.

    Path membership is tracked in a set alongside the returned list: this
    algorithm's deliberate depth-first commitment (see run_search's
    docstring on LCB1-Uniform's divergence) produces much deeper paths than
    the broader blind-rollout modules ever see, so an O(depth) linear scan
    of a list here -- fine there -- turns select() cubic in budget. A set
    keeps each membership check O(1)."""
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


def expand(
    graph: MCTSGraph, leaf_key: object, goal: frozenset, task,
    config: GUCTUniformConfig, rng: random.Random, heuristic,
) -> tuple[object, object]:
    """Baseline (no-merge) keys are a plain incrementing counter, not a
    path-move tuple like every other module here: pyperplan's `Operator`
    hashes by rehashing its precondition/effect frozensets on every call
    (no memoized __hash__), so a growing path-tuple of Operators would need
    an increasingly expensive rehash on every dict lookup as depth grows --
    and this bandit's paths get much deeper than the other modules' ever do
    (same reasoning as select()'s path-set fix above). An integer counter
    gives the same guarantee (every baseline node is globally unique, so it
    never merges) at O(1) cost regardless of depth."""
    node = graph.nodes[leaf_key]
    op = rng.choice(node.untried_moves)
    node.untried_moves.remove(op)
    new_state = apply_move(node.state, op)

    if config.merge_enabled:
        child_key = new_state
    else:
        child_key = graph.next_id
        graph.next_id += 1
    if child_key not in graph.nodes:
        graph.nodes[child_key] = _make_node(new_state, goal, task, heuristic)
    node.children[op] = MCTSEdge(child_key=child_key)
    return child_key, op


def backup(graph: MCTSGraph, path: list, path_moves: list, value: float) -> None:
    """Full Bellman: each ancestor's h_gbfs is the min heuristic value ever
    backed up through it. Each traversed edge's own (t, l_hat, u_hat) is
    updated from the same sample -- these are what LCB1-Uniform reads."""
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
    start_state: frozenset, goal: frozenset, task,
    config: GUCTUniformConfig, budget: int, rng: random.Random, heuristic,
) -> MCTSGraph:
    """budget counts EXPANSIONS, matching every other classical module's
    accounting exactly. No rollout_depth -- evaluation is one heuristic call,
    never a rollout.

    On a cycle-block (select() reaches a fully-expanded leaf whose every
    child is already on the current path), this module does NOT reuse the
    leaf's own heuristic value as a free extra backup sample the way every
    rollout-based module here does on the identical cycle-block case. Under
    LCB1-Uniform that reuse is actively harmful: repeatedly feeding the same
    frozen sample through an edge keeps growing that edge's visit count t
    while its (l_hat, u_hat) spread never moves, and LCB1-Uniform's score
    diverges toward -inf as t grows for a fixed spread (the paper's own
    "focuses on one plateau" behavior, Fig. 3) -- so the blocked edge would
    look MORE attractive every time it fails, permanently outcompeting every
    untouched sibling and locking the search into one dead lane until
    max_iterations. (UCB1's exploration term shrinks with more visits
    instead, which is why the identical reuse pattern is harmless in every
    other module here.) Instead: mark the edge into the blocked leaf as
    `closed` (excluded by select() from here on) and retry with no backup --
    the next top-level select() call starts fresh from root and naturally
    tries a different branch. If root itself ends up with no untried moves
    and every child closed, the reachable graph is genuinely exhausted;
    stop early rather than spinning to max_iterations.

    The exact same divergence risk applies to a *confirmed terminal*: once a
    goal is found, its value is always exactly 0.0 forever (deterministic,
    no uncertainty left), so l_hat == u_hat == 0.0 and its score is pinned
    at exactly 0 -- not diverging, but also never getting any worse, so
    nothing else ever naturally outcompetes it either. Left unclosed, the
    search would keep re-walking into an already-fully-known terminal on
    essentially every remaining iteration instead of exploring anything new
    (this was the actual dominant cost in early testing, not the cycle-block
    case above -- ~99% of iterations at a modest budget were wasted
    re-confirming an already-found goal). A terminal's value needs exactly
    one visit; close its edge immediately after that first backup too."""
    graph = create_root(start_state, goal, task, config.merge_enabled, heuristic)
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
        child_key, op = expand(graph, path[-1], goal, task, config, rng, heuristic)
        expansions_used += 1
        path.append(child_key)
        path_moves.append(op)
        child = graph.nodes[child_key]
        backup(graph, path, path_moves, child.own_h)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal for n in graph.nodes.values())


def _random_rollout(state: frozenset, goal: frozenset, task, rollout_depth: int, rng: random.Random) -> bool:
    """This module has no `simulate()` of its own (GUCT-Uniform's own
    evaluation is a single heuristic call, never a rollout) -- this is a
    plain random walk built fresh, matching classical_mcts_blocksworld.py's
    own simulate() shape, purely for the random-search ablation below."""
    if is_goal(state, goal):
        return True
    for _ in range(rollout_depth):
        moves = legal_moves(state, task)
        state = apply_move(state, rng.choice(moves))
        if is_goal(state, goal):
            return True
    return False


def run_random_search(start_state: frozenset, goal: frozenset, task, budget: int, rollout_depth: int, rng: random.Random) -> bool:
    """The "no MCTS at all" ablation (see RANDOM_VS_MCTS_FINDINGS.md):
    `budget` independent random rollouts from the start state, no tree, no
    LCB1-Uniform, nothing carried between attempts."""
    for _ in range(budget):
        if _random_rollout(start_state, goal, task, rollout_depth, rng):
            return True
    return False
