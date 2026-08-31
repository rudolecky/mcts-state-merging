"""Classical MCTS on Connect Four with K-independent-candidate-sampling
expansion -- no LLM, no learned metric. Directly tests the claim
`CONNECT_FOUR_GUIDANCE_ABLATION_FINDINGS.md` inferred: plain classical
MCTS's null was guidance-quality-independent because its
single-simulation, `untried_moves`-tracked expansion gives baseline a
*structural* guarantee it never creates duplicate nodes -- worse guidance
can only reorder visits, not manufacture redundancy that was never
possible. This module removes that guarantee on purpose, mirroring the
LLM harness's actual expansion mechanism: every expansion event draws K
independent random legal moves *with replacement* (the same move can be
drawn twice), and baseline never deduplicates even within one K-batch --
exactly like K independent LLM samples that happen to produce identical
continuations. If the guidance-ablation module's diagnosis is right, this
variant should show a guidance-driven merge benefit that plain Connect
Four never did.

`simulate()` is imported directly from `classical_mcts.py`, unchanged --
it only estimates a leaf's value and has no dependency on how the tree
above it was built.

Node keying: treatment merges by `canonical_state` as everywhere else in
this project. Baseline's key is a fresh, globally-unique id per draw
(not move- or path-based) -- the one genuine departure from every other
classical module's convention, and the load-bearing part of this design:
it's what makes two draws of the identical move produce two *distinct*
baseline nodes instead of silently collapsing into one, exactly matching
the LLM baseline's "never deduplicates, not even exact repeats" behavior.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from itertools import count

from .classical_mcts import simulate
from .datasets.connect_four_engine import apply_move, canonical_state, check_win, is_full, legal_moves, opponent, replay


@dataclass
class MCTSEdge:
    child_key: tuple
    n_edge: int = 0


@dataclass
class MCTSNode:
    board: tuple
    to_move: str
    n_visits: int = 0
    w_value: float = 0.0
    children: dict[int, MCTSEdge] = field(default_factory=dict)
    expanded: bool = False  # replaces untried_moves: a node is expanded at most once, in one K-batch
    is_terminal: bool = False
    terminal_value: float | None = None


@dataclass
class MCTSGraph:
    nodes: dict[tuple, MCTSNode]
    root_key: tuple


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    K: int = 4
    c: float = 1.4  # ~sqrt(2), the standard UCB1 exploration constant
    guidance_depth_cap: int | None = None  # None = honest full rollout, matching classical_mcts.py


def _make_node(board: tuple, to_move: str, hero: str, height: int, just_won: bool) -> MCTSNode:
    is_terminal = just_won or is_full(board, height)
    terminal_value = None
    if is_terminal:
        mover = opponent(to_move)
        terminal_value = 1.0 if (just_won and mover == hero) else 0.0
    return MCTSNode(board=board, to_move=to_move, is_terminal=is_terminal, terminal_value=terminal_value)


def create_root(pre_moves: tuple[int, ...], to_move: str, width: int, height: int, merge_enabled: bool) -> MCTSGraph:
    board = replay(pre_moves, width, height)
    root_key = canonical_state(board, to_move) if merge_enabled else ()
    root = MCTSNode(board=board, to_move=to_move)
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(parent.n_visits) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list[tuple], list[int]]:
    """Returns (path, edge_draws): edge_draws[i] is the draw index used to
    step from path[i] to path[i+1]. Threading the exact draw index
    explicitly (rather than re-deriving it from child_key in backup())
    matters here specifically because two *different* draws from the same
    parent can point at the same merged child -- child_key alone can't
    tell those edges apart."""
    path = [graph.root_key]
    edge_draws = []
    node = graph.nodes[graph.root_key]
    while node.expanded and node.children and not node.is_terminal:
        best_draw = max(
            node.children,
            key=lambda d: _ucb1_score(node, node.children[d], graph.nodes[node.children[d].child_key], config.c),
        )
        edge = node.children[best_draw]
        path.append(edge.child_key)
        edge_draws.append(best_draw)
        node = graph.nodes[edge.child_key]
    return path, edge_draws


_baseline_serial = count()


def expand_batch(
    graph: MCTSGraph, leaf_key: tuple, hero: str, height: int, config: ClassicalMCTSConfig, rng: random.Random,
) -> list[tuple]:
    """Draws K independent legal moves WITH replacement and creates (or
    merges, in treatment) a child per draw. Marks the leaf expanded --
    it will never be expanded again."""
    node = graph.nodes[leaf_key]
    moves = legal_moves(node.board, height)
    child_keys = []
    for draw_index in range(config.K):
        col = rng.choice(moves)
        new_board = apply_move(node.board, col, node.to_move)
        row = len(new_board[col]) - 1
        won = check_win(new_board, col, row, node.to_move)
        new_to_move = opponent(node.to_move)

        if config.merge_enabled:
            child_key = canonical_state(new_board, new_to_move)
        else:
            child_key = ("baseline", next(_baseline_serial))  # fresh id per draw -- never merges, even exact repeats

        if child_key not in graph.nodes:
            graph.nodes[child_key] = _make_node(new_board, new_to_move, hero, height, won)
        node.children[draw_index] = MCTSEdge(child_key=child_key)
        child_keys.append(child_key)
    node.expanded = True
    return child_keys


def backup(graph: MCTSGraph, path: list[tuple], edge_draws: list[int], value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            parent.children[edge_draws[i - 1]].n_edge += 1


def run_search(
    pre_moves: tuple[int, ...], to_move: str, width: int, height: int,
    config: ClassicalMCTSConfig, budget: int, rng: random.Random,
) -> MCTSGraph:
    """budget counts EXPANSION EVENTS (K-draw batches), matching the LLM
    harness's own convention (budget=10, K=4 -> up to 40 nodes possible)."""
    graph = create_root(pre_moves, to_move, width, height, config.merge_enabled)
    hero = to_move
    expansions_used = 0
    iterations = 0
    max_iterations = 20 * budget
    while expansions_used < budget and iterations < max_iterations:
        iterations += 1
        path, edge_draws = select(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.is_terminal:
            backup(graph, path, edge_draws, leaf.terminal_value)
            continue
        child_keys = expand_batch(graph, path[-1], hero, height, config, rng)
        expansions_used += 1
        for draw_index, child_key in enumerate(child_keys):
            child = graph.nodes[child_key]
            value = child.terminal_value if child.is_terminal else simulate(
                child.board, child.to_move, hero, width, height, rng, config.guidance_depth_cap
            )
            backup(graph, path + [child_key], edge_draws + [draw_index], value)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal and n.terminal_value == 1.0 for n in graph.nodes.values())
