"""Classical Monte Carlo Tree Search on Three Men's Morris (movement phase)
-- no LLM, no learned metric. Third no-LLM merge-vs-tree domain, filling
the one cell Connect Four (irreversible, adversarial) and the 8-puzzle
(reversible, single-agent) leave open: adversarial AND reversible.

Combines classical_mcts.py's hero/opponent node shape and fixed-perspective
terminal-value convention with classical_mcts_puzzle.py's cycle-safe
select()/run_search -- Morris's movement phase is exactly as reversible as
the 8-puzzle (sliding a piece out and back returns the prior board), so the
same real bug (select() deterministically re-preferring a cycling child,
run_search's node count plateauing regardless of budget) would recur
without it. Neither existing module has both properties, so this is a
third, deliberate duplication of the small UCB1/select/backup core -- see
cheerful-jumping-moler.md's design section for why extraction still isn't
worth it at this call site count (both prior modules are closed,
already-reported results).

Selection uses classical UCB1, not PUCT, for the same reason as the other
two modules: no policy network exists here to supply a prior.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .datasets.morris_engine import apply_move, check_win, legal_moves, opponent


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
    children: dict[tuple, MCTSEdge] = field(default_factory=dict)
    untried_moves: list[tuple] = field(default_factory=list)
    is_terminal: bool = False
    terminal_value: float | None = None  # 1.0 hero wins, 0.0 opponent wins or hero boxed in


@dataclass
class MCTSGraph:
    nodes: dict[tuple, MCTSNode]
    root_key: tuple


@dataclass
class ClassicalMCTSConfig:
    merge_enabled: bool
    c: float = 1.4  # ~sqrt(2), the standard UCB1 exploration constant


def _make_node(board: tuple, to_move: str, hero: str, just_won_by: str | None) -> MCTSNode:
    """`to_move` is whoever moves next; `just_won_by` is the side whose move
    just landed (opponent(to_move)), if this node was reached via a win."""
    moves = legal_moves(board, to_move)
    is_terminal = just_won_by is not None or not moves
    terminal_value = None
    if is_terminal:
        if just_won_by is not None:
            terminal_value = 1.0 if just_won_by == hero else 0.0
        else:
            # to_move has no legal moves -- boxed in, to_move loses
            terminal_value = 0.0 if to_move == hero else 1.0
    return MCTSNode(
        board=board, to_move=to_move,
        untried_moves=[] if is_terminal else moves,
        is_terminal=is_terminal, terminal_value=terminal_value,
    )


def create_root(board: tuple, to_move: str, merge_enabled: bool) -> MCTSGraph:
    root_key = (board, to_move) if merge_enabled else ()
    root = MCTSNode(board=board, to_move=to_move, untried_moves=legal_moves(board, to_move))
    return MCTSGraph(nodes={root_key: root}, root_key=root_key)


def _ucb1_score(parent: MCTSNode, edge: MCTSEdge, child: MCTSNode, c: float) -> float:
    if edge.n_edge == 0:
        return float("inf")
    q = child.w_value / child.n_visits
    return q + c * math.sqrt(math.log(parent.n_visits) / edge.n_edge)


def select(graph: MCTSGraph, config: ClassicalMCTSConfig) -> tuple[list[tuple], tuple]:
    """Descend via UCB1, restricted to children not already on this path --
    Morris's movement phase is reversible, so an unrestricted argmax could
    deterministically prefer a cycling child forever. Gives up on a node
    only once every child would cycle back (rare: legal_moves is never
    permanently exhausted except at a true terminal)."""
    path = [graph.root_key]
    path_moves: tuple = ()
    node = graph.nodes[graph.root_key]
    while not node.is_terminal and not node.untried_moves and node.children:
        candidates = {m: e for m, e in node.children.items() if e.child_key not in path}
        if not candidates:
            break
        best_move = max(
            candidates,
            key=lambda m: _ucb1_score(node, candidates[m], graph.nodes[candidates[m].child_key], config.c),
        )
        edge = candidates[best_move]
        path.append(edge.child_key)
        path_moves = path_moves + (best_move,)
        node = graph.nodes[edge.child_key]
    return path, path_moves


def expand(
    graph: MCTSGraph, leaf_key: tuple, path_moves: tuple,
    hero: str, config: ClassicalMCTSConfig, rng: random.Random,
) -> tuple:
    node = graph.nodes[leaf_key]
    move = rng.choice(node.untried_moves)
    node.untried_moves.remove(move)
    from_cell, to_cell = move
    new_board = apply_move(node.board, from_cell, to_cell, node.to_move)
    won = check_win(new_board, node.to_move)
    new_to_move = opponent(node.to_move)

    child_key = (new_board, new_to_move) if config.merge_enabled else path_moves + (move,)
    if child_key not in graph.nodes:
        graph.nodes[child_key] = _make_node(new_board, new_to_move, hero, node.to_move if won else None)
    node.children[move] = MCTSEdge(child_key=child_key)
    return child_key


def simulate(board: tuple, to_move: str, hero: str, rollout_depth: int, rng: random.Random) -> float:
    """Uniformly random legal moves for both sides up to `rollout_depth`
    steps. 1.0 iff hero wins during the walk, else 0.0 (opponent wins, a
    side gets boxed in, or the depth cap is reached without a winner) --
    same fixed-perspective binary convention as classical_mcts.py."""
    for _ in range(rollout_depth):
        moves = legal_moves(board, to_move)
        if not moves:
            return 0.0 if to_move == hero else 1.0  # to_move boxed in -> to_move loses
        from_cell, to_cell = rng.choice(moves)
        board = apply_move(board, from_cell, to_cell, to_move)
        if check_win(board, to_move):
            return 1.0 if to_move == hero else 0.0
        to_move = opponent(to_move)
    return 0.0


def backup(graph: MCTSGraph, path: list[tuple], value: float) -> None:
    for i, key in enumerate(path):
        node = graph.nodes[key]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            move = next(m for m, e in parent.children.items() if e.child_key == key)
            parent.children[move].n_edge += 1


def run_search(
    board: tuple, to_move: str, config: ClassicalMCTSConfig, budget: int, rollout_depth: int, rng: random.Random,
) -> MCTSGraph:
    """budget counts EXPANSIONS, matching the other two modules' accounting exactly."""
    graph = create_root(board, to_move, config.merge_enabled)
    hero = to_move
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
            # select() gave up on a cycle: leaf is fully expanded with
            # nowhere new to go. Refine its own estimate with one more
            # rollout instead of expanding -- free, like the terminal case.
            value = simulate(leaf.board, leaf.to_move, hero, rollout_depth, rng)
            backup(graph, path, value)
            continue
        child_key = expand(graph, path[-1], path_moves, hero, config, rng)
        expansions_used += 1
        path.append(child_key)
        child = graph.nodes[child_key]
        value = child.terminal_value if child.is_terminal else simulate(child.board, child.to_move, hero, rollout_depth, rng)
        backup(graph, path, value)
    return graph


def is_solved(graph: MCTSGraph) -> bool:
    return any(n.is_terminal and n.terminal_value == 1.0 for n in graph.nodes.values())


def run_random_search(board: tuple, to_move: str, hero: str, budget: int, rollout_depth: int, rng: random.Random) -> bool:
    """The "no MCTS at all" ablation (see RANDOM_VS_MCTS_FINDINGS.md):
    `budget` independent random rollouts (both sides moving randomly) from
    the start position, no tree, no UCB1, reusing this module's own
    `simulate()`."""
    for _ in range(budget):
        if simulate(board, to_move, hero, rollout_depth, rng) == 1.0:
            return True
    return False
