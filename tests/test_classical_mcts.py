"""Verification for classical_mcts.py, independent of any model -- pure
game-tree logic, no GPU/network dependency at all.
"""

import random

from mcts_phase0.classical_mcts import (
    ClassicalMCTSConfig,
    MCTSEdge,
    MCTSNode,
    _make_node,
    _ucb1_score,
    backup,
    create_root,
    expand,
    is_solved,
    run_random_search,
    run_search,
    select,
    simulate,
)
from mcts_phase0.datasets.connect_four_engine import X, O, apply_move, canonical_state, make_empty_board


class _ScriptedRng:
    """Returns a pre-programmed sequence of choices, ignoring true randomness
    entirely -- lets tests force an exact expansion/rollout order."""

    def __init__(self, choices):
        self._choices = list(choices)

    def choice(self, seq):
        val = self._choices.pop(0)
        assert val in seq, f"scripted choice {val} not in {seq}"
        return val


# ---------- _ucb1_score / select ----------

def test_ucb1_gives_infinite_priority_to_untried_edges():
    parent = MCTSNode(board=(), to_move=X, n_visits=10)
    child = MCTSNode(board=(), to_move=O, n_visits=3, w_value=2.0)
    untried_edge = MCTSEdge(child_key=(1,), n_edge=0)
    assert _ucb1_score(parent, untried_edge, child, c=1.4) == float("inf")


def test_select_picks_expected_ucb1_argmax():
    root = MCTSNode(board=(), to_move=X, n_visits=10)
    child_a = MCTSNode(board=(), to_move=O, n_visits=8, w_value=6.0)  # Q=0.75, well-explored
    child_b = MCTSNode(board=(), to_move=O, n_visits=2, w_value=0.2)  # Q=0.1, barely explored
    root.children = {0: MCTSEdge(child_key="a", n_edge=8), 1: MCTSEdge(child_key="b", n_edge=2)}
    graph_nodes = {(): root, "a": child_a, "b": child_b}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    config = ClassicalMCTSConfig(merge_enabled=True, c=0.1)  # small c: exploitation should dominate
    path, _ = select(_Graph(), config)
    assert path == [(), "a"]  # higher Q wins when exploration term is small


# ---------- backup pooling on a hand-built two-parent DAG ----------

def test_backup_pools_at_shared_node_keeps_edge_counts_local():
    root = MCTSNode(board=(), to_move=X)
    a = MCTSNode(board=(), to_move=O)
    b = MCTSNode(board=(), to_move=O)
    m = MCTSNode(board=(), to_move=X)  # shared merge target, two parents
    root.children = {0: MCTSEdge(child_key="a"), 1: MCTSEdge(child_key="b")}
    a.children = {2: MCTSEdge(child_key="m")}
    b.children = {3: MCTSEdge(child_key="m")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "m": m}
        root_key = ()

    graph = _Graph()
    backup(graph, [(), "a", "m"], 0.8)
    backup(graph, [(), "b", "m"], 0.2)

    assert m.n_visits == 2
    assert m.w_value == 1.0  # pooled: 0.8 + 0.2
    assert a.children[2].n_edge == 1
    assert b.children[3].n_edge == 1
    assert root.n_visits == 2


# ---------- expand: merge-vs-tree structural neutrality ----------

def test_expand_merges_transposed_paths_when_enabled_not_when_disabled():
    # Two move orders (col0,col5,col1) vs (col1,col5,col0) reach the
    # identical board+side-to-move -- the same transposition fixture
    # test_connect_four_engine.py validates at the engine level.
    width, height, hero = 6, 2, X

    def _play(config, move_sequence):
        graph = create_root((), X, width, height, config.merge_enabled)
        key = graph.root_key
        path_moves = ()
        for col in move_sequence:
            rng = _ScriptedRng([col])
            key = expand(graph, key, path_moves, hero, height, config, rng)
            path_moves = path_moves + (col,)
        return graph, key

    treatment_cfg = ClassicalMCTSConfig(merge_enabled=True)
    baseline_cfg = ClassicalMCTSConfig(merge_enabled=False)

    t_graph_a, t_key_a = _play(treatment_cfg, [0, 5, 1])
    t_graph_b, t_key_b = _play(treatment_cfg, [1, 5, 0])
    assert t_key_a == t_key_b  # merged: identical canonical state

    b_graph_a, b_key_a = _play(baseline_cfg, [0, 5, 1])
    b_graph_b, b_key_b = _play(baseline_cfg, [1, 5, 0])
    assert b_key_a != b_key_b  # never merges: distinct paths -> distinct keys
    # but the underlying boards are still identical (the game itself is the same)
    assert b_graph_a.nodes[b_key_a].board == b_graph_b.nodes[b_key_b].board


# ---------- terminal value convention ----------

def test_make_node_terminal_value_is_from_hero_perspective():
    board = make_empty_board(4)
    board = apply_move(board, 0, X)
    board = apply_move(board, 1, X)
    board = apply_move(board, 2, X)
    board = apply_move(board, 3, X)  # X just completed four in a row
    node = _make_node(board, to_move=O, hero=X, height=2, just_won=True)
    assert node.is_terminal is True
    assert node.terminal_value == 1.0  # hero (X) was the one who just won

    node_from_opponent_pov = _make_node(board, to_move=O, hero=O, height=2, just_won=True)
    assert node_from_opponent_pov.terminal_value == 0.0  # hero (O) did not win


def test_make_node_draw_is_terminal_zero_value():
    # a full board with no win is a draw -- is_full triggers is_terminal
    width, height = 2, 2
    board = make_empty_board(width)
    for col, player in [(0, X), (1, O), (0, O), (1, X)]:
        board = apply_move(board, col, player)
    node = _make_node(board, to_move=X, hero=X, height=height, just_won=False)
    assert node.is_terminal is True
    assert node.terminal_value == 0.0


# ---------- simulate: deterministic via scripted rng ----------

def test_simulate_returns_one_when_hero_wins_the_rollout():
    board = make_empty_board(5)
    rng = _ScriptedRng([0, 1, 0, 1, 0, 1, 0])  # X:0,0,0,0 (vertical 4) ; O:1,1,1
    value = simulate(board, to_move=X, hero=X, width=5, height=5, rng=rng)
    assert value == 1.0


def test_simulate_returns_zero_when_opponent_wins_the_rollout():
    board = make_empty_board(5)
    # X: col0,col0,col0,col1 (never 4-in-a-row) ; O: col2,col2,col2,col2 (vertical 4th win)
    rng = _ScriptedRng([0, 2, 0, 2, 0, 2, 1, 2])
    value = simulate(board, to_move=X, hero=X, width=5, height=5, rng=rng)
    assert value == 0.0  # opponent (O) won, not hero


# ---------- simulate: guidance_depth_cap ablation ----------

def test_simulate_guidance_depth_cap_none_matches_honest_rollout():
    board = make_empty_board(5)
    rng = _ScriptedRng([0, 1, 0, 1, 0, 1, 0])
    value = simulate(board, to_move=X, hero=X, width=5, height=5, rng=rng, guidance_depth_cap=None)
    assert value == 1.0  # explicit None is a no-op, same as omitting the argument


def test_simulate_depth_cap_returns_neutral_value_when_exhausted_without_a_win():
    board = make_empty_board(5)
    # Same opening as the opponent-wins fixture, truncated before O's 4th piece lands (ply 8).
    rng = _ScriptedRng([0, 2, 0, 2, 0])
    value = simulate(board, to_move=X, hero=X, width=5, height=5, rng=rng, guidance_depth_cap=5)
    assert value == 0.5


def test_simulate_depth_cap_still_returns_decisive_win_within_cap():
    board = make_empty_board(5)
    rng = _ScriptedRng([0, 1, 0, 1, 0, 1, 0])  # X wins on the 7th ply
    value = simulate(board, to_move=X, hero=X, width=5, height=5, rng=rng, guidance_depth_cap=7)
    assert value == 1.0


# ---------- run_search: end-to-end smoke test on a known forced-win position ----------

def test_run_search_solves_the_known_double_threat_puzzle_most_of_the_time():
    # Same hand-built fixture as test_connect_four_engine.py: X at col1,col2;
    # col3 opens a double threat at col0/col4 that O can only block one side of.
    width, height, hero = 5, 2, X
    board = make_empty_board(width)
    board = apply_move(board, 1, X)
    board = apply_move(board, 2, X)

    def _root_for(pre_board):
        graph_nodes = {}
        key = canonical_state(pre_board, hero)
        graph_nodes[key] = MCTSNode(board=pre_board, to_move=hero, untried_moves=list(range(width)))
        return graph_nodes, key

    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = ClassicalMCTSConfig(merge_enabled=True)
        # build a graph rooted at the pre-built position directly (bypass create_root's replay)
        graph_nodes, start_key = _root_for(board)

        class _Graph:
            nodes = graph_nodes
            root_key = start_key

        graph = _Graph()
        expansions_used = 0
        iterations = 0
        while expansions_used < 100 and iterations < 2000:
            iterations += 1
            path, path_moves = select(graph, config)
            leaf = graph.nodes[path[-1]]
            if leaf.is_terminal:
                backup(graph, path, leaf.terminal_value)
                continue
            child_key = expand(graph, path[-1], path_moves, hero, height, config, rng)
            expansions_used += 1
            path.append(child_key)
            child = graph.nodes[child_key]
            value = child.terminal_value if child.is_terminal else simulate(
                child.board, child.to_move, hero, width, height, rng
            )
            backup(graph, path, value)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7  # a trivial K=3 puzzle should solve reliably, not just occasionally


def test_run_random_search_solves_the_known_double_threat_puzzle_most_of_the_time():
    width, height, hero = 5, 2, X
    board = make_empty_board(width)
    board = apply_move(board, 1, X)
    board = apply_move(board, 2, X)
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        if run_random_search(board, hero, hero, width, height, budget=100, rng=rng):
            solved_count += 1
    assert solved_count >= trials * 0.7
