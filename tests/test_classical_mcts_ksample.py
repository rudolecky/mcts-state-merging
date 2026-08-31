"""Verification for classical_mcts_ksample.py, independent of any model --
pure search logic, no GPU/network dependency. The core, load-bearing
behavior this module exists to test: two independent draws of the
identical move within one K-batch stay distinct nodes in baseline but
merge into one in treatment -- everything else mirrors the other
classical_mcts_*.py test suites.
"""

import random

from mcts_phase0.classical_mcts import simulate
from mcts_phase0.classical_mcts_ksample import (
    ClassicalMCTSConfig,
    MCTSEdge,
    MCTSGraph,
    MCTSNode,
    _make_node,
    _ucb1_score,
    backup,
    create_root,
    expand_batch,
    is_solved,
    run_search,
    select,
)
from mcts_phase0.datasets.connect_four_engine import X, apply_move, make_empty_board


class _ScriptedRng:
    def __init__(self, choices):
        self._choices = list(choices)

    def choice(self, seq):
        val = self._choices.pop(0)
        assert val in seq, f"scripted choice {val} not in {seq}"
        return val


# ---------- the load-bearing behavior ----------

def test_two_identical_draws_stay_distinct_in_baseline_but_merge_in_treatment():
    width, height, hero = 5, 2, X
    rng_treat = _ScriptedRng([2, 2])  # both draws pick column 2
    treat_cfg = ClassicalMCTSConfig(merge_enabled=True, K=2)
    treat_graph = create_root((), X, width, height, treat_cfg.merge_enabled)
    child_keys_treat = expand_batch(treat_graph, treat_graph.root_key, hero, height, treat_cfg, rng_treat)
    assert child_keys_treat[0] == child_keys_treat[1]  # merged: identical resulting state
    assert len(treat_graph.nodes) == 2  # root + 1 merged child, not 3

    rng_base = _ScriptedRng([2, 2])
    base_cfg = ClassicalMCTSConfig(merge_enabled=False, K=2)
    base_graph = create_root((), X, width, height, base_cfg.merge_enabled)
    child_keys_base = expand_batch(base_graph, base_graph.root_key, hero, height, base_cfg, rng_base)
    assert child_keys_base[0] != child_keys_base[1]  # never merges, even an exact repeat
    assert len(base_graph.nodes) == 3  # root + 2 distinct baseline children
    # but the underlying board is still identical for both
    assert base_graph.nodes[child_keys_base[0]].board == base_graph.nodes[child_keys_base[1]].board


def test_backup_after_merged_draws_pools_visits_at_the_shared_node():
    width, height, hero = 5, 2, X
    rng = _ScriptedRng([2, 2])
    config = ClassicalMCTSConfig(merge_enabled=True, K=2)
    graph = create_root((), X, width, height, config.merge_enabled)
    child_keys = expand_batch(graph, graph.root_key, hero, height, config, rng)
    backup(graph, [graph.root_key, child_keys[0]], [0], 0.8)
    backup(graph, [graph.root_key, child_keys[1]], [1], 0.2)
    merged = graph.nodes[child_keys[0]]
    assert merged.n_visits == 2
    assert merged.w_value == 1.0  # pooled: 0.8 + 0.2
    # both draw-indexed edges point at the same child and each recorded its own visit
    assert graph.nodes[graph.root_key].children[0].n_edge == 1
    assert graph.nodes[graph.root_key].children[1].n_edge == 1


# ---------- _ucb1_score / select ----------

def test_ucb1_gives_infinite_priority_to_untried_edges():
    parent = MCTSNode(board=(), to_move=X, n_visits=10)
    child = MCTSNode(board=(), to_move="O", n_visits=3, w_value=2.0)
    untried_edge = MCTSEdge(child_key="c", n_edge=0)
    assert _ucb1_score(parent, untried_edge, child, c=1.4) == float("inf")


def test_select_picks_expected_ucb1_argmax():
    root = MCTSNode(board=(), to_move=X, n_visits=10, expanded=True)
    child_a = MCTSNode(board=(), to_move="O", n_visits=8, w_value=6.0)  # Q=0.75
    child_b = MCTSNode(board=(), to_move="O", n_visits=2, w_value=0.2)  # Q=0.1
    root.children = {0: MCTSEdge(child_key="a", n_edge=8), 1: MCTSEdge(child_key="b", n_edge=2)}
    graph_nodes = {(): root, "a": child_a, "b": child_b}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    config = ClassicalMCTSConfig(merge_enabled=True, c=0.1)
    path, edge_draws = select(_Graph(), config)
    assert path == [(), "a"]
    assert edge_draws == [0]


# ---------- terminal value convention ----------

def test_make_node_terminal_value_is_from_hero_perspective():
    board = make_empty_board(4)
    board = apply_move(board, 0, X)
    board = apply_move(board, 1, X)
    board = apply_move(board, 2, X)
    board = apply_move(board, 3, X)
    node = _make_node(board, to_move="O", hero=X, height=2, just_won=True)
    assert node.is_terminal is True
    assert node.terminal_value == 1.0


# ---------- run_search: end-to-end smoke test on a known forced-win position ----------

def test_run_search_solves_the_known_double_threat_puzzle_most_of_the_time():
    # Same fixture as test_classical_mcts.py: X at col1,col2 on a width=5,height=2 board --
    # a hand-built position (two X moves in a row), so bypass create_root/replay's
    # alternating-turns assumption, same precedent as classical_mcts.py's own test.
    width, height, hero = 5, 2, X
    board = make_empty_board(width)
    board = apply_move(board, 1, X)
    board = apply_move(board, 2, X)

    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = ClassicalMCTSConfig(merge_enabled=True, K=4)
        graph = MCTSGraph(nodes={(): MCTSNode(board=board, to_move=hero)}, root_key=())
        expansions_used = 0
        iterations = 0
        while expansions_used < 100 and iterations < 2000:
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
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7
