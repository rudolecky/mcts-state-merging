"""Verification for classical_mcts_morris.py, independent of any model --
pure search logic, no GPU/network dependency at all. Mirrors the other two
classical_mcts_*.py test suites, plus an explicit cycle-avoidance
regression test (the exact bug class hit and fixed during the 8-puzzle
work, since Morris's movement phase is equally reversible).
"""

import random

from mcts_phase0.classical_mcts_morris import (
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
from mcts_phase0.datasets.morris_engine import O, X, replay_placements


class _ScriptedRng:
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
    untried_edge = MCTSEdge(child_key=((1,), (2,)), n_edge=0)
    assert _ucb1_score(parent, untried_edge, child, c=1.4) == float("inf")


def test_select_picks_expected_ucb1_argmax():
    root = MCTSNode(board=(), to_move=X, n_visits=10)
    child_a = MCTSNode(board=(), to_move=O, n_visits=8, w_value=6.0)  # Q=0.75
    child_b = MCTSNode(board=(), to_move=O, n_visits=2, w_value=0.2)  # Q=0.1
    root.children = {(0, 3): MCTSEdge(child_key="a", n_edge=8), (1, 4): MCTSEdge(child_key="b", n_edge=2)}
    graph_nodes = {(): root, "a": child_a, "b": child_b}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    config = ClassicalMCTSConfig(merge_enabled=True, c=0.1)
    path, _ = select(_Graph(), config)
    assert path == [(), "a"]


# ---------- backup pooling on a hand-built two-parent DAG ----------

def test_backup_pools_at_shared_node_keeps_edge_counts_local():
    root = MCTSNode(board=(), to_move=X)
    a = MCTSNode(board=(), to_move=O)
    b = MCTSNode(board=(), to_move=O)
    m = MCTSNode(board=(), to_move=X)
    root.children = {(0, 1): MCTSEdge(child_key="a"), (0, 3): MCTSEdge(child_key="b")}
    a.children = {(4, 5): MCTSEdge(child_key="m")}
    b.children = {(4, 6): MCTSEdge(child_key="m")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "m": m}
        root_key = ()

    graph = _Graph()
    backup(graph, [(), "a", "m"], 0.8)
    backup(graph, [(), "b", "m"], 0.2)

    assert m.n_visits == 2
    assert m.w_value == 1.0
    assert a.children[(4, 5)].n_edge == 1
    assert b.children[(4, 6)].n_edge == 1
    assert root.n_visits == 2


# ---------- expand: merge-vs-tree structural neutrality ----------

def test_expand_merges_transposed_paths_when_enabled_not_when_disabled():
    # ((0,1),(4,0),(1,4)) vs ((0,3),(4,0),(3,4)) -- the same transposition
    # fixture test_morris_engine.py validates at the engine level.
    start_board = replay_placements((0, 4, 8, 2, 6, 5))
    hero = X

    def _play(config, move_sequence):
        graph = create_root(start_board, X, config.merge_enabled)
        key = graph.root_key
        path_moves = ()
        for move in move_sequence:
            rng = _ScriptedRng([move])
            key = expand(graph, key, path_moves, hero, config, rng)
            path_moves = path_moves + (move,)
        return graph, key

    treatment_cfg = ClassicalMCTSConfig(merge_enabled=True)
    baseline_cfg = ClassicalMCTSConfig(merge_enabled=False)

    seq_a = ((0, 1), (4, 0), (1, 4))
    seq_b = ((0, 3), (4, 0), (3, 4))

    t_graph_a, t_key_a = _play(treatment_cfg, seq_a)
    t_graph_b, t_key_b = _play(treatment_cfg, seq_b)
    assert t_key_a == t_key_b

    b_graph_a, b_key_a = _play(baseline_cfg, seq_a)
    b_graph_b, b_key_b = _play(baseline_cfg, seq_b)
    assert b_key_a != b_key_b
    assert b_graph_a.nodes[b_key_a].board == b_graph_b.nodes[b_key_b].board


# ---------- terminal value convention ----------

def test_make_node_win_value_is_from_hero_perspective():
    board = (X, X, X, None, None, None, None, None, None)
    node = _make_node(board, to_move=O, hero=X, just_won_by=X)
    assert node.is_terminal is True
    assert node.terminal_value == 1.0

    node_from_opponent_pov = _make_node(board, to_move=O, hero=O, just_won_by=X)
    assert node_from_opponent_pov.terminal_value == 0.0


def test_make_node_boxed_in_side_loses():
    # X pieces at 6,7,8 all have only occupied neighbors -- a reachable,
    # verified boxed-in position (found by exhaustive combinatorial search).
    board = (None, None, None, O, O, O, X, X, X)
    node = _make_node(board, to_move=X, hero=X, just_won_by=None)
    assert node.is_terminal is True
    assert node.terminal_value == 0.0  # hero (X) is the one boxed in -> loses

    node_from_opponent_pov = _make_node(board, to_move=X, hero=O, just_won_by=None)
    assert node_from_opponent_pov.terminal_value == 1.0  # opponent's (X's) box-in is a win for hero O


def test_make_node_non_terminal_has_untried_moves():
    board = replay_placements((0, 4, 8, 2, 6, 5))
    node = _make_node(board, to_move=X, hero=X, just_won_by=None)
    assert node.is_terminal is False
    assert node.untried_moves != []


# ---------- simulate: deterministic via scripted rng ----------

def test_simulate_returns_one_when_hero_wins_the_rollout():
    # X pieces at 0,7,8; move (7,4) completes the (0,4,8) diagonal.
    board = (X, O, O, None, None, None, O, X, X)
    rng = _ScriptedRng([(7, 4)])
    value = simulate(board, to_move=X, hero=X, rollout_depth=5, rng=rng)
    assert value == 1.0


def test_simulate_returns_zero_when_opponent_wins_the_rollout():
    board = (X, O, O, None, None, None, O, X, X)
    rng = _ScriptedRng([(7, 4)])
    value = simulate(board, to_move=X, hero=O, rollout_depth=5, rng=rng)
    assert value == 0.0


def test_simulate_returns_zero_when_hero_gets_boxed_in():
    board = (None, None, None, O, O, O, X, X, X)
    rng = _ScriptedRng([])  # never consulted -- boxed in immediately
    value = simulate(board, to_move=X, hero=X, rollout_depth=5, rng=rng)
    assert value == 0.0


# ---------- run_search: cycle-avoidance regression + end-to-end smoke test ----------

def test_run_search_does_not_hang_on_a_reversible_position():
    # Directly guards against the exact bug hit and fixed during the
    # 8-puzzle work: select() deterministically preferring a cycling
    # child and never escaping, node count plateauing regardless of budget.
    board = replay_placements((0, 4, 8, 2, 6, 5))
    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True)
    graph = run_search(board, X, config, budget=200, rollout_depth=15, rng=rng)
    assert len(graph.nodes) > 50  # would plateau near a handful of nodes if stuck


def test_run_search_solves_a_forced_win_puzzle_most_of_the_time():
    # X pieces at 0,7,8; (7,4) wins immediately -- a trivial k_plies=1 puzzle.
    board = (X, O, O, None, None, None, O, X, X)
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = ClassicalMCTSConfig(merge_enabled=True)
        graph = run_search(board, X, config, budget=50, rollout_depth=10, rng=rng)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7


def test_run_random_search_solves_a_forced_win_puzzle_most_of_the_time():
    board = (X, O, O, None, None, None, O, X, X)
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        if run_random_search(board, X, X, budget=50, rollout_depth=10, rng=rng):
            solved_count += 1
    assert solved_count >= trials * 0.7
