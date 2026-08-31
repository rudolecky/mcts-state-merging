"""Verification for classical_mcts_sokoban.py, independent of any model --
pure search logic, no GPU/network dependency at all. Mirrors the other
three classical_mcts_*.py test suites, plus the established
cycle-avoidance regression test (walking is fully reversible here too).
"""

import random

from mcts_phase0.classical_mcts_sokoban import (
    ClassicalMCTSConfig,
    MCTSEdge,
    MCTSNode,
    _box_goal_heuristic,
    _player_box_heuristic,
    _heuristic_value,
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
from mcts_phase0.datasets.sokoban_engine import GOAL, make_state


class _ScriptedRng:
    def __init__(self, choices):
        self._choices = list(choices)

    def choice(self, seq):
        val = self._choices.pop(0)
        assert val in seq, f"scripted choice {val} not in {seq}"
        return val


# ---------- _ucb1_score / select ----------

def test_ucb1_gives_infinite_priority_to_untried_edges():
    parent = MCTSNode(state=(), n_visits=10)
    child = MCTSNode(state=(), n_visits=3, w_value=2.0)
    untried_edge = MCTSEdge(child_key=(1, 0), n_edge=0)
    assert _ucb1_score(parent, untried_edge, child, c=1.4) == float("inf")


def test_select_picks_expected_ucb1_argmax():
    root = MCTSNode(state=(), n_visits=10)
    child_a = MCTSNode(state=(), n_visits=8, w_value=6.0)  # Q=0.75
    child_b = MCTSNode(state=(), n_visits=2, w_value=0.2)  # Q=0.1
    root.children = {(-1, 0): MCTSEdge(child_key="a", n_edge=8), (1, 0): MCTSEdge(child_key="b", n_edge=2)}
    graph_nodes = {(): root, "a": child_a, "b": child_b}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    config = ClassicalMCTSConfig(merge_enabled=True, c=0.1)
    path, _ = select(_Graph(), config)
    assert path == [(), "a"]


# ---------- backup pooling on a hand-built two-parent DAG ----------

def test_backup_pools_at_shared_node_keeps_edge_counts_local():
    root = MCTSNode(state=())
    a = MCTSNode(state=())
    b = MCTSNode(state=())
    m = MCTSNode(state=())
    root.children = {(-1, 0): MCTSEdge(child_key="a"), (1, 0): MCTSEdge(child_key="b")}
    a.children = {(0, 1): MCTSEdge(child_key="m")}
    b.children = {(0, -1): MCTSEdge(child_key="m")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "m": m}
        root_key = ()

    graph = _Graph()
    backup(graph, [(), "a", "m"], 0.8)
    backup(graph, [(), "b", "m"], 0.2)

    assert m.n_visits == 2
    assert m.w_value == 1.0
    assert a.children[(0, 1)].n_edge == 1
    assert b.children[(0, -1)].n_edge == 1
    assert root.n_visits == 2


# ---------- expand: merge-vs-tree structural neutrality ----------

def test_expand_merges_transposed_paths_when_enabled_not_when_disabled():
    # Same pure-walk transposition fixture test_sokoban_engine.py validates:
    # ((-1,0),(-1,0),(1,0)) vs ((-1,0),(0,-1),(0,1)) from (52,(53,)).
    start_state = make_state(52, (53,))

    def _play(config, move_sequence):
        graph = create_root(start_state, config.merge_enabled)
        key = graph.root_key
        path_moves = ()
        for move in move_sequence:
            rng = _ScriptedRng([move])
            key = expand(graph, key, path_moves, config, rng)
            path_moves = path_moves + (move,)
        return graph, key

    treatment_cfg = ClassicalMCTSConfig(merge_enabled=True)
    baseline_cfg = ClassicalMCTSConfig(merge_enabled=False)

    seq_a = ((-1, 0), (-1, 0), (1, 0))
    seq_b = ((-1, 0), (0, -1), (0, 1))

    t_graph_a, t_key_a = _play(treatment_cfg, seq_a)
    t_graph_b, t_key_b = _play(treatment_cfg, seq_b)
    assert t_key_a == t_key_b

    b_graph_a, b_key_a = _play(baseline_cfg, seq_a)
    b_graph_b, b_key_b = _play(baseline_cfg, seq_b)
    assert b_key_a != b_key_b
    assert b_graph_a.nodes[b_key_a].state == b_graph_b.nodes[b_key_b].state


# ---------- terminal value convention ----------

def test_make_node_terminal_value_is_one_when_solved():
    node = _make_node(make_state(GOAL - 1, (GOAL,)))  # box already on the goal
    assert node.is_terminal is True
    assert node.terminal_value == 1.0
    assert node.untried_moves == []


def test_make_node_non_solved_is_not_terminal():
    node = _make_node(make_state(53, (54,)))  # one push away, not yet solved
    assert node.is_terminal is False
    assert node.terminal_value is None
    assert node.untried_moves != []


# ---------- simulate: deterministic via scripted rng ----------

def test_simulate_returns_one_when_rollout_solves():
    state = make_state(53, (54,))  # one push right solves it
    rng = _ScriptedRng([(0, 1)])
    value = simulate(state, rollout_depth=5, rng=rng)
    assert value == 1.0


def test_simulate_returns_one_immediately_if_already_solved():
    state = make_state(GOAL - 1, (GOAL,))
    rng = _ScriptedRng([])  # never consulted
    value = simulate(state, rollout_depth=5, rng=rng)
    assert value == 1.0


def test_simulate_returns_zero_when_depth_cap_exhausted_without_solving():
    state = make_state(52, (53,))  # 2 pushes needed
    # walk away and back, never pushing toward the goal
    rng = _ScriptedRng([(-1, 0), (1, 0)])
    value = simulate(state, rollout_depth=2, rng=rng)
    assert value == 0.0


# ---------- run_search: cycle-avoidance regression + end-to-end smoke test ----------

def test_run_search_does_not_hang_on_a_reversible_position():
    # Directly guards against the exact bug class hit and fixed during the
    # 8-puzzle work: select() deterministically preferring a cycling child
    # and never escaping.
    state = make_state(52, (53,))
    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True)
    graph = run_search(state, config, budget=200, rollout_depth=15, rng=rng)
    assert len(graph.nodes) > 50  # would plateau near a handful of nodes if stuck


def test_run_search_solves_a_two_push_puzzle_most_of_the_time():
    state = make_state(52, (53,))  # 2 pushes needed
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = ClassicalMCTSConfig(merge_enabled=True)
        graph = run_search(state, config, budget=50, rollout_depth=10, rng=rng)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7


def test_run_random_search_solves_a_two_push_puzzle_most_of_the_time():
    state = make_state(52, (53,))
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        if run_random_search(state, budget=50, rollout_depth=10, rng=rng):
            solved_count += 1
    assert solved_count >= trials * 0.7


# ---------- heuristic value_source ----------

def test_box_goal_heuristic_zero_at_goal():
    assert _box_goal_heuristic(make_state(GOAL - 1, (GOAL,))) == 0


def test_box_goal_heuristic_hand_computed():
    # GOAL=55 -> row 5, col 5 (width 10). Box at 53 -> row 5, col 3: distance 2.
    state = make_state(52, (53,))
    assert _box_goal_heuristic(state) == 2


def test_player_box_heuristic_hand_computed():
    # player at 52 -> row 5, col 2; box at 53 -> row 5, col 3: distance 1.
    state = make_state(52, (53,))
    assert _player_box_heuristic(state) == 1


def test_player_box_heuristic_picks_nearest_of_multiple_boxes():
    state = make_state(0, (99, 1))  # player at 0 (row0,col0): box 99 far, box 1 adjacent
    assert _player_box_heuristic(state) == 1


def test_heuristic_value_increases_as_box_and_player_get_closer_to_goal():
    # box.state[1]==(GOAL,) is box-solved (h_box=0), player adjacent (h_player=1) -- as
    # close to "done" as _heuristic_value ever sees in real search (terminal states bypass
    # it entirely, see _leaf_value), still strictly less than a state further from goal.
    near_goal = make_state(GOAL - 1, (GOAL,))
    farther = make_state(52, (53,))
    assert 0.0 < _heuristic_value(farther) < _heuristic_value(near_goal) < 1.0


def test_run_search_heuristic_mode_never_calls_simulate(monkeypatch):
    import mcts_phase0.classical_mcts_sokoban as sokoban_mod

    state = make_state(52, (53,))

    def _boom(*a, **kw):
        raise AssertionError("simulate() should not be called in heuristic mode")

    monkeypatch.setattr(sokoban_mod, "simulate", _boom)

    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True, value_source="heuristic")
    graph = run_search(state, config, budget=50, rollout_depth=10, rng=rng)
    assert is_solved(graph)  # a trivial 2-push puzzle should solve easily under real guidance
