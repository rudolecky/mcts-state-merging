"""Verification for classical_mcts_blocksworld.py, independent of any
model -- pure search logic, no GPU/network dependency at all. Mirrors the
other classical_mcts_*.py test suites, plus the established
cycle-avoidance regression test (Blocksworld's actions are reversible,
same real bug class as the 8-puzzle and Morris).
"""

import random

from pyperplan.heuristics.relaxation import hFFHeuristic

from mcts_phase0.classical_mcts_blocksworld import (
    ClassicalMCTSConfig,
    MCTSEdge,
    MCTSNode,
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
from mcts_phase0.datasets.blocksworld_engine import goal_state, make_task


_START_3 = frozenset({"ontable(1)", "ontable(2)", "ontable(3)", "clear(1)", "clear(2)", "clear(3)", "handempty"})


class _ScriptedRng:
    def __init__(self, choices):
        self._choices = list(choices)

    def choice(self, seq):
        val = self._choices.pop(0)
        assert val in seq, f"scripted choice {val} not in {seq}"
        return val


# ---------- _ucb1_score / select ----------

def test_ucb1_gives_infinite_priority_to_untried_edges():
    parent = MCTSNode(state=frozenset(), n_visits=10)
    child = MCTSNode(state=frozenset(), n_visits=3, w_value=2.0)
    untried_edge = MCTSEdge(child_key="c", n_edge=0)
    assert _ucb1_score(parent, untried_edge, child, c=1.4) == float("inf")


def test_select_picks_expected_ucb1_argmax():
    root = MCTSNode(state=frozenset(), n_visits=10)
    child_a = MCTSNode(state=frozenset(), n_visits=8, w_value=6.0)  # Q=0.75
    child_b = MCTSNode(state=frozenset(), n_visits=2, w_value=0.2)  # Q=0.1
    root.children = {"opA": MCTSEdge(child_key="a", n_edge=8), "opB": MCTSEdge(child_key="b", n_edge=2)}
    graph_nodes = {(): root, "a": child_a, "b": child_b}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    config = ClassicalMCTSConfig(merge_enabled=True, c=0.1)
    path, path_moves = select(_Graph(), config)
    assert path == [(), "a"]
    assert path_moves == ["opA"]


# ---------- backup pooling on a hand-built two-parent DAG ----------

def test_backup_pools_at_shared_node_keeps_edge_counts_local():
    root = MCTSNode(state=frozenset())
    a = MCTSNode(state=frozenset())
    b = MCTSNode(state=frozenset())
    m = MCTSNode(state=frozenset())
    root.children = {"opA": MCTSEdge(child_key="a"), "opB": MCTSEdge(child_key="b")}
    a.children = {"opM1": MCTSEdge(child_key="m")}
    b.children = {"opM2": MCTSEdge(child_key="m")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "m": m}
        root_key = ()

    graph = _Graph()
    backup(graph, [(), "a", "m"], ["opA", "opM1"], 0.8)
    backup(graph, [(), "b", "m"], ["opB", "opM2"], 0.2)

    assert m.n_visits == 2
    assert m.w_value == 1.0
    assert a.children["opM1"].n_edge == 1
    assert b.children["opM2"].n_edge == 1
    assert root.n_visits == 2


# ---------- expand: merge-vs-tree structural neutrality ----------

def test_expand_merges_transposed_paths_when_enabled_not_when_disabled():
    # Same independent-block-pair transposition test_blocksworld_engine.py validates.
    task = make_task(4)
    goal = goal_state(4)
    start = frozenset({
        "ontable(1)", "ontable(2)", "ontable(3)", "ontable(4)",
        "clear(1)", "clear(2)", "clear(3)", "clear(4)", "handempty",
    })

    def _op(state, name):
        from mcts_phase0.datasets.blocksworld_engine import legal_moves
        return next(o for o in legal_moves(state, task) if o.name == name)

    def _play(config, names):
        graph = create_root(start, goal, task, config.merge_enabled)
        key = graph.root_key
        path_moves = []
        for name in names:
            node = graph.nodes[key]
            op = _op(node.state, name)
            rng = _ScriptedRng([op])
            key, _ = expand(graph, key, path_moves, goal, task, config, rng)
            path_moves.append(op)
        return graph, key

    treatment_cfg = ClassicalMCTSConfig(merge_enabled=True)
    baseline_cfg = ClassicalMCTSConfig(merge_enabled=False)

    seq_a = ["pickup(1)", "stack(1,2)", "pickup(3)", "stack(3,4)"]
    seq_b = ["pickup(3)", "stack(3,4)", "pickup(1)", "stack(1,2)"]

    t_graph_a, t_key_a = _play(treatment_cfg, seq_a)
    t_graph_b, t_key_b = _play(treatment_cfg, seq_b)
    assert t_key_a == t_key_b

    b_graph_a, b_key_a = _play(baseline_cfg, seq_a)
    b_graph_b, b_key_b = _play(baseline_cfg, seq_b)
    assert b_key_a != b_key_b
    assert b_graph_a.nodes[b_key_a].state == b_graph_b.nodes[b_key_b].state


# ---------- terminal value convention ----------

def test_make_node_terminal_value_is_one_when_goal_reached():
    task = make_task(3)
    goal = goal_state(3)
    node = _make_node(goal, goal, task)
    assert node.is_terminal is True
    assert node.terminal_value == 1.0
    assert node.untried_moves == []


def test_make_node_non_goal_has_untried_moves():
    task = make_task(3)
    goal = goal_state(3)
    node = _make_node(_START_3, goal, task)
    assert node.is_terminal is False
    assert node.untried_moves != []


# ---------- simulate: deterministic via scripted rng ----------

def test_simulate_returns_one_immediately_if_already_solved():
    task = make_task(3)
    goal = goal_state(3)
    rng = _ScriptedRng([])
    assert simulate(goal, goal, task, rollout_depth=5, rng=rng) == 1.0


def test_simulate_returns_zero_when_depth_cap_exhausted_without_solving():
    task = make_task(3)
    goal = goal_state(3)
    # from the fully-scattered start, one pickup then one putdown never solves
    from mcts_phase0.datasets.blocksworld_engine import legal_moves
    pickup_1 = next(o for o in legal_moves(_START_3, task) if o.name == "pickup(1)")
    intermediate = pickup_1.apply(_START_3)
    putdown_1 = next(o for o in legal_moves(intermediate, task) if o.name == "putdown(1)")
    rng = _ScriptedRng([pickup_1, putdown_1])
    value = simulate(_START_3, goal, task, rollout_depth=2, rng=rng)
    assert value == 0.0


# ---------- run_search: cycle-avoidance regression + end-to-end smoke test ----------

def test_run_search_does_not_hang_on_a_reversible_position():
    task = make_task(4)
    goal = goal_state(4)
    start = frozenset({
        "ontable(1)", "ontable(2)", "ontable(3)", "ontable(4)",
        "clear(1)", "clear(2)", "clear(3)", "clear(4)", "handempty",
    })
    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True)
    graph = run_search(start, goal, task, config, budget=200, rollout_depth=15, rng=rng)
    assert len(graph.nodes) > 50  # would plateau near a handful of nodes if stuck


def test_run_search_solves_a_small_puzzle_most_of_the_time():
    task = make_task(3)
    goal = goal_state(3)
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = ClassicalMCTSConfig(merge_enabled=True)
        graph = run_search(_START_3, goal, task, config, budget=100, rollout_depth=15, rng=rng)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7


def test_run_random_search_solves_a_small_puzzle_most_of_the_time():
    task = make_task(3)
    goal = goal_state(3)
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        if run_random_search(_START_3, goal, task, budget=100, rollout_depth=15, rng=rng):
            solved_count += 1
    assert solved_count >= trials * 0.7


# ---------- heuristic value_source ----------

def test_heuristic_value_is_one_at_goal():
    task = make_task(3)
    goal = goal_state(3)
    heuristic = hFFHeuristic(task)
    assert _heuristic_value(goal, heuristic) == 1.0


def test_heuristic_value_below_one_away_from_goal():
    task = make_task(3)
    goal = goal_state(3)
    heuristic = hFFHeuristic(task)
    assert _START_3 != goal
    assert 0.0 < _heuristic_value(_START_3, heuristic) < 1.0


def test_run_search_heuristic_mode_never_calls_simulate(monkeypatch):
    import mcts_phase0.classical_mcts_blocksworld as blocksworld_mod

    task = make_task(3)
    goal = goal_state(3)
    heuristic = hFFHeuristic(task)

    def _boom(*a, **kw):
        raise AssertionError("simulate() should not be called in heuristic mode")

    monkeypatch.setattr(blocksworld_mod, "simulate", _boom)

    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True, value_source="heuristic")
    graph = run_search(_START_3, goal, task, config, budget=100, rollout_depth=15, rng=rng, heuristic=heuristic)
    assert is_solved(graph)  # a trivial 3-block puzzle should solve easily under real guidance
