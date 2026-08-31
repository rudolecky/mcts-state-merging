"""Verification for guct_uniform_blocksworld.py -- the paper's own algorithm
(LCB1-Uniform bandit, Full Bellman backup, hFF heuristic reward, no rollout),
reimplemented faithfully. Uses the real hFFHeuristic throughout (cheap,
~30us/call) rather than a mock, matching this project's convention of testing
against real oracles wherever affordable.
"""

import math
import random

from pyperplan.heuristics.relaxation import hFFHeuristic

from mcts_phase0.datasets.blocksworld_engine import goal_state, make_task
from mcts_phase0.guct_uniform_blocksworld import (
    GUCTUniformConfig,
    MCTSEdge,
    MCTSNode,
    _lcb1_uniform_score,
    backup,
    create_root,
    expand,
    is_solved,
    run_random_search,
    run_search,
    select,
)


class _ScriptedRng:
    def __init__(self, choices):
        self._choices = list(choices)

    def choice(self, seq):
        val = self._choices.pop(0)
        assert val in seq, f"scripted choice {val} not in {seq}"
        return val


# ---------- _lcb1_uniform_score ----------

def test_lcb1_uniform_score_equals_the_sample_when_only_one_pull_total():
    # t=1, T=1: log(1)=0, so the exploration term vanishes and the score is
    # exactly the single observed sample (l_hat == u_hat == that sample).
    edge = MCTSEdge(child_key="x", t=1, l_hat=3.0, u_hat=3.0)
    assert _lcb1_uniform_score(edge, T=1) == 3.0


def test_lcb1_uniform_score_matches_hand_computed_value():
    edge = MCTSEdge(child_key="x", t=1, l_hat=2.0, u_hat=6.0)
    expected = (6.0 + 2.0) / 2.0 - (6.0 - 2.0) * math.sqrt(6.0 * 1 * math.log(4))
    assert _lcb1_uniform_score(edge, T=4) == expected


def test_untried_edge_gets_negative_infinity_priority():
    edge = MCTSEdge(child_key="x", t=0)
    assert _lcb1_uniform_score(edge, T=10) == float("-inf")


# ---------- select: picks the minimum-score child ----------

def test_select_picks_expected_lcb1_uniform_argmin():
    root = MCTSNode(state=frozenset(), n_visits=10)
    child_a = MCTSNode(state=frozenset())
    child_b = MCTSNode(state=frozenset())
    # a: tight, mediocre (score ~4.0). b: wide spread, lower mid -- more attractive.
    root.children = {
        "opA": MCTSEdge(child_key="a", t=8, l_hat=4.0, u_hat=4.0),
        "opB": MCTSEdge(child_key="b", t=8, l_hat=0.0, u_hat=2.0),
    }
    graph_nodes = {(): root, "a": child_a, "b": child_b}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    config = GUCTUniformConfig(merge_enabled=True)
    path, path_moves = select(_Graph(), config)
    assert path == [(), "b"]
    assert path_moves == ["opB"]


# ---------- backup: Full Bellman pools the MINIMUM, not the average ----------

def test_full_bellman_backup_pools_minimum_not_average():
    # Same shared-child DAG shape as classical_mcts_blocksworld's own backup
    # test, but this backup rule pools by min, not by running mean.
    root = MCTSNode(state=frozenset(), h_gbfs=5.0)
    a = MCTSNode(state=frozenset(), h_gbfs=5.0)
    b = MCTSNode(state=frozenset(), h_gbfs=5.0)
    m = MCTSNode(state=frozenset(), h_gbfs=5.0)
    root.children = {"opA": MCTSEdge(child_key="a"), "opB": MCTSEdge(child_key="b")}
    a.children = {"opM1": MCTSEdge(child_key="m")}
    b.children = {"opM2": MCTSEdge(child_key="m")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "m": m}
        root_key = ()

    graph = _Graph()
    backup(graph, [(), "a", "m"], ["opA", "opM1"], 0.8)
    backup(graph, [(), "b", "m"], ["opB", "opM2"], 0.2)

    assert m.h_gbfs == 0.2  # min of {0.8, 0.2}, not their average
    assert a.h_gbfs == 0.8  # only ever saw its own path's sample
    assert b.h_gbfs == 0.2
    assert root.h_gbfs == 0.2  # propagates all the way up
    assert m.n_visits == 2
    assert root.n_visits == 2

    # Each edge's own (t, l_hat, u_hat) reflects only its own path's samples.
    edge_a_m = a.children["opM1"]
    edge_b_m = b.children["opM2"]
    assert edge_a_m.t == 1 and edge_a_m.l_hat == 0.8 and edge_a_m.u_hat == 0.8
    assert edge_b_m.t == 1 and edge_b_m.l_hat == 0.2 and edge_b_m.u_hat == 0.2


# ---------- expand: merge-vs-tree structural neutrality ----------

def test_expand_merges_transposed_paths_when_enabled_not_when_disabled():
    task = make_task(4)
    goal = goal_state(4)
    heuristic = hFFHeuristic(task)
    start = frozenset({
        "ontable(1)", "ontable(2)", "ontable(3)", "ontable(4)",
        "clear(1)", "clear(2)", "clear(3)", "clear(4)", "handempty",
    })

    def _op(state, name):
        from mcts_phase0.datasets.blocksworld_engine import legal_moves
        return next(o for o in legal_moves(state, task) if o.name == name)

    def _play(graph, config, names):
        # Baseline keys are now a plain per-graph counter (see expand()'s
        # docstring), so the meaningful check is within ONE shared graph:
        # do two move orders land on the same node (merge) or not (tree)?
        key = graph.root_key
        for name in names:
            node = graph.nodes[key]
            op = _op(node.state, name)
            rng = _ScriptedRng([op])
            key, _ = expand(graph, key, goal, task, config, rng, heuristic)
        return key

    seq_a = ["pickup(1)", "stack(1,2)", "pickup(3)", "stack(3,4)"]
    seq_b = ["pickup(3)", "stack(3,4)", "pickup(1)", "stack(1,2)"]

    treatment_cfg = GUCTUniformConfig(merge_enabled=True)
    t_graph = create_root(start, goal, task, True, heuristic)
    t_key_a = _play(t_graph, treatment_cfg, seq_a)
    t_key_b = _play(t_graph, treatment_cfg, seq_b)
    assert t_key_a == t_key_b  # same resulting state -> the same shared node

    baseline_cfg = GUCTUniformConfig(merge_enabled=False)
    b_graph = create_root(start, goal, task, False, heuristic)
    b_key_a = _play(b_graph, baseline_cfg, seq_a)
    b_key_b = _play(b_graph, baseline_cfg, seq_b)
    assert b_key_a != b_key_b  # distinct nodes even though the state matches
    assert b_graph.nodes[b_key_a].state == b_graph.nodes[b_key_b].state


# ---------- run_search: cycle-avoidance regression + end-to-end smoke test ----------

def test_run_search_does_not_hang_on_a_reversible_position():
    task = make_task(4)
    goal = goal_state(4)
    heuristic = hFFHeuristic(task)
    start = frozenset({
        "ontable(1)", "ontable(2)", "ontable(3)", "ontable(4)",
        "clear(1)", "clear(2)", "clear(3)", "clear(4)", "handempty",
    })
    rng = random.Random(0)
    config = GUCTUniformConfig(merge_enabled=True)
    graph = run_search(start, goal, task, config, budget=200, rng=rng, heuristic=heuristic)
    assert len(graph.nodes) > 50  # would plateau near a handful of nodes if stuck


def test_run_search_solves_a_small_puzzle_most_of_the_time():
    task = make_task(3)
    goal = goal_state(3)
    heuristic = hFFHeuristic(task)
    start = frozenset({
        "ontable(1)", "ontable(2)", "ontable(3)",
        "clear(1)", "clear(2)", "clear(3)", "handempty",
    })
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = GUCTUniformConfig(merge_enabled=True)
        graph = run_search(start, goal, task, config, budget=50, rng=rng, heuristic=heuristic)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.9


def test_run_random_search_solves_a_small_puzzle_most_of_the_time():
    task = make_task(3)
    goal = goal_state(3)
    start = frozenset({
        "ontable(1)", "ontable(2)", "ontable(3)",
        "clear(1)", "clear(2)", "clear(3)", "handempty",
    })
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        if run_random_search(start, goal, task, budget=50, rollout_depth=15, rng=rng):
            solved_count += 1
    assert solved_count >= trials * 0.9
