"""Verification for classical_mcts_puzzle.py, independent of any model --
pure search logic, no GPU/network dependency at all. Mirrors
test_classical_mcts.py's suite for the single-agent, unbounded-depth
sliding-puzzle variant.
"""

import random

from mcts_phase0.classical_mcts_puzzle import (
    ClassicalMCTSConfig,
    MCTSEdge,
    MCTSNode,
    _heuristic_value,
    _make_node,
    _manhattan_heuristic,
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
from mcts_phase0.datasets.sliding_puzzle_engine import apply_move, goal_state


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
    parent = MCTSNode(state=(), n_visits=10)
    child = MCTSNode(state=(), n_visits=3, w_value=2.0)
    untried_edge = MCTSEdge(child_key=(1,), n_edge=0)
    assert _ucb1_score(parent, untried_edge, child, c=1.4) == float("inf")


def test_select_picks_expected_ucb1_argmax():
    root = MCTSNode(state=(), n_visits=10)
    child_a = MCTSNode(state=(), n_visits=8, w_value=6.0)  # Q=0.75, well-explored
    child_b = MCTSNode(state=(), n_visits=2, w_value=0.2)  # Q=0.1, barely explored
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
    root = MCTSNode(state=())
    a = MCTSNode(state=())
    b = MCTSNode(state=())
    m = MCTSNode(state=())  # shared merge target, two parents
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
    # (4,1,4) vs (4,3,4) -- two move sequences that diverge and reconverge
    # on the same state, with no intermediate state equal to the goal (so
    # expand() is legitimately callable at every step of both -- unlike
    # the raw DFS-found (2,1,2)/(4,5,2) pair, which passes through the
    # goal mid-path and would make expand() try to expand a terminal node).
    width, height = 3, 2

    def _play(config, move_sequence):
        graph = create_root(goal_state(width, height), width, height, config.merge_enabled)
        key = graph.root_key
        path_moves = ()
        for move in move_sequence:
            rng = _ScriptedRng([move])
            key = expand(graph, key, path_moves, width, height, config, rng)
            path_moves = path_moves + (move,)
        return graph, key

    treatment_cfg = ClassicalMCTSConfig(merge_enabled=True)
    baseline_cfg = ClassicalMCTSConfig(merge_enabled=False)

    t_graph_a, t_key_a = _play(treatment_cfg, (4, 1, 4))
    t_graph_b, t_key_b = _play(treatment_cfg, (4, 3, 4))
    assert t_key_a == t_key_b  # merged: identical state

    b_graph_a, b_key_a = _play(baseline_cfg, (4, 1, 4))
    b_graph_b, b_key_b = _play(baseline_cfg, (4, 3, 4))
    assert b_key_a != b_key_b  # never merges: distinct paths -> distinct keys
    # but the underlying puzzle state is still identical
    assert b_graph_a.nodes[b_key_a].state == b_graph_b.nodes[b_key_b].state


# ---------- terminal value convention ----------

def test_make_node_terminal_value_is_one_at_goal():
    width, height = 3, 2
    node = _make_node(goal_state(width, height), width, height)
    assert node.is_terminal is True
    assert node.terminal_value == 1.0
    assert node.untried_moves == []


def test_make_node_non_goal_is_not_terminal():
    width, height = 3, 2
    state = apply_move(goal_state(width, height), 4)  # one move away, not solved
    node = _make_node(state, width, height)
    assert node.is_terminal is False
    assert node.terminal_value is None
    assert node.untried_moves != []


# ---------- simulate: deterministic via scripted rng ----------

def test_simulate_returns_one_when_rollout_reaches_goal():
    width, height = 3, 2
    start = apply_move(goal_state(width, height), 4)  # blank moves to idx4; tile "5" now at idx5
    rng = _ScriptedRng([5])  # swap blank back with idx5 -> undoes the move, back to goal
    value = simulate(start, width, height, rollout_depth=5, rng=rng)
    assert value == 1.0


def test_simulate_returns_zero_when_depth_cap_exhausted_without_solving():
    width, height = 3, 2
    start = apply_move(goal_state(width, height), 4)  # one move from goal
    # deliberately move away and stay away for the whole rollout budget
    rng = _ScriptedRng([1, 2, 1, 2])  # bounce between two non-goal states
    value = simulate(start, width, height, rollout_depth=4, rng=rng)
    assert value == 0.0


def test_simulate_returns_one_immediately_if_already_at_goal():
    width, height = 3, 2
    rng = _ScriptedRng([])  # never consulted
    value = simulate(goal_state(width, height), width, height, rollout_depth=5, rng=rng)
    assert value == 1.0


# ---------- run_search: end-to-end smoke test on a known small puzzle ----------

def test_run_search_solves_a_three_move_puzzle_most_of_the_time():
    width, height = 3, 2
    # exactly 3 moves from goal (see test_sliding_puzzle_engine.py's hand-verified fixture family)
    start = apply_move(apply_move(goal_state(width, height), 2), 1)
    start = apply_move(start, 2)
    assert start != goal_state(width, height)

    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = ClassicalMCTSConfig(merge_enabled=True)
        graph = run_search(start, width, height, config, budget=50, rollout_depth=10, rng=rng)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7  # a trivial 3-move puzzle should solve reliably


def test_run_random_search_solves_a_three_move_puzzle_most_of_the_time():
    width, height = 3, 2
    start = apply_move(apply_move(goal_state(width, height), 2), 1)
    start = apply_move(start, 2)
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        if run_random_search(start, width, height, budget=50, rollout_depth=10, rng=rng):
            solved_count += 1
    assert solved_count >= trials * 0.7


# ---------- heuristic value_source ----------

def test_manhattan_heuristic_zero_at_goal():
    width, height = 3, 2
    assert _manhattan_heuristic(goal_state(width, height), width) == 0


def test_manhattan_heuristic_hand_computed():
    # width=2: goal = (1,2,3,0). Swap tiles 1 and 2 -> (2,1,3,0).
    # tile 2 sits at index 0, goal index 1 -> dist 1. tile 1 sits at index 1, goal index 0 -> dist 1.
    width = 2
    state = (2, 1, 3, 0)
    assert _manhattan_heuristic(state, width) == 2


def test_heuristic_value_is_one_at_goal_and_decreases_with_distance():
    width, height = 3, 2
    goal = goal_state(width, height)
    assert _heuristic_value(goal, width) == 1.0
    one_away = apply_move(goal, 2)
    assert 0.0 < _heuristic_value(one_away, width) < 1.0


def test_run_search_heuristic_mode_never_calls_simulate(monkeypatch):
    import mcts_phase0.classical_mcts_puzzle as puzzle_mod

    width, height = 3, 2
    start = apply_move(apply_move(goal_state(width, height), 2), 1)
    start = apply_move(start, 2)

    def _boom(*a, **kw):
        raise AssertionError("simulate() should not be called in heuristic mode")

    monkeypatch.setattr(puzzle_mod, "simulate", _boom)

    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True, value_source="heuristic")
    graph = run_search(start, width, height, config, budget=50, rollout_depth=10, rng=rng)
    assert is_solved(graph)  # a trivial 3-move puzzle should solve easily under real guidance
