"""Verification for guct_uniform_sokoban.py -- mirrors
test_guct_uniform_puzzle.py's own suite. Sokoban's full bfs_distances()
(2,268 states) is cheap enough to compute directly in tests.
"""

import math
import random

from mcts_phase0.datasets.sokoban_engine import apply_move, bfs_distances, make_state
from mcts_phase0.guct_uniform_sokoban import (
    GUCTUniformConfig,
    MCTSEdge,
    MCTSNode,
    _lcb1_uniform_score,
    _make_node,
    _UNSOLVABLE_H,
    backup,
    create_root,
    expand,
    is_solved,
    run_search,
    select,
)

_DISTANCES = bfs_distances()


class _ScriptedRng:
    def __init__(self, choices):
        self._choices = list(choices)

    def choice(self, seq):
        val = self._choices.pop(0)
        assert val in seq, f"scripted choice {val} not in {seq}"
        return val


def test_lcb1_uniform_score_matches_hand_computed_value():
    edge = MCTSEdge(child_key="x", t=1, l_hat=2.0, u_hat=6.0)
    expected = (6.0 + 2.0) / 2.0 - (6.0 - 2.0) * math.sqrt(6.0 * 1 * math.log(4))
    assert _lcb1_uniform_score(edge, T=4) == expected


def test_untried_edge_gets_negative_infinity_priority():
    assert _lcb1_uniform_score(MCTSEdge(child_key="x", t=0), T=10) == float("-inf")


def test_select_picks_expected_lcb1_uniform_argmin():
    root = MCTSNode(state=(), n_visits=10)
    child_a = MCTSNode(state=())
    child_b = MCTSNode(state=())
    root.children = {
        (0, 1): MCTSEdge(child_key="a", t=8, l_hat=4.0, u_hat=4.0),
        (1, 0): MCTSEdge(child_key="b", t=8, l_hat=0.0, u_hat=2.0),
    }

    class _Graph:
        nodes = {(): root, "a": child_a, "b": child_b}
        root_key = ()

    config = GUCTUniformConfig(merge_enabled=True)
    path, path_moves = select(_Graph(), config)
    assert path == [(), "b"]
    assert path_moves == [(1, 0)]


def test_full_bellman_backup_pools_minimum_not_average():
    root = MCTSNode(state=(), h_gbfs=5.0)
    a = MCTSNode(state=(), h_gbfs=5.0)
    b = MCTSNode(state=(), h_gbfs=5.0)
    m = MCTSNode(state=(), h_gbfs=5.0)
    root.children = {(0, 1): MCTSEdge(child_key="a"), (1, 0): MCTSEdge(child_key="b")}
    a.children = {(0, -1): MCTSEdge(child_key="m")}
    b.children = {(-1, 0): MCTSEdge(child_key="m")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "m": m}
        root_key = ()

    graph = _Graph()
    backup(graph, [(), "a", "m"], [(0, 1), (0, -1)], 0.8)
    backup(graph, [(), "b", "m"], [(1, 0), (-1, 0)], 0.2)

    assert m.h_gbfs == 0.2
    assert a.h_gbfs == 0.8
    assert b.h_gbfs == 0.2
    assert root.h_gbfs == 0.2
    assert m.n_visits == 2


def test_expand_merges_when_enabled_not_when_disabled():
    state = make_state(52, (53,))

    treatment_cfg = GUCTUniformConfig(merge_enabled=True)
    t_graph = create_root(state, True, _DISTANCES)
    move0 = t_graph.nodes[t_graph.root_key].untried_moves[0]
    rng = _ScriptedRng([move0])
    key_a, move = expand(t_graph, t_graph.root_key, [], treatment_cfg, rng, _DISTANCES)
    assert key_a == apply_move(state, move)

    baseline_cfg = GUCTUniformConfig(merge_enabled=False)
    b_graph = create_root(state, False, _DISTANCES)
    rng2 = _ScriptedRng([move])
    key_b, _ = expand(b_graph, b_graph.root_key, [], baseline_cfg, rng2, _DISTANCES)
    assert key_b == (move,)


def test_make_node_uses_sentinel_for_a_state_absent_from_distances():
    # a genuine deadlock (or any state simply not in the table) must not crash --
    # it should be treated as maximally bad, not looked up and KeyError'd.
    deadlocked_state = make_state(0, (1,))
    node = _make_node(deadlocked_state, distances={})
    assert node.own_h == _UNSOLVABLE_H


def test_run_search_solves_a_two_push_puzzle_reliably():
    state = make_state(52, (53,))  # 2 pushes needed
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = GUCTUniformConfig(merge_enabled=True)
        graph = run_search(state, config, budget=30, rng=rng, distances=_DISTANCES)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.9
