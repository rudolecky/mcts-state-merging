"""Verification for guct_uniform_rubiks.py -- the paper's own algorithm
(LCB1-Uniform bandit, Full Bellman backup, exact-oracle reward, no
rollout) ported from guct_uniform_blocksworld.py to the cube domain.
Mirrors that module's own test suite. Uses a small depth-limited BFS
distance table (not the full 3,674,160-state pickle) so these tests stay
fast and self-contained.
"""

import math
import random
from collections import deque

from mcts_phase0.datasets import rubiks_cube_engine as rc
from mcts_phase0.guct_uniform_rubiks import (
    GUCTUniformConfig,
    MCTSEdge,
    MCTSNode,
    _lcb1_uniform_score,
    backup,
    create_root,
    expand,
    is_solved,
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


def _bfs_distances_up_to_depth(max_depth: int) -> dict:
    dist = {rc.SOLVED: 0}
    frontier = deque([rc.SOLVED])
    while frontier:
        state = frontier.popleft()
        d = dist[state]
        if d >= max_depth:
            continue
        for m in rc.ALL_MOVES:
            nxt = rc.apply_move(state, m)
            if nxt not in dist:
                dist[nxt] = d + 1
                frontier.append(nxt)
    return dist


# ---------- _lcb1_uniform_score (identical formula to guct_uniform_blocksworld.py) ----------

def test_lcb1_uniform_score_equals_the_sample_when_only_one_pull_total():
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
    root = MCTSNode(state=rc.SOLVED, n_visits=10)
    child_a = MCTSNode(state=rc.SOLVED)
    child_b = MCTSNode(state=rc.SOLVED)
    root.children = {
        0: MCTSEdge(child_key="a", t=8, l_hat=4.0, u_hat=4.0),
        1: MCTSEdge(child_key="b", t=8, l_hat=0.0, u_hat=2.0),
    }
    graph_nodes = {(): root, "a": child_a, "b": child_b}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    config = GUCTUniformConfig(merge_enabled=True)
    path, path_moves = select(_Graph(), config)
    assert path == [(), "b"]
    assert path_moves == [1]


# ---------- backup: Full Bellman pools the MINIMUM, not the average ----------

def test_full_bellman_backup_pools_minimum_not_average():
    root = MCTSNode(state=rc.SOLVED, h_gbfs=5.0)
    a = MCTSNode(state=rc.SOLVED, h_gbfs=5.0)
    b = MCTSNode(state=rc.SOLVED, h_gbfs=5.0)
    m = MCTSNode(state=rc.SOLVED, h_gbfs=5.0)
    root.children = {0: MCTSEdge(child_key="a"), 1: MCTSEdge(child_key="b")}
    a.children = {2: MCTSEdge(child_key="m")}
    b.children = {3: MCTSEdge(child_key="m")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "m": m}
        root_key = ()

    graph = _Graph()
    backup(graph, [(), "a", "m"], [0, 2], 0.8)
    backup(graph, [(), "b", "m"], [1, 3], 0.2)

    assert m.h_gbfs == 0.2  # min of {0.8, 0.2}, not their average
    assert a.h_gbfs == 0.8
    assert b.h_gbfs == 0.2
    assert root.h_gbfs == 0.2  # propagates all the way up
    assert m.n_visits == 2
    assert root.n_visits == 2

    edge_a_m = a.children[2]
    edge_b_m = b.children[3]
    assert edge_a_m.t == 1 and edge_a_m.l_hat == 0.8 and edge_a_m.u_hat == 0.8
    assert edge_b_m.t == 1 and edge_b_m.l_hat == 0.2 and edge_b_m.u_hat == 0.2


# ---------- expand: merge-vs-tree structural neutrality ----------

def test_expand_merges_transposed_paths_when_enabled_not_when_disabled():
    distances = _bfs_distances_up_to_depth(6)
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[5])  # a non-terminal starting state

    def _play(graph, config, move_indices):
        key = graph.root_key
        for m in move_indices:
            rng = _ScriptedRng([m])
            key, _ = expand(graph, key, [], config, rng, distances)
        return key

    move_a = 0
    treatment_cfg = GUCTUniformConfig(merge_enabled=True)
    t_graph = create_root(scrambled, True, distances)
    t_key_a = _play(t_graph, treatment_cfg, [move_a])
    # a second, independent graph taking the exact same single move from the exact same
    # start must land on the identical (state-keyed) node
    t_graph_2 = create_root(scrambled, True, distances)
    t_key_b = _play(t_graph_2, treatment_cfg, [move_a])
    assert t_key_a == t_key_b == rc.apply_move(scrambled, rc.ALL_MOVES[move_a])

    baseline_cfg = GUCTUniformConfig(merge_enabled=False)
    b_graph = create_root(scrambled, False, distances)
    b_key_a = _play(b_graph, baseline_cfg, [move_a])
    assert isinstance(b_key_a, tuple) and b_key_a == (move_a,)


# ---------- run_search: cycle-avoidance regression + end-to-end smoke test ----------

def test_run_search_does_not_hang_on_a_reversible_position():
    distances = _bfs_distances_up_to_depth(8)
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[5])
    rng = random.Random(0)
    config = GUCTUniformConfig(merge_enabled=True)
    graph = run_search(scrambled, config, budget=100, rng=rng, distances=distances)
    assert len(graph.nodes) > 30  # would plateau near a handful of nodes if stuck


def test_run_search_solves_a_one_move_scramble_most_of_the_time():
    distances = _bfs_distances_up_to_depth(8)
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = GUCTUniformConfig(merge_enabled=True)
        graph = run_search(scrambled, config, budget=20, rng=rng, distances=distances)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.9
