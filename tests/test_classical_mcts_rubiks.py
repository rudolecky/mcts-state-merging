"""Verification for classical_mcts_rubiks.py, independent of the (slow)
full BFS table -- pure search logic, no GPU/network dependency, mirrors
test_classical_mcts_puzzle.py's suite.
"""

import random

from mcts_phase0.classical_mcts_rubiks import (
    ClassicalMCTSConfig,
    MCTSEdge,
    MCTSNode,
    _make_node,
    _misplaced_corners_heuristic,
    _ucb1_score,
    _value_from_h,
    backup,
    create_root,
    expand,
    is_solved,
    run_random_search,
    run_search,
    select,
    simulate,
)
from mcts_phase0.datasets import rubiks_cube_engine as rc


class _ScriptedRng:
    def __init__(self, choices):
        self._choices = list(choices)

    def choice(self, seq):
        val = self._choices.pop(0)
        assert val in seq, f"scripted choice {val} not in {seq}"
        return val


# ---------- _ucb1_score / select ----------

def test_ucb1_gives_infinite_priority_to_untried_edges():
    parent = MCTSNode(state=rc.SOLVED, n_visits=10)
    child = MCTSNode(state=rc.SOLVED, n_visits=3, w_value=2.0)
    untried_edge = MCTSEdge(child_key=1, n_edge=0)
    assert _ucb1_score(parent, untried_edge, child, c=1.4) == float("inf")


def test_select_picks_expected_ucb1_argmax():
    root = MCTSNode(state=rc.SOLVED, n_visits=10)
    child_a = MCTSNode(state=rc.SOLVED, n_visits=8, w_value=6.0)  # Q=0.75
    child_b = MCTSNode(state=rc.SOLVED, n_visits=2, w_value=0.2)  # Q=0.1
    root.children = {0: MCTSEdge(child_key="a", n_edge=8), 1: MCTSEdge(child_key="b", n_edge=2)}
    graph = MCTSGraphStub({(): root, "a": child_a, "b": child_b}, ())
    config = ClassicalMCTSConfig(merge_enabled=True, c=0.1)
    path, path_moves = select(graph, config)
    assert path == [(), "a"]
    assert path_moves == (0,)


class MCTSGraphStub:
    def __init__(self, nodes, root_key):
        self.nodes = nodes
        self.root_key = root_key


# ---------- _make_node / heuristic ----------

def test_make_node_solved_is_terminal():
    node = _make_node(rc.SOLVED)
    assert node.is_terminal
    assert node.terminal_value == 1.0
    assert node.untried_moves == []


def test_make_node_scrambled_is_not_terminal():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    node = _make_node(scrambled)
    assert not node.is_terminal
    assert node.terminal_value is None
    assert len(node.untried_moves) == 12


def test_misplaced_corners_heuristic_zero_at_solved():
    assert _misplaced_corners_heuristic(rc.SOLVED) == 0


def test_misplaced_corners_heuristic_positive_after_one_move():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    assert _misplaced_corners_heuristic(scrambled) > 0


def test_value_from_h_is_one_at_zero_and_decreases():
    assert _value_from_h(0) == 1.0
    assert _value_from_h(1) < _value_from_h(0)
    assert _value_from_h(8) < _value_from_h(1)


# ---------- backup ----------

def test_backup_pools_at_shared_node():
    root = MCTSNode(state=rc.SOLVED)
    a = MCTSNode(state=rc.SOLVED)
    root.children = {0: MCTSEdge(child_key="a")}
    graph = MCTSGraphStub({(): root, "a": a}, ())
    backup(graph, [(), "a"], 1.0)
    assert a.n_visits == 1 and a.w_value == 1.0
    assert root.children[0].n_edge == 1


# ---------- simulate ----------

def test_simulate_returns_one_when_already_solved():
    assert simulate(rc.SOLVED, rollout_depth=5, rng=random.Random(0)) == 1.0


def test_simulate_can_solve_a_one_move_scramble_via_its_inverse():
    move = rc.ALL_MOVES[0]
    inv_idx = next(i for i, m in enumerate(rc.ALL_MOVES) if m.rotation == rc.inverse(move.rotation) and m.affected == move.affected)
    scrambled = rc.apply_move(rc.SOLVED, move)
    rng = _ScriptedRng([rc.ALL_MOVES[inv_idx]])
    assert simulate(scrambled, rollout_depth=1, rng=rng) == 1.0


# ---------- run_search / run_random_search (real, small scale) ----------

def test_run_search_solves_a_one_move_scramble_most_of_the_time():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = ClassicalMCTSConfig(merge_enabled=True)
        graph = run_search(scrambled, config, budget=20, rollout_depth=5, rng=rng)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7


def test_run_random_search_solves_a_one_move_scramble_most_of_the_time():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        if run_random_search(scrambled, budget=20, rollout_depth=5, rng=rng):
            solved_count += 1
    assert solved_count >= trials * 0.7


def test_run_search_heuristic_mode_never_calls_simulate(monkeypatch):
    import mcts_phase0.classical_mcts_rubiks as rubiks_mod

    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])

    def _boom(*a, **kw):
        raise AssertionError("simulate() should not be called in heuristic mode")

    monkeypatch.setattr(rubiks_mod, "simulate", _boom)

    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True, value_source="heuristic")
    graph = run_search(scrambled, config, budget=20, rollout_depth=5, rng=rng)
    assert is_solved(graph)


def _bfs_distances_up_to_depth(max_depth: int) -> dict:
    from collections import deque

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


def test_run_search_oracle_mode_uses_distances_table():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    # depth 8 gives generous margin over what budget=20 could possibly reach
    distances = _bfs_distances_up_to_depth(8)
    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True, value_source="oracle")
    graph = run_search(scrambled, config, budget=20, rollout_depth=5, rng=rng, distances=distances)
    assert is_solved(graph)


def test_run_search_stats_counts_new_nodes_and_merge_hits():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=False, value_source="rollout")
    stats = {}
    graph = run_search(scrambled, config, budget=20, rollout_depth=5, rng=rng, stats=stats)
    # tree mode (no merge) can never hit an existing state -- every expansion is a new node
    assert stats.get("new_nodes", 0) == 20
    assert stats.get("merge_hits", 0) == 0


def test_run_search_stats_merge_mode_can_record_merge_hits():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True, value_source="rollout")
    stats = {}
    run_search(scrambled, config, budget=20, rollout_depth=5, rng=rng, stats=stats)
    total = stats.get("new_nodes", 0) + stats.get("merge_hits", 0)
    assert total == 20
    assert stats.get("merge_hits", 0) > 0  # dense transpositions -- some hits expected even at budget=20


# ---------- merge_parent_cap ----------

def test_expand_falls_back_to_path_key_once_merge_target_is_full():
    import mcts_phase0.classical_mcts_rubiks as rubiks_mod

    graph = create_root(rc.SOLVED, merge_enabled=True)
    target_state = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    # pre-populate the merge target and mark it already at cap
    graph.nodes[target_state] = rubiks_mod._make_node(target_state)
    graph.nodes[target_state].parent_count = 1

    config = ClassicalMCTSConfig(merge_enabled=True, merge_parent_cap=1)
    rng = _ScriptedRng([0])  # force move index 0, which leads to target_state
    child_key = expand(graph, rc.SOLVED, path_moves=(), config=config, rng=rng)

    assert child_key == (0,)  # fell back to a fresh path-keyed node, not target_state
    assert (0,) in graph.nodes
    assert graph.nodes[(0,)].parent_count == 1
    assert graph.nodes[target_state].parent_count == 1  # unchanged -- capped node got no new parent


def test_expand_merges_normally_when_under_the_cap():
    import mcts_phase0.classical_mcts_rubiks as rubiks_mod

    graph = create_root(rc.SOLVED, merge_enabled=True)
    target_state = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    graph.nodes[target_state] = rubiks_mod._make_node(target_state)
    graph.nodes[target_state].parent_count = 0  # room under a cap of 1

    config = ClassicalMCTSConfig(merge_enabled=True, merge_parent_cap=1)
    rng = _ScriptedRng([0])
    child_key = expand(graph, rc.SOLVED, path_moves=(), config=config, rng=rng)

    assert child_key == target_state  # merged normally, still under the cap
    assert graph.nodes[target_state].parent_count == 1


def test_merge_parent_cap_none_is_uncapped_merging():
    graph = create_root(rc.SOLVED, merge_enabled=True)
    config = ClassicalMCTSConfig(merge_enabled=True, merge_parent_cap=None)
    rng = _ScriptedRng([0])
    child_key = expand(graph, rc.SOLVED, path_moves=(), config=config, rng=rng)
    assert child_key == rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])


# ---------- merge_visit_cap ----------

def test_expand_falls_back_to_path_key_once_merge_target_is_visit_capped():
    import mcts_phase0.classical_mcts_rubiks as rubiks_mod

    graph = create_root(rc.SOLVED, merge_enabled=True)
    target_state = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    graph.nodes[target_state] = rubiks_mod._make_node(target_state)
    graph.nodes[target_state].n_visits = 5  # already at/over the cap

    config = ClassicalMCTSConfig(merge_enabled=True, merge_visit_cap=5)
    rng = _ScriptedRng([0])
    child_key = expand(graph, rc.SOLVED, path_moves=(), config=config, rng=rng)

    assert child_key == (0,)  # fell back -- target's n_visits already at the cap
    assert (0,) in graph.nodes


def test_expand_merges_normally_when_under_the_visit_cap():
    import mcts_phase0.classical_mcts_rubiks as rubiks_mod

    graph = create_root(rc.SOLVED, merge_enabled=True)
    target_state = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    graph.nodes[target_state] = rubiks_mod._make_node(target_state)
    graph.nodes[target_state].n_visits = 4  # under a cap of 5

    config = ClassicalMCTSConfig(merge_enabled=True, merge_visit_cap=5)
    rng = _ScriptedRng([0])
    child_key = expand(graph, rc.SOLVED, path_moves=(), config=config, rng=rng)

    assert child_key == target_state  # merged normally, still under the cap
