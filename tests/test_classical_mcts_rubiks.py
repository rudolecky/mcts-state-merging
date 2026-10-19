"""Verification for classical_mcts_rubiks.py, independent of the (slow)
full BFS table -- pure search logic, no GPU/network dependency, mirrors
test_classical_mcts_puzzle.py's suite.
"""

import functools
import random

import pytest

from mcts_phase0.classical_mcts_rubiks import (
    ClassicalMCTSConfig,
    MCTSEdge,
    MCTSGraph,
    MCTSNode,
    _make_node,
    approx_match,
    relation_value,
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


# ---------- reject-duplicates arm (admission filter, no statistic pooling) ----------

def test_reject_duplicates_keeps_states_unique_and_never_pools():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    config = ClassicalMCTSConfig(merge_enabled=False, reject_duplicates=True)
    graph = run_search(scrambled, config, budget=200, rollout_depth=5, rng=random.Random(0))

    assert graph.rejected > 0  # the cube is the densest transposition domain here
    in_degree = {}
    for node in graph.nodes.values():
        for edge in node.children.values():
            in_degree[edge.child_key] = in_degree.get(edge.child_key, 0) + 1
    assert all(d == 1 for d in in_degree.values())
    assert all(n.parent_count <= 1 for n in graph.nodes.values())

    states = [n.state for n in graph.nodes.values()]
    assert len(set(states)) == len(states)


def test_reject_duplicates_expansion_can_fail_without_crashing_the_search():
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    config = ClassicalMCTSConfig(merge_enabled=False, reject_duplicates=True)
    graph = create_root(scrambled, merge_enabled=False, reject_duplicates=True)
    root = graph.nodes[graph.root_key]
    for move in list(root.untried_moves):
        graph.seen_states.add(rc.apply_move(root.state, rc.ALL_MOVES[move]))
    assert expand(graph, graph.root_key, (), config, random.Random(0)) is None
    assert root.untried_moves == []


def test_reject_duplicates_solves_a_one_move_scramble_where_merging_is_catastrophic():
    """Merging drops the cube's solve rate to exactly zero (see
    RUBIKS_CUBE_FINDINGS.md). This arm gets merging's distinctness without
    its statistic pooling, so it should behave like the tree arm."""
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    solved_count = 0
    trials = 20
    for seed in range(trials):
        config = ClassicalMCTSConfig(merge_enabled=False, reject_duplicates=True)
        graph = run_search(scrambled, config, budget=20, rollout_depth=5, rng=random.Random(seed))
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7


# ---------- approximate equivalence relation (key_tau) ----------

class _TotalDistances(dict):
    """Exact distances for the shallow region, and a finite stand-in beyond.

    The real experiments pass the complete 3.67M-state table. A unit test
    cannot afford to build that, and a truncated dict raises KeyError
    mid-run once the search leaves it -- which it does, because the filter
    arm goes deeper than the others. Production keeps the strict dict (a
    genuine gap should be loud); these tests supply a total mapping so they
    exercise the relation rather than the table's edge."""

    def __init__(self, exact, beyond):
        super().__init__(exact)
        self._beyond = beyond

    def __missing__(self, key):
        return self._beyond


@functools.lru_cache(maxsize=1)
def _tiny_distances():
    """Cached: the BFS build takes ~6.5s and every test below needs it."""
    exact = _bfs_distances_up_to_depth(8)
    return _TotalDistances(exact, beyond=9)


def test_key_tau_none_is_the_exact_relation():
    """The default must remain exact state identity: no depth index built,
    no matches counted."""
    distances = _tiny_distances()
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    graph = run_search(scrambled, ClassicalMCTSConfig(merge_enabled=True), 30, 5,
                       random.Random(0), distances=distances)
    assert graph.depth_index == {}
    assert graph.total_matches == 0 and graph.false_matches == 0


def test_approx_match_is_nearest_within_tau_at_the_same_depth():
    """Mirrors search.py.find_merge_target: strict <, nearest wins, and the
    depth bucket is never crossed."""
    graph = MCTSGraph(nodes={}, root_key=())
    for key in ("s_far", "s_near", "s_mid", "other_depth"):
        graph.nodes[key] = MCTSNode(state=key)  # the index holds node keys now
    graph.depth_index = {1: [("s_far", 10), ("s_near", 4), ("s_mid", 6)],
                         2: [("other_depth", 5)]}
    # Nearest must win, not first-within-tau. At value 6 with tau 3 the first
    # candidate inside tau is s_near (delta 2) while the nearest is s_mid
    # (delta 0) -- the one assertion that separates the two rules.
    assert approx_match(graph, 1, 6, tau=3.0) == "s_mid"
    assert approx_match(graph, 1, 5, tau=1.5) == "s_near"   # tie on delta -> first seen
    assert approx_match(graph, 1, 6, tau=0.5) == "s_mid"    # exact hit
    assert approx_match(graph, 1, 8, tau=1.0) is None       # nothing within tau
    assert approx_match(graph, 3, 5, tau=99) is None        # empty bucket, no cross-depth match
    assert approx_match(graph, 1, 10, tau=0.0) is None      # strict <, so tau=0 never matches


def test_approximate_relation_produces_measurable_false_matches():
    """The reason to run an approximate relation on a domain that does not
    need one: ground truth makes the false-merge rate a measurement."""
    distances = _tiny_distances()
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    graph = run_search(scrambled, ClassicalMCTSConfig(merge_enabled=True, key_tau=2.0),
                       40, 5, random.Random(0), distances=distances)
    assert graph.total_matches > 0
    assert graph.false_matches > 0  # a 1-D value cannot separate cube states
    assert graph.false_matches <= graph.total_matches


def test_approximate_filter_still_never_pools():
    """The structural guarantee has to survive the relation being wrong:
    with an approximate key the filter must still leave a tree."""
    distances = _tiny_distances()
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    graph = run_search(scrambled,
                       ClassicalMCTSConfig(merge_enabled=False, reject_duplicates=True, key_tau=2.0),
                       40, 5, random.Random(0), distances=distances)
    assert graph.rejected > 0
    assert graph.total_matches > 0, "the approximate relation never fired -- test is vacuous"
    assert graph.false_matches > 0, "no false matches -- relation is behaving as an exact one"
    in_degree = {}
    for node in graph.nodes.values():
        for edge in node.children.values():
            in_degree[edge.child_key] = in_degree.get(edge.child_key, 0) + 1
    assert all(d == 1 for d in in_degree.values())
    assert all(n.parent_count <= 1 for n in graph.nodes.values())


def test_approximate_merge_pools_into_a_genuinely_different_state():
    """The failure mode being modelled: under an approximate relation the
    merge target is usually NOT the state that was just generated, so a
    node ends up holding statistics from paths that never reached it."""
    distances = _tiny_distances()
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    graph = run_search(scrambled, ClassicalMCTSConfig(merge_enabled=True, key_tau=2.0),
                       40, 5, random.Random(0), distances=distances)
    # Assert what the name claims: some edge leads to a node holding a
    # DIFFERENT state than the move actually produced. Zero by construction
    # under an exact relation, so this cannot pass vacuously.
    wrong_target = 0
    for node in graph.nodes.values():
        for move, edge in node.children.items():
            produced = rc.apply_move(node.state, rc.ALL_MOVES[move])
            if produced != graph.nodes[edge.child_key].state:
                wrong_target += 1
    assert wrong_target > 0, "every edge landed on the state its move produced -- relation was exact"
    assert graph.false_matches > 0


def test_approx_match_skips_exhausted_candidates():
    """`search.py` skips dead candidates for a stated reason: matching one
    leaves a state already proven to go nowhere as the sole representative
    of its class, which the filter arm then destroys candidates in favour
    of."""
    graph = MCTSGraph(nodes={}, root_key=())
    for key in ("dead", "live"):
        graph.nodes[key] = MCTSNode(state=key)
    graph.depth_index = {1: [("dead", 5.0), ("live", 5.4)]}
    assert approx_match(graph, 1, 5.0, tau=1.0) == "dead"  # nearest while alive
    graph.nodes["dead"].exhausted = True
    assert approx_match(graph, 1, 5.0, tau=1.0) == "live"  # skipped once dead


def test_key_tau_refuses_the_cap_knobs_it_bypasses():
    import pytest
    for caps in ({"merge_parent_cap": 1}, {"merge_visit_cap": 2}):
        with pytest.raises(ValueError):
            ClassicalMCTSConfig(merge_enabled=True, key_tau=2.0, **caps)


def test_the_goal_is_spared_by_default():
    """Matching `search.py`'s reject_answer_nodes=False. Without it a
    measured collapse is the relation eating the answer, not the search."""
    distances = _tiny_distances()
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    for kwargs in ({"merge_enabled": True}, {"merge_enabled": False, "reject_duplicates": True}):
        graph = run_search(scrambled,
                           ClassicalMCTSConfig(key_tau=4.0, key_noise=1.0, **kwargs),
                           60, 5, random.Random(0), distances=distances)
        states = [n.state for n in graph.nodes.values()]
        assert rc.SOLVED in states, f"goal was swallowed by the relation ({kwargs})"


def test_relation_value_is_stable_across_processes():
    """`hash()` is randomized per process; this project already lost a set of
    results to exactly that."""
    import subprocess
    import sys

    distances = _tiny_distances()
    state = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[3])
    local = relation_value(state, distances, noise=1.0)
    code = (
        "from mcts_phase0.classical_mcts_rubiks import relation_value;"
        "from mcts_phase0.datasets import rubiks_cube_engine as rc;"
        "s = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[3]);"
        "print(repr(relation_value(s, {s: 1}, noise=1.0)))"
    )
    for seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"})
        assert out.returncode == 0, out.stderr
        assert abs(float(out.stdout) - (1.0 + (local - distances[state]))) < 1e-12


def test_key_noise_actually_perturbs_the_relation_value():
    """Confound #1 of the discarded first experiment: with integer oracle
    distances every gap inside a depth bucket is even, so tau collapses to a
    three-position dial whose first band is *transitive* -- exact identity on
    a coarsened space, not an approximate relation. `key_noise` is what makes
    the value continuous. Nothing else in this suite fails if the noise term
    is dropped from `relation_value`, so this is that test."""
    distances = _tiny_distances()
    states = [rc.SOLVED] + [rc.apply_move(rc.SOLVED, m) for m in rc.ALL_MOVES[:6]]

    for state in states:
        assert relation_value(state, distances, noise=0.0) == float(distances[state])

    offsets = [relation_value(s, distances, noise=1.0) - distances[s] for s in states]
    assert all(abs(o) <= 1.0 for o in offsets), "noise must stay inside +/- key_noise"
    assert any(o != 0.0 for o in offsets), "noise term is not being applied at all"
    # The point of the noise is to break the parity, so the perturbed values
    # must not all land back on the integer grid the confound came from.
    assert any(abs(o) > 1e-9 and abs(o - round(o)) > 1e-9 for o in offsets)
    # Distinct states must get distinct offsets, or the relation is still a
    # coarsening rather than a noisy estimate.
    assert len(set(offsets)) > 1, "every state got the same offset"
    # Scaling key_noise must scale the perturbation: pins it as a magnitude,
    # not a flag.
    for state in states:
        o1 = relation_value(state, distances, noise=1.0) - distances[state]
        o2 = relation_value(state, distances, noise=4.0) - distances[state]
        assert abs(o2 - 4.0 * o1) < 1e-12


def test_charge_rejections_moves_the_budget_a_rejection_costs():
    """The whole comparative claim for the filter arm rests on this flag: a
    rejected move never becomes an expansion but does cost a real
    `apply_move`, so leaving it uncharged hands the arm extra compute. Had no
    coverage here, on the module whose published result depends on it."""
    distances = _tiny_distances()
    scrambled = rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0])
    budget = 40

    def run(charge):
        return run_search(scrambled,
                          ClassicalMCTSConfig(merge_enabled=False, reject_duplicates=True,
                                              charge_rejections=charge),
                          budget, 5, random.Random(0), distances=distances)

    free, charged = run(False), run(True)
    assert free.rejected > 0, "fixture produced no rejections; it cannot test the flag"
    # Charging spends budget on rejections, so strictly less of it reaches
    # expansions -- the arm must end up smaller.
    assert len(charged.nodes) < len(free.nodes)
    # And the charged arm's real work stays within budget, modulo the known
    # overshoot of at most one expand() call's worth of rejections.
    assert len(charged.nodes) - 1 + charged.rejected <= budget + len(rc.ALL_MOVES)


def test_expand_noise_zero_proposes_a_distance_reducing_move_first():
    """expand_noise=0.0 is a perfect proposal policy: the first child
    expanded from any non-goal state must be one step closer to solved.
    The default (None) must stay uniformly random -- covered by the no-op
    equivalence check -- so this pins only the new branch."""
    distances = _tiny_distances()
    start = rc.apply_move(rc.apply_move(rc.SOLVED, rc.ALL_MOVES[0]), rc.ALL_MOVES[4])
    d0 = distances[start]
    assert d0 == 2, "fixture must be two moves from solved"
    for seed in range(5):
        graph = create_root(start, False, False)
        cfg = ClassicalMCTSConfig(merge_enabled=False, value_source="oracle", expand_noise=0.0)
        child_key = expand(graph, graph.root_key, (), cfg, random.Random(seed), distances)
        assert distances[graph.nodes[child_key].state] == d0 - 1


def test_expand_noise_requires_distances():
    with pytest.raises(ValueError, match="expand_noise"):
        run_search(rc.SOLVED, ClassicalMCTSConfig(merge_enabled=False, expand_noise=0.5),
                   5, 3, random.Random(0), distances=None)


def test_key_tau_is_refused_on_a_tree_arm():
    """Without merge or reject, expand's key_tau branch pooled into matches,
    so a 'tree' arm silently became a DAG (found by audit: 10 nodes and 91
    matches where a real tree had 96 nodes)."""
    with pytest.raises(ValueError, match="key_tau needs"):
        ClassicalMCTSConfig(merge_enabled=False, key_tau=1.0)
