"""Verification for guct_uniform_arc.py -- mirrors the other GUCT-Uniform
ports' own test suites, plus hand-computed checks for the new
_arc_heuristic (this domain has no pre-existing heuristic anywhere else in
the project to reuse). Uses the same real, easy ARC task
(`68b16354`, O = hmirror(I)) test_classical_mcts_arc_program.py already
established for end-to-end smoke tests.
"""

import json
import math
import random

import mcts_phase0.guct_uniform_arc as m
from mcts_phase0.datasets import arc_engine as ae
from mcts_phase0.datasets import arc_program_engine as engine


def _load_real_task(task_id):
    raw = json.load(open(f"data/arc_agi/tasks/{task_id}.json"))
    return ae.load_task(raw, task_id)


# ---------- _grid_overlap_fraction / _arc_heuristic ----------

def test_grid_overlap_fraction_full_match():
    grid = ((1, 2), (3, 4))
    assert m._grid_overlap_fraction(grid, grid) == 1.0


def test_grid_overlap_fraction_partial_match():
    candidate = ((1, 2), (3, 0))
    target = ((1, 2), (3, 4))
    assert m._grid_overlap_fraction(candidate, target) == 3 / 4


def test_grid_overlap_fraction_shape_mismatch_is_zero():
    candidate = ((1, 2, 3),)
    target = ((1, 2), (3, 4))
    assert m._grid_overlap_fraction(candidate, target) == 0.0


def test_grid_overlap_fraction_jagged_non_grid_tuple_does_not_crash():
    # real bug caught during calibration: row 0 is a nested tuple (passes a
    # naive "is candidate[0] a tuple" check) but row 1 is a bare int -- some
    # other DSL tuple type, not a real Grid. Must return 0.0, not crash.
    candidate = ((1, 2), 3)
    target = ((1, 2), (3, 4))
    assert m._grid_overlap_fraction(candidate, target) == 0.0


def test_grid_overlap_fraction_non_grid_value_is_zero():
    assert m._grid_overlap_fraction(frozenset({(1, 1)}), ((1, 2), (3, 4))) == 0.0


def test_arc_heuristic_zero_when_every_context_ends_on_its_target():
    target = ((1, 2), (3, 4))
    state = engine.ProgramState(contexts=((target,),), type_schema=(frozenset({"GRID"}),))
    assert m._arc_heuristic(state, (target,)) == 0.0


def test_arc_heuristic_positive_when_not_matching():
    target = ((1, 2), (3, 4))
    wrong = ((0, 0), (0, 0))
    state = engine.ProgramState(contexts=((wrong,),), type_schema=(frozenset({"GRID"}),))
    assert m._arc_heuristic(state, (target,)) == 1.0  # 0/4 matching cells


def test_arc_heuristic_matches_is_goal_at_the_boundary():
    # h == 0.0 must imply is_goal, since own_h's 0.0-vs-nonzero split is what
    # run_search uses to decide is_terminal. The converse does not hold any
    # more: is_goal now matches the target at ANY context index (it has to,
    # or goal-ness would depend on information canonical_key cannot see), so
    # a solved state can carry h > 0.0 here -- see
    # test_make_node_forces_zero_h_on_a_goal_at_a_non_final_index.
    target = ((1, 2), (3, 4))
    solved_state = engine.ProgramState(contexts=((target,),), type_schema=(frozenset({"GRID"}),))
    unsolved_state = engine.ProgramState(contexts=(((9, 9), (9, 9)),), type_schema=(frozenset({"GRID"}),))
    assert engine.is_goal(solved_state, (target,)) is True
    assert m._arc_heuristic(solved_state, (target,)) == 0.0
    assert engine.is_goal(unsolved_state, (target,)) is False
    assert m._arc_heuristic(unsolved_state, (target,)) > 0.0


# ---------- _lcb1_uniform_score / backup (identical formulas to the other ports) ----------

def test_lcb1_uniform_score_matches_hand_computed_value():
    edge = m.MCTSEdge(child_key="x", t=1, l_hat=2.0, u_hat=6.0)
    expected = (6.0 + 2.0) / 2.0 - (6.0 - 2.0) * math.sqrt(6.0 * 1 * math.log(4))
    assert m._lcb1_uniform_score(edge, T=4) == expected


def test_full_bellman_backup_pools_minimum_not_average():
    root = m.MCTSNode(program_state=None, h_gbfs=5.0)
    a = m.MCTSNode(program_state=None, h_gbfs=5.0)
    b = m.MCTSNode(program_state=None, h_gbfs=5.0)
    shared = m.MCTSNode(program_state=None, h_gbfs=5.0)
    root.children = {"mvA": m.MCTSEdge(child_key="a"), "mvB": m.MCTSEdge(child_key="b")}
    a.children = {"mv1": m.MCTSEdge(child_key="shared")}
    b.children = {"mv2": m.MCTSEdge(child_key="shared")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "shared": shared}
        root_key = ()

    graph = _Graph()
    m.backup(graph, [(), "a", "shared"], ["mvA", "mv1"], 0.8)
    m.backup(graph, [(), "b", "shared"], ["mvB", "mv2"], 0.2)
    assert shared.h_gbfs == 0.2  # min, not average
    assert root.h_gbfs == 0.2


# ---------- run_search: real end-to-end smoke test ----------

def test_run_search_solves_a_real_easy_task_most_of_the_time():
    task = _load_real_task("68b16354")  # solve_68b16354(I): O = hmirror(I)
    solved_count = 0
    trials = 15
    for seed in range(trials):
        rng = random.Random(seed)
        config = m.GUCTUniformConfig(merge_enabled=True)
        graph = m.run_search(task.train_inputs, task.train_outputs, config, budget=200, rng=rng)
        if m.is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.5


def test_merge_mode_never_creates_more_nodes_than_tree_mode_for_the_same_budget():
    task = _load_real_task("68b16354")
    rng = random.Random(0)
    tree_graph = m.run_search(task.train_inputs, task.train_outputs, m.GUCTUniformConfig(merge_enabled=False), budget=150, rng=rng)
    rng = random.Random(0)
    merge_graph = m.run_search(task.train_inputs, task.train_outputs, m.GUCTUniformConfig(merge_enabled=True), budget=150, rng=rng)
    assert len(merge_graph.nodes) <= len(tree_graph.nodes)


def test_run_search_stats_counts_new_nodes_and_merge_hits():
    task = _load_real_task("68b16354")
    rng = random.Random(0)
    config = m.GUCTUniformConfig(merge_enabled=True)
    stats = {}
    m.run_search(task.train_inputs, task.train_outputs, config, budget=150, rng=rng, stats=stats)
    total = stats.get("new_nodes", 0) + stats.get("merge_hits", 0)
    assert total <= 150  # <= only if the whole reachable tree is exhausted first
    assert total > 0
    assert stats.get("new_nodes", 0) > 0


def test_arc_heuristic_is_zero_exactly_when_is_goal():
    """Both directions, restored: h reads the best context index, so it is
    0.0 iff some index matches every target, which is exactly `is_goal`."""
    import random
    target = ((1, 2), (3, 4))
    # Target present, but not as the most recent value.
    goal_not_last = engine.ProgramState(
        contexts=((target, 7),),
        type_schema=(frozenset({"GRID"}), frozenset({"INTEGER"})),
    )
    assert engine.is_goal(goal_not_last, (target,)) is True
    assert m._arc_heuristic(goal_not_last, (target,)) == 0.0
    node = m._make_node(goal_not_last, (target,), m.GUCTUniformConfig(merge_enabled=True), random.Random(0))
    assert node.is_terminal is True
    assert node.own_h == 0.0

    unsolved = engine.ProgramState(contexts=((((9, 9), (9, 9)), 7),),
                                   type_schema=(frozenset({"GRID"}), frozenset({"INTEGER"})))
    assert engine.is_goal(unsolved, (target,)) is False
    assert m._arc_heuristic(unsolved, (target,)) > 0.0


def test_arc_heuristic_is_determined_by_the_merge_key():
    """The asymmetry this fixes: two orderings sharing a merge key must get
    the same h, or the merge arm (which keeps whichever ordering arrived
    first) and the tree arm score different things -- and GUCT-Uniform is
    entirely h-driven, so that difference is not washed out by rollouts."""
    target = ((1, 2), (3, 4))
    near = ((1, 2), (3, 9))
    a = engine.ProgramState(contexts=((near, 7),),
                            type_schema=(frozenset({"GRID"}), frozenset({"INTEGER"})))
    b = engine.ProgramState(contexts=((7, near),),
                            type_schema=(frozenset({"INTEGER"}), frozenset({"GRID"})))
    assert engine.canonical_key(a) == engine.canonical_key(b)
    assert m._arc_heuristic(a, (target,)) == m._arc_heuristic(b, (target,))
    assert 0.0 < m._arc_heuristic(a, (target,)) < 1.0


def test_a_failed_expansion_at_a_root_with_live_children_does_not_end_the_search(monkeypatch):
    """expand() returns None when every remaining untried move fails at
    execution. run_search used to `break` whenever that happened at the root
    -- even though the root still had live children to descend into -- and,
    below the root, to close the incoming edge unconditionally, pruning the
    leaf's existing subtree. An audit measured it stopping 23/120 tree and
    16/120 merge runs after ~15-24 nodes at budget 500. Forced here: the
    second expansion (at the root, which by then has one child) fails."""
    task = _load_real_task("68b16354")
    real = m.expand
    calls = {"n": 0}

    def flaky(graph, leaf_key, target_outputs, config, rng):
        calls["n"] += 1
        if calls["n"] == 2:
            assert leaf_key == graph.root_key and graph.nodes[leaf_key].children, "fixture premise"
            graph.nodes[leaf_key].untried_moves.clear()  # every remaining move "failed"
            return None
        return real(graph, leaf_key, target_outputs, config, rng)

    monkeypatch.setattr(m, "expand", flaky)
    for merge in (False, True):
        calls["n"] = 0
        stats = {}
        m.run_search(task.train_inputs, task.train_outputs, m.GUCTUniformConfig(merge_enabled=merge),
                     budget=20, rng=random.Random(0), stats=stats)
        assert stats.get("new_nodes", 0) + stats.get("merge_hits", 0) == 20, f"merge={merge} stopped early"
