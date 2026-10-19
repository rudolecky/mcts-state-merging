"""Verification for classical_mcts_arc_program.py: UCB1 argmax, backup
pooling (Monte Carlo average, same convention as classical_mcts_blocksworld.py),
the safe_apply_move robustness fallback (legal_moves type-checks but doesn't
verify arity/shape against how a closure-consumer will actually call a
bound function -- real execution catches what the type tags miss), and an
end-to-end solve on a real, easy task using both conditions.
"""

import json
import random

import mcts_phase0.classical_mcts_arc_program as m
from mcts_phase0.datasets import arc_engine as ae
from mcts_phase0.datasets import arc_program_engine as engine


def _load_real_task(task_id):
    raw = json.load(open(f"data/arc_agi/tasks/{task_id}.json"))
    return ae.load_task(raw, task_id)


# ---------- UCB1 / backup, hand-built ----------

def test_ucb1_score_gives_infinite_priority_to_untried_edges():
    parent = m.MCTSNode(program_state=None, n_visits=10)
    child = m.MCTSNode(program_state=None, n_visits=3, w_value=2.0)
    edge = m.MCTSEdge(child_key="c", n_edge=0)
    assert m._ucb1_score(parent, edge, child, c=1.4) == float("inf")


def test_backup_pools_the_average_at_a_shared_node():
    root = m.MCTSNode(program_state=None)
    a = m.MCTSNode(program_state=None)
    b = m.MCTSNode(program_state=None)
    shared = m.MCTSNode(program_state=None)
    root.children = {"mvA": m.MCTSEdge(child_key="a"), "mvB": m.MCTSEdge(child_key="b")}
    a.children = {"mv1": m.MCTSEdge(child_key="shared")}
    b.children = {"mv2": m.MCTSEdge(child_key="shared")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "shared": shared}
        root_key = ()

    graph = _Graph()
    m.backup(graph, [(), "a", "shared"], ["mvA", "mv1"], 1.0)
    m.backup(graph, [(), "b", "shared"], ["mvB", "mv2"], 0.0)
    assert shared.n_visits == 2
    assert shared.w_value == 1.0  # mean 0.5 -- pooled evidence from both paths


# ---------- safe_apply_move: robustness against arity/shape mismatches legal_moves can't catch ----------

def test_safe_apply_move_returns_none_on_a_real_execution_failure():
    state = engine.create_initial_state((((1, 2), (3, 4)),))
    # argmin's compfunc must take exactly one argument; legal_moves' type
    # tags don't check arity, so a wrong-arity function like sizefilter (it
    # needs a container AND a size) can be offered here and fail for real --
    # exactly the gap safe_apply_move exists to catch (confirmed to raise a
    # real TypeError when called directly against vendored dsl.py).
    bad_move = ("argmin", (("ctx", 0), ("fnref", "sizefilter")))
    assert m.safe_apply_move(state, bad_move) is None


def test_safe_apply_move_returns_a_state_on_success():
    state = engine.create_initial_state((((1, 2), (3, 4)),))
    good_move = ("hmirror", (("ctx", 0),))
    result = m.safe_apply_move(state, good_move)
    assert result is not None
    assert len(result.contexts[0]) == 2


# ---------- end-to-end: a real, easy task (single hmirror) ----------

def test_run_search_solves_a_real_easy_task_most_of_the_time():
    task = _load_real_task("68b16354")  # solve_68b16354(I): O = hmirror(I)
    solved_count = 0
    trials = 15
    for seed in range(trials):
        rng = random.Random(seed)
        config = m.ClassicalMCTSConfig(merge_enabled=True)
        graph = m.run_search(task.train_inputs, task.train_outputs, config, budget=200, rollout_depth=5, rng=rng)
        if m.is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.6


def test_run_random_search_solves_a_real_easy_task_most_of_the_time():
    task = _load_real_task("68b16354")  # solve_68b16354(I): O = hmirror(I)
    solved_count = 0
    trials = 15
    for seed in range(trials):
        rng = random.Random(seed)
        if m.run_random_search(task.train_inputs, task.train_outputs, budget=200, rollout_depth=5, rng=rng):
            solved_count += 1
    assert solved_count >= trials * 0.6


def test_merge_mode_never_creates_more_nodes_than_tree_mode_for_the_same_budget():
    task = _load_real_task("68b16354")
    rng = random.Random(0)
    tree_graph = m.run_search(task.train_inputs, task.train_outputs, m.ClassicalMCTSConfig(merge_enabled=False), budget=150, rollout_depth=5, rng=rng)
    rng = random.Random(0)
    merge_graph = m.run_search(task.train_inputs, task.train_outputs, m.ClassicalMCTSConfig(merge_enabled=True), budget=150, rollout_depth=5, rng=rng)
    assert len(merge_graph.nodes) <= len(tree_graph.nodes)


# ---------- article-derived items: goal/key consistency, graded reward, reject arm ----------

def test_make_node_marks_terminal_for_a_goal_at_a_non_final_index():
    """`_make_node` caches `is_terminal` once per merge key, so goal-ness
    has to be a function of that key -- the old `ctx[-1]` test was not.
    The states that would expose it are unreachable in the current search
    (see `engine.goal_index`), so this pins an invariant rather than
    recovering lost solves."""
    task = _load_real_task("68b16354")
    config = m.ClassicalMCTSConfig(merge_enabled=True)
    rng = random.Random(0)
    state = engine.create_initial_state(task.train_inputs)
    mirrored = engine.apply_move(state, ("hmirror", (("ctx", 0),)))
    kept_going = engine.apply_move(mirrored, ("width", (("ctx", 0),)))

    assert engine.canonical_key(kept_going) != engine.canonical_key(mirrored)
    assert m._make_node(mirrored, task.train_outputs, config, rng).is_terminal is True
    # target is at index 1, not last -- still terminal.
    assert m._make_node(kept_going, task.train_outputs, config, rng).is_terminal is True


def test_same_merge_key_states_agree_on_terminality():
    task = _load_real_task("68b16354")
    config = m.ClassicalMCTSConfig(merge_enabled=True)
    rng = random.Random(0)
    state = engine.create_initial_state(task.train_inputs)
    goal_last = engine.apply_move(
        engine.apply_move(state, ("width", (("ctx", 0),))), ("hmirror", (("ctx", 0),)))
    goal_first = engine.apply_move(
        engine.apply_move(state, ("hmirror", (("ctx", 0),))), ("width", (("ctx", 0),)))
    assert engine.canonical_key(goal_last) == engine.canonical_key(goal_first)
    a = m._make_node(goal_last, task.train_outputs, config, rng)
    b = m._make_node(goal_first, task.train_outputs, config, rng)
    assert a.is_terminal == b.is_terminal is True


def test_graded_reward_returns_partial_credit_where_binary_returns_zero():
    """Same unsolvable-in-one-step state, two reward modes: binary reports
    nothing, graded reports the cell-level overlap it actually achieved."""
    task = _load_real_task("68b16354")
    state = engine.create_initial_state(task.train_inputs)
    binary = m.ClassicalMCTSConfig(merge_enabled=False, reward_mode="binary")
    graded = m.ClassicalMCTSConfig(merge_enabled=False, reward_mode="graded")
    # rollout_depth=0 -> no moves taken, so the value is entirely the
    # starting state's own score under each mode.
    assert m.simulate(state, task.train_outputs, 0, binary, random.Random(0)) == 0.0
    graded_value = m.simulate(state, task.train_outputs, 0, graded, random.Random(0))
    assert 0.0 < graded_value < 1.0
    assert graded_value == engine.goal_progress(state, task.train_outputs)


def test_graded_reward_still_reports_one_on_a_solved_rollout():
    task = _load_real_task("68b16354")
    solved = engine.apply_move(engine.create_initial_state(task.train_inputs), ("hmirror", (("ctx", 0),)))
    for mode in ("binary", "graded"):
        config = m.ClassicalMCTSConfig(merge_enabled=False, reward_mode=mode)
        assert m.simulate(solved, task.train_outputs, 3, config, random.Random(0)) == 1.0


def test_reject_duplicates_keeps_every_node_state_unique_without_pooling():
    task = _load_real_task("68b16354")
    config = m.ClassicalMCTSConfig(merge_enabled=False, reject_duplicates=True)
    rng = random.Random(3)
    graph = m.run_search(task.train_inputs, task.train_outputs, config, budget=150, rollout_depth=5, rng=rng)

    # No pooling ever happened: path-keyed nodes, one incoming edge each.
    assert graph.merge_hits == 0
    in_degree = {}
    for node in graph.nodes.values():
        for edge in node.children.values():
            in_degree[edge.child_key] = in_degree.get(edge.child_key, 0) + 1
    assert all(d == 1 for d in in_degree.values())

    # Every node holds a distinct state, which is the point of the arm.
    keys = [engine.canonical_key(n.program_state) for n in graph.nodes.values()]
    assert len(set(keys)) == len(keys)


def test_reject_duplicates_actually_rejects_on_a_domain_with_transpositions():
    task = _load_real_task("68b16354")
    rng = random.Random(7)
    config = m.ClassicalMCTSConfig(merge_enabled=False, reject_duplicates=True)
    graph = m.run_search(task.train_inputs, task.train_outputs, config, budget=400, rollout_depth=3, rng=rng)
    assert graph.rejected > 0


def test_values_key_mode_merges_at_least_as_much_as_typed():
    task = _load_real_task("68b16354")
    counts = {}
    for mode in ("typed", "values"):
        config = m.ClassicalMCTSConfig(merge_enabled=True, merge_key_mode=mode)
        graph = m.run_search(task.train_inputs, task.train_outputs, config, budget=300,
                             rollout_depth=3, rng=random.Random(11))
        counts[mode] = graph.merge_hits
    assert counts["values"] >= counts["typed"]
