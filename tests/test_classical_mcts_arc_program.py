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
