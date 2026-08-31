"""Verification for the paper_repro package: the PDDL loader against a real
vendored instance, plain GUCT's leaf-count NEC formula against hand-computed
values, Monte Carlo backup pooling by AVERAGE (contrasted directly against
guct_uniform_blocksworld's min-pooling test and classical_mcts_blocksworld's
sum/count-pooling test -- three different backup rules, three explicit
tests, none assumed), and a cycle-avoidance check using a real PDDL fixture.
"""

import math
import random

from pyperplan.heuristics.relaxation import hFFHeuristic

from mcts_phase0.paper_repro import guct as guct_mod
from mcts_phase0.paper_repro.capped_gbfs import capped_gbfs
from mcts_phase0.paper_repro.pddl_loader import load_task

_FIXTURES = "tests/fixtures/pddl"


def _real_task():
    return load_task(f"{_FIXTURES}/blocks/domain.pddl", f"{_FIXTURES}/blocks/probBLOCKS-4-0.pddl")


# ---------- pddl_loader ----------

def test_load_task_parses_and_grounds_a_real_blocksworld_instance():
    task = _real_task()
    assert len(task.operators) > 0
    assert len(task.goals) > 0
    assert len(task.initial_state) > 0
    assert not task.goal_reached(task.initial_state)  # not already solved


def test_load_task_parses_a_real_gripper_instance():
    task = load_task(f"{_FIXTURES}/gripper/domain.pddl", f"{_FIXTURES}/gripper/prob01.pddl")
    assert len(task.operators) > 0
    assert len(task.goals) > 0


# ---------- capped_gbfs solves a real, easy instance ----------

def test_capped_gbfs_solves_a_real_small_instance():
    task = _real_task()
    heuristic = hFFHeuristic(task)
    solved, evals = capped_gbfs(task, heuristic, node_eval_limit=1000)
    assert solved is True
    assert evals <= 1000


# ---------- _guct_score: hand-computed values ----------

def test_guct_score_matches_hand_computed_value():
    parent = guct_mod.MCTSNode(state=frozenset(), n_visits=10)
    child = guct_mod.MCTSNode(state=frozenset(), n_visits=4, sum_h=20.0)  # mean=5.0
    edge = guct_mod.MCTSEdge(child_key="c")
    expected = 5.0 - 1.4 * math.sqrt(2.0 * math.log(10) / 4)
    assert guct_mod._guct_score(parent, edge, child, c=1.4) == expected


def test_guct_score_gives_negative_infinity_to_an_unvisited_child():
    parent = guct_mod.MCTSNode(state=frozenset(), n_visits=10)
    child = guct_mod.MCTSNode(state=frozenset(), n_visits=0)
    edge = guct_mod.MCTSEdge(child_key="c")
    assert guct_mod._guct_score(parent, edge, child, c=1.4) == float("-inf")


# ---------- backup: Monte Carlo pools the AVERAGE, not the min or the sum alone ----------

def test_monte_carlo_backup_pools_the_average():
    root = guct_mod.MCTSNode(state=frozenset())
    a = guct_mod.MCTSNode(state=frozenset())
    b = guct_mod.MCTSNode(state=frozenset())
    m = guct_mod.MCTSNode(state=frozenset())
    root.children = {"opA": guct_mod.MCTSEdge(child_key="a"), "opB": guct_mod.MCTSEdge(child_key="b")}
    a.children = {"opM1": guct_mod.MCTSEdge(child_key="m")}
    b.children = {"opM2": guct_mod.MCTSEdge(child_key="m")}

    class _Graph:
        nodes = {(): root, "a": a, "b": b, "m": m}
        root_key = ()

    graph = _Graph()
    guct_mod.backup(graph, [(), "a", "m"], ["opA", "opM1"], 0.8)
    guct_mod.backup(graph, [(), "b", "m"], ["opB", "opM2"], 0.2)

    assert m.n_visits == 2
    assert m.sum_h == 1.0
    assert m.sum_h / m.n_visits == 0.5  # average of 0.8 and 0.2, not their min (0.2) or raw sum (1.0)
    assert a.sum_h == 0.8 and a.n_visits == 1
    assert b.sum_h == 0.2 and b.n_visits == 1
    assert root.n_visits == 2
    assert root.sum_h == 1.0


# ---------- run_search: cycle-avoidance on a real reversible instance ----------

def test_run_search_does_not_hang_on_a_real_reversible_instance():
    task = _real_task()
    heuristic = hFFHeuristic(task)
    rng = random.Random(0)
    config = guct_mod.GUCTConfig(merge_enabled=True)
    graph = guct_mod.run_search(task.initial_state, task, config, budget=300, rng=rng, heuristic=heuristic)
    root = graph.nodes[graph.root_key]
    assert root.n_visits >= 300  # used its full budget, not stuck spinning on cycle-blocks
