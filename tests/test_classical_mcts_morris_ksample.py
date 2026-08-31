"""Verification for classical_mcts_morris_ksample.py, independent of any
model -- pure search logic, no GPU/network dependency. Mirrors
test_classical_mcts_ksample.py's load-bearing behavior (identical draws
stay distinct in baseline, merge in treatment) plus the cycle-avoidance
regression test established for every reversible-capable classical module
this session, and the guidance_depth_cap honest-vs-neutral distinction.
"""

import random

from mcts_phase0.classical_mcts_morris_ksample import (
    ClassicalMCTSConfig,
    MCTSEdge,
    MCTSNode,
    _make_node,
    _ucb1_score,
    backup,
    create_root,
    expand_batch,
    is_solved,
    run_search,
    select,
    simulate,
)
from mcts_phase0.datasets.morris_engine import O, X, replay_placements


class _ScriptedRng:
    def __init__(self, choices):
        self._choices = list(choices)

    def choice(self, seq):
        val = self._choices.pop(0)
        assert val in seq, f"scripted choice {val} not in {seq}"
        return val


_FORCED_WIN_BOARD = (X, O, O, None, None, None, O, X, X)  # X at 0,7,8; (7,4) wins immediately


# ---------- the load-bearing behavior ----------

def test_two_identical_draws_stay_distinct_in_baseline_but_merge_in_treatment():
    hero = X
    rng_treat = _ScriptedRng([(7, 4), (7, 4)])
    treat_cfg = ClassicalMCTSConfig(merge_enabled=True, K=2)
    treat_graph = create_root(_FORCED_WIN_BOARD, X, treat_cfg.merge_enabled)
    child_keys_treat = expand_batch(treat_graph, treat_graph.root_key, hero, treat_cfg, rng_treat)
    assert child_keys_treat[0] == child_keys_treat[1]
    assert len(treat_graph.nodes) == 2  # root + 1 merged child

    rng_base = _ScriptedRng([(7, 4), (7, 4)])
    base_cfg = ClassicalMCTSConfig(merge_enabled=False, K=2)
    base_graph = create_root(_FORCED_WIN_BOARD, X, base_cfg.merge_enabled)
    child_keys_base = expand_batch(base_graph, base_graph.root_key, hero, base_cfg, rng_base)
    assert child_keys_base[0] != child_keys_base[1]
    assert len(base_graph.nodes) == 3  # root + 2 distinct baseline children
    assert base_graph.nodes[child_keys_base[0]].board == base_graph.nodes[child_keys_base[1]].board


def test_backup_after_merged_draws_pools_visits_at_the_shared_node():
    hero = X
    rng = _ScriptedRng([(7, 4), (7, 4)])
    config = ClassicalMCTSConfig(merge_enabled=True, K=2)
    graph = create_root(_FORCED_WIN_BOARD, X, config.merge_enabled)
    child_keys = expand_batch(graph, graph.root_key, hero, config, rng)
    backup(graph, [graph.root_key, child_keys[0]], [0], 0.8)
    backup(graph, [graph.root_key, child_keys[1]], [1], 0.2)
    merged = graph.nodes[child_keys[0]]
    assert merged.n_visits == 2
    assert merged.w_value == 1.0
    assert graph.nodes[graph.root_key].children[0].n_edge == 1
    assert graph.nodes[graph.root_key].children[1].n_edge == 1


# ---------- _ucb1_score / select ----------

def test_ucb1_gives_infinite_priority_to_untried_edges():
    parent = MCTSNode(board=(), to_move=X, n_visits=10)
    child = MCTSNode(board=(), to_move=O, n_visits=3, w_value=2.0)
    untried_edge = MCTSEdge(child_key=((1, 2)), n_edge=0)
    assert _ucb1_score(parent, untried_edge, child, c=1.4) == float("inf")


# ---------- guidance_depth_cap: honest vs. neutral ----------

def test_simulate_none_cap_returns_zero_on_exhaustion():
    # X plays (0,1) [no win], then O plays (4,0) [no win] -- 2 non-decisive plies.
    board = replay_placements((0, 4, 8, 2, 6, 5))
    rng = _ScriptedRng([(0, 1), (4, 0)])
    config = ClassicalMCTSConfig(merge_enabled=True, rollout_depth=2, guidance_depth_cap=None)
    value = simulate(board, X, hero=X, config=config, rng=rng)
    assert value == 0.0


def test_simulate_with_cap_returns_neutral_value_on_exhaustion():
    board = replay_placements((0, 4, 8, 2, 6, 5))
    rng = _ScriptedRng([(0, 1), (4, 0)])
    config = ClassicalMCTSConfig(merge_enabled=True, rollout_depth=20, guidance_depth_cap=2)
    value = simulate(board, X, hero=X, config=config, rng=rng)
    assert value == 0.5


def test_simulate_still_returns_decisive_win_within_cap():
    rng = _ScriptedRng([(7, 4)])
    config = ClassicalMCTSConfig(merge_enabled=True, rollout_depth=20, guidance_depth_cap=1)
    value = simulate(_FORCED_WIN_BOARD, X, hero=X, config=config, rng=rng)
    assert value == 1.0


# ---------- run_search: cycle-avoidance regression + end-to-end smoke test ----------

def test_run_search_does_not_hang_on_a_reversible_position():
    board = replay_placements((0, 4, 8, 2, 6, 5))
    rng = random.Random(0)
    config = ClassicalMCTSConfig(merge_enabled=True, K=4, rollout_depth=15)
    graph = run_search(board, X, config, budget=100, rng=rng)
    assert len(graph.nodes) > 50  # would plateau near a handful of nodes if stuck


def test_run_search_solves_a_forced_win_puzzle_most_of_the_time():
    solved_count = 0
    trials = 20
    for seed in range(trials):
        rng = random.Random(seed)
        config = ClassicalMCTSConfig(merge_enabled=True, K=4, rollout_depth=10)
        graph = run_search(_FORCED_WIN_BOARD, X, config, budget=20, rng=rng)
        if is_solved(graph):
            solved_count += 1
    assert solved_count >= trials * 0.7
