"""Verification for search_deepsearch.py, independent of any real model
(mirrors test_search.py's own established convention of fully mocking
generate_traces/hidden_states_for_sequence rather than loading a real
model -- followed here too, not the real-model smoke test this module's
own plan sketched, since the existing file's approach is already the
project's convention for this harness):
- _frontier_score / _q_parent: hand-computed values, including the
  multi-parent-mean generalization and the no-visited-parent neutral case
- select_frontier_global: argmax over a hand-built graph
- backup_constrained: hand-worked trajectory, explicitly verifying the
  non-negative invariant is preserved (not just that some update happens)
- expand_node_entropy: harness neutrality (merge_enabled=False vs True),
  mirroring test_search.py's own expand_node test
"""

import math

import numpy as np
import torch

import mcts_phase0.search_deepsearch as ds_mod
from mcts_phase0.projection import FrozenProjection
from mcts_phase0.search import Edge, Node, SearchGraph, create_root
from mcts_phase0.search_deepsearch import (
    DeepSearchConfig,
    _frontier_score,
    _q_parent,
    backup_constrained,
    expand_node_entropy,
    select_frontier_global,
)

_VOCAB = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:\n=+-*/,.")
_CHAR_TO_ID = {c: i for i, c in enumerate(_VOCAB)}


class _FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(_VOCAB[i] for i in ids)


def _text_to_ids(text: str) -> list[int]:
    return [_CHAR_TO_ID[c] for c in text]


class _FakeLM:
    def __init__(self, num_hidden_layers=4):
        self.tokenizer = _FakeTokenizer()
        self.num_hidden_layers = num_hidden_layers


def _trivial_projection(layer="mid") -> FrozenProjection:
    return FrozenProjection(layer=layer, mean=np.array([0.0, 0.0]), std=np.array([1.0, 1.0]),
                             coef=np.array([1.0, 0.0]), intercept=0.0, alpha=1.0)


def _mk_node(node_id, depth, kind, entropy=None, w_value=0.0, n_visits=0, parents=None):
    return Node(node_id=node_id, depth=depth, kind=kind, full_ids=torch.tensor([0]),
                proj_value=0.0, entropy=entropy, w_value=w_value, n_visits=n_visits,
                parents=parents or [])


def _config(**overrides):
    defaults = dict(K=3, max_new_tokens_step=20, temperature=0.8, projection=_trivial_projection(),
                    merge_enabled=False, tau=0.05, max_depth=6, lambda1=0.4, lambda2=0.4, lambda3=0.01,
                    gamma_min=0.1, iterations_per_backup_round=4)
    defaults.update(overrides)
    return DeepSearchConfig(**defaults)


# ---------- _q_parent / _frontier_score ----------

def test_q_parent_is_neutral_when_no_parent_has_been_visited():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    parent = _mk_node(1, depth=0, kind="root", n_visits=0)
    child = _mk_node(2, depth=1, kind="step", parents=[1])
    graph.nodes = {1: parent, 2: child}
    assert _q_parent(graph, child) == 0.0


def test_q_parent_averages_across_multiple_visited_parents():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    parent_a = _mk_node(1, depth=0, kind="step", w_value=0.5, n_visits=1)
    parent_b = _mk_node(2, depth=0, kind="step", w_value=-0.3, n_visits=2)
    child = _mk_node(3, depth=1, kind="step", parents=[1, 2])
    graph.nodes = {1: parent_a, 2: parent_b, 3: child}
    assert _q_parent(graph, child) == (0.5 + -0.3) / 2


def test_frontier_score_matches_hand_computed_value():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    parent = _mk_node(1, depth=2, kind="step", w_value=0.5, n_visits=1)
    child = _mk_node(2, depth=3, kind="step", entropy=2.0, parents=[1])
    graph.nodes = {1: parent, 2: child}
    config = _config(lambda1=0.4, lambda2=0.4, lambda3=0.01, max_depth=6)
    expected = 0.4 * math.tanh(0.5) + 0.4 * 2.0 + 0.01 * math.sqrt(3 / 6)
    assert _frontier_score(graph, child, config) == expected


def test_frontier_score_zero_entropy_when_unset():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    parent = _mk_node(1, depth=0, kind="root", n_visits=0)
    child = _mk_node(2, depth=1, kind="step", entropy=None, parents=[1])
    graph.nodes = {1: parent, 2: child}
    config = _config()
    expected = config.lambda1 * math.tanh(0.0) + config.lambda3 * math.sqrt(1 / config.max_depth)
    assert _frontier_score(graph, child, config) == expected


# ---------- select_frontier_global ----------

def test_select_frontier_global_picks_expected_argmax():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    root = _mk_node(0, depth=0, kind="root", n_visits=1, w_value=0.0)
    root.has_been_expanded = True
    low = _mk_node(1, depth=1, kind="step", entropy=0.1, parents=[0])
    high = _mk_node(2, depth=1, kind="step", entropy=5.0, parents=[0])
    graph.nodes = {0: root, 1: low, 2: high}
    config = _config(lambda1=0.0, lambda2=1.0, lambda3=0.0)
    assert select_frontier_global(graph, config) == 2  # higher entropy -> higher score here


def test_select_frontier_global_none_when_nothing_left():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    root = _mk_node(0, depth=0, kind="answer")
    graph.nodes = {0: root}
    assert select_frontier_global(graph, _config()) is None


# ---------- backup_constrained ----------

def test_backup_constrained_terminal_gets_the_raw_trajectory_value():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    graph.nodes = {
        0: _mk_node(0, depth=0, kind="root"),
        1: _mk_node(1, depth=1, kind="step"),
        2: _mk_node(2, depth=2, kind="answer"),
    }
    backup_constrained(graph, [0, 1, 2], value=1.0, gamma_min=0.1)
    assert graph.nodes[2].w_value == 1.0
    assert graph.nodes[2].n_visits == 1


def test_backup_constrained_depth_decay_matches_hand_computed_value():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    graph.nodes = {
        0: _mk_node(0, depth=0, kind="root", w_value=0.0),
        1: _mk_node(1, depth=1, kind="step", w_value=0.0),
        2: _mk_node(2, depth=2, kind="answer"),
    }
    backup_constrained(graph, [0, 1, 2], value=1.0, gamma_min=0.1)
    # l=3 (terminal at 1-based index 3); node at path[0]=root is i=1: gamma=max(1/3,0.1)=1/3
    assert graph.nodes[0].w_value == (1 / 3) * 1.0
    # node at path[1] is i=2: gamma=max(2/3,0.1)=2/3
    assert graph.nodes[1].w_value == (2 / 3) * 1.0


def test_backup_constrained_preserves_the_non_negative_invariant():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    graph.nodes = {
        0: _mk_node(0, depth=0, kind="root", w_value=0.2),  # already positive
        1: _mk_node(1, depth=1, kind="answer"),
    }
    # A negative trajectory through this node would drag it below zero --
    # per the invariant, it should stay unchanged instead.
    backup_constrained(graph, [0, 1], value=-1.0, gamma_min=0.1)
    assert graph.nodes[0].w_value == 0.2  # unchanged, not flipped negative


def test_backup_constrained_allows_negative_q_when_starting_from_zero_or_below():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    graph.nodes = {
        0: _mk_node(0, depth=0, kind="root", w_value=0.0),
        1: _mk_node(1, depth=1, kind="answer"),
    }
    backup_constrained(graph, [0, 1], value=-1.0, gamma_min=0.1)
    assert graph.nodes[0].w_value < 0  # no positive evidence yet, so this is allowed to go negative


# ---------- expand_node_entropy: harness neutrality (mocked, no real model) ----------

def _fake_hidden_states_constant(value: float, num_hidden_layers: int, seq_len: int = 40):
    tup = tuple(torch.full((1, seq_len, 2), 0.0) for _ in range(num_hidden_layers + 1))
    for t in tup:
        t[0, :, 0] = value
    return tup


def test_expand_node_entropy_merges_identical_candidates_when_enabled_not_when_disabled(monkeypatch):
    lm = _FakeLM(num_hidden_layers=4)
    prompt_ids = torch.tensor(_text_to_ids(""))
    graph_baseline = create_root(prompt_ids, prompt_len=0)
    graph_treatment = create_root(prompt_ids, prompt_len=0)

    step_text = "Step 1: 2 + 3 = 5\n"
    candidate_ids = torch.tensor(_text_to_ids(step_text))

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        return [candidate_ids for _ in range(K)]

    def fake_hidden_states(lm_, full_ids):
        return _fake_hidden_states_constant(0.42, lm_.num_hidden_layers)

    def fake_entropy(lm_, hidden_vec):
        return 1.23  # constant, value doesn't matter for this structural-neutrality test

    monkeypatch.setattr(ds_mod, "generate_traces", fake_generate_traces)
    monkeypatch.setattr(ds_mod, "hidden_states_for_sequence", fake_hidden_states)
    monkeypatch.setattr(ds_mod, "next_token_entropy_from_hidden", fake_entropy)

    def dummy_verifier(instance, step_bodies, answer_body):
        return False, {}

    config_baseline = _config(K=3, merge_enabled=False)
    config_treatment = _config(K=3, merge_enabled=True, tau=0.05)

    expand_node_entropy(lm, graph_baseline, graph_baseline.root_id, instance=None, verifier_fn=dummy_verifier, config=config_baseline)
    expand_node_entropy(lm, graph_treatment, graph_treatment.root_id, instance=None, verifier_fn=dummy_verifier, config=config_treatment)

    assert len([n for n in graph_baseline.nodes.values() if n.kind == "step"]) == 3
    merged_nodes = [n for n in graph_treatment.nodes.values() if n.kind == "step"]
    assert len(merged_nodes) == 1
    assert merged_nodes[0].entropy == 1.23
