"""Verification for search.py, independent of any real model:
- find_merge_target: scope correctness (depth/kind bucketing, tau threshold)
- backup: pooling correctness on a hand-built two-parent DAG
- select_leaf: PUCT selection on a hand-built graph
- expand_node: harness neutrality (merge_enabled=False vs True) and
  exhausted-node handling, via monkeypatched generate_traces /
  hidden_states_for_sequence -- no real model, no network, no torch device.
"""

import math

import numpy as np
import torch

import mcts_phase0.search as search_mod
from mcts_phase0.projection import FrozenProjection
from mcts_phase0.search import (
    Edge,
    Node,
    SearchConfig,
    SearchGraph,
    backup,
    create_root,
    expand_node,
    find_merge_target,
    run_random_search,
    select_leaf,
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
    # coef=[1,0], mean=0, std=1 -> apply_projection(vec) == vec[0], clipped [0,1]
    return FrozenProjection(layer=layer, mean=np.array([0.0, 0.0]), std=np.array([1.0, 1.0]),
                             coef=np.array([1.0, 0.0]), intercept=0.0, alpha=1.0)


# ---------- find_merge_target ----------

def _mk_node(node_id, depth, kind, proj_value):
    return Node(node_id=node_id, depth=depth, kind=kind, full_ids=torch.tensor([0]), proj_value=proj_value)


def test_find_merge_target_fires_within_tau_at_same_depth_and_kind():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    graph.register(_mk_node(1, depth=2, kind="step", proj_value=0.50))
    graph.register(_mk_node(2, depth=2, kind="step", proj_value=0.90))

    target = find_merge_target(graph, new_depth=2, new_kind="step", proj_value=0.52, tau=0.05)
    assert target == 1  # closest within tau


def test_find_merge_target_none_when_outside_tau():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    graph.register(_mk_node(1, depth=2, kind="step", proj_value=0.50))
    assert find_merge_target(graph, new_depth=2, new_kind="step", proj_value=0.90, tau=0.05) is None


def test_find_merge_target_never_crosses_depth():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    graph.register(_mk_node(1, depth=3, kind="step", proj_value=0.50))  # different depth
    assert find_merge_target(graph, new_depth=2, new_kind="step", proj_value=0.50, tau=1.0) is None


def test_find_merge_target_never_crosses_kind():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    graph.register(_mk_node(1, depth=2, kind="answer", proj_value=0.50))  # different kind
    assert find_merge_target(graph, new_depth=2, new_kind="step", proj_value=0.50, tau=1.0) is None


# ---------- backup: pooling on a hand-built two-parent DAG ----------

def test_backup_pools_at_shared_node_keeps_edge_counts_local():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    root = Node(node_id=0, depth=0, kind="root", full_ids=torch.tensor([0]))
    graph.nodes[0] = root
    a = _mk_node(1, depth=1, kind="step", proj_value=0.5)
    b = _mk_node(2, depth=1, kind="step", proj_value=0.5)
    m = _mk_node(3, depth=2, kind="step", proj_value=0.5)  # merge target, two parents
    graph.nodes[1], graph.nodes[2], graph.nodes[3] = a, b, m
    root.children = [Edge(child_id=1, prior=0.5), Edge(child_id=2, prior=0.5)]
    a.children = [Edge(child_id=3, prior=1.0)]
    b.children = [Edge(child_id=3, prior=1.0)]

    backup(graph, [0, 1, 3], 0.8)
    backup(graph, [0, 2, 3], 0.2)

    assert m.n_visits == 2
    assert m.w_value == 1.0  # pooled across both lineages: 0.8 + 0.2
    assert a.children[0].n_edge == 1  # each edge's own count reflects only its own traversals
    assert b.children[0].n_edge == 1
    assert root.n_visits == 2  # visited via both paths


# ---------- select_leaf: PUCT on a hand-built graph ----------

def test_select_leaf_picks_expected_argmax():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    root = Node(node_id=0, depth=0, kind="root", full_ids=torch.tensor([0]), n_visits=10, has_been_expanded=True)
    child_a = _mk_node(1, depth=1, kind="step", proj_value=0.9)  # unvisited, high prior value -> high Q via FPU
    child_b = _mk_node(2, depth=1, kind="step", proj_value=0.1)
    graph.nodes = {0: root, 1: child_a, 2: child_b}
    root.children = [Edge(child_id=1, prior=0.5, n_edge=5), Edge(child_id=2, prior=0.5, n_edge=0)]
    child_a.has_been_expanded = False
    child_b.has_been_expanded = False

    config = SearchConfig(K=2, max_new_tokens_step=10, temperature=0.8, c_puct=0.1,
                           projection=_trivial_projection(), merge_enabled=False)
    path = select_leaf(graph, config)
    # child_a has much higher FPU value (0.9 vs 0.1) and a small c_puct, so it should
    # dominate despite its edge having more prior visits (larger denominator)
    assert path == [0, 1]


def test_select_leaf_stops_at_unexpanded_root():
    graph = create_root(torch.tensor([1, 2, 3]), prompt_len=0)
    config = SearchConfig(K=2, max_new_tokens_step=10, temperature=0.8, c_puct=1.0,
                           projection=_trivial_projection(), merge_enabled=False)
    assert select_leaf(graph, config) == [graph.root_id]


def test_select_leaf_stops_at_answer_node():
    graph = SearchGraph(problem_prompt_len=0, nodes={}, root_id=0)
    root = Node(node_id=0, depth=0, kind="root", full_ids=torch.tensor([0]), has_been_expanded=True)
    answer = Node(node_id=1, depth=1, kind="answer", proj_value=1.0, full_ids=torch.tensor([0]),
                  is_terminal_correct=True)
    graph.nodes = {0: root, 1: answer}
    root.children = [Edge(child_id=1, prior=1.0)]

    config = SearchConfig(K=2, max_new_tokens_step=10, temperature=0.8, c_puct=1.0,
                           projection=_trivial_projection(), merge_enabled=False)
    assert select_leaf(graph, config) == [0, 1]


# ---------- expand_node: harness neutrality + exhausted handling (mocked) ----------

def _fake_hidden_states_constant(value: float, num_hidden_layers: int, seq_len: int = 40):
    tup = tuple(torch.full((1, seq_len, 2), 0.0) for _ in range(num_hidden_layers + 1))
    for t in tup:
        t[0, :, 0] = value
    return tup


def test_expand_node_merges_identical_candidates_when_enabled_not_when_disabled(monkeypatch):
    lm = _FakeLM(num_hidden_layers=4)
    prompt_ids = torch.tensor(_text_to_ids(""))
    graph_baseline = create_root(prompt_ids, prompt_len=0)
    graph_treatment = create_root(prompt_ids, prompt_len=0)

    step_text = "Step 1: 2 + 3 = 5\n"
    candidate_ids = torch.tensor(_text_to_ids(step_text))

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        return [candidate_ids for _ in range(K)]  # K identical candidates

    def fake_hidden_states(lm_, full_ids):
        return _fake_hidden_states_constant(0.42, lm_.num_hidden_layers)  # identical proj_value every time

    monkeypatch.setattr(search_mod, "generate_traces", fake_generate_traces)
    monkeypatch.setattr(search_mod, "hidden_states_for_sequence", fake_hidden_states)

    def dummy_verifier(instance, step_bodies, answer_body):
        return False, {}

    config_baseline = SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                                    projection=_trivial_projection(), merge_enabled=False)
    config_treatment = SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                                     projection=_trivial_projection(), merge_enabled=True, tau=0.05)

    expand_node(lm, graph_baseline, graph_baseline.root_id, instance=None, verifier_fn=dummy_verifier, config=config_baseline)
    expand_node(lm, graph_treatment, graph_treatment.root_id, instance=None, verifier_fn=dummy_verifier, config=config_treatment)

    # baseline: every candidate becomes its own node -> 3 new "step" nodes
    assert len([n for n in graph_baseline.nodes.values() if n.kind == "step"]) == 3
    # treatment: identical proj_value for all K -> they all merge into ONE node
    assert len([n for n in graph_treatment.nodes.values() if n.kind == "step"]) == 1
    merged = next(n for n in graph_treatment.nodes.values() if n.kind == "step")
    assert len(merged.parents) == 1  # same parent for all K draws in this scripted case
    # the single edge from root to the merged node should have accumulated all K priors
    edge = graph_treatment.nodes[graph_treatment.root_id].children[0]
    assert math.isclose(edge.prior, 1.0, rel_tol=1e-9)


def test_node_guide_value_defaults_to_proj_value():
    node = Node(node_id=1, depth=1, kind="step", full_ids=torch.tensor([0]), proj_value=0.37)
    assert node.guide_value == 0.37


def test_node_guide_value_explicit_override_not_clobbered():
    node = Node(node_id=1, depth=1, kind="step", full_ids=torch.tensor([0]), proj_value=0.37, guide_value=0.91)
    assert node.guide_value == 0.91


def test_expand_node_rollout_value_source_overrides_guide_value_not_proj_value(monkeypatch):
    # C6 ablation: value_source="rollout" must (a) still compute proj_value normally (the merge
    # decision stays projection-based), (b) set guide_value from score_rollouts instead, and
    # (c) never call score_rollouts for a candidate that merges into an existing node.
    lm = _FakeLM(num_hidden_layers=4)
    prompt_ids = torch.tensor(_text_to_ids(""))
    graph = create_root(prompt_ids, prompt_len=0)

    step_text = "Step 1: 2 + 3 = 5\n"
    candidate_ids = torch.tensor(_text_to_ids(step_text))

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        return [candidate_ids for _ in range(K)]  # K identical candidates -> all merge together

    def fake_hidden_states(lm_, full_ids):
        return _fake_hidden_states_constant(0.42, lm_.num_hidden_layers)

    rollout_calls = []

    def fake_score_rollouts(lm_, instance, verifier_fn, prefix_ids, prompt_len, num_rollouts, max_new_tokens, temperature):
        rollout_calls.append(prefix_ids)
        return 0.99  # deliberately far from proj_value=0.42, so a mix-up is obvious

    monkeypatch.setattr(search_mod, "generate_traces", fake_generate_traces)
    monkeypatch.setattr(search_mod, "hidden_states_for_sequence", fake_hidden_states)
    monkeypatch.setattr(search_mod, "score_rollouts", fake_score_rollouts)

    config = SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                           projection=_trivial_projection(), merge_enabled=True, tau=0.05,
                           value_source="rollout", num_rollouts=2, rollout_max_new_tokens=10)
    expand_node(lm, graph, graph.root_id, instance=None, verifier_fn=lambda *a: (False, {}), config=config)

    merged = next(n for n in graph.nodes.values() if n.kind == "step")
    assert math.isclose(merged.proj_value, 0.42, rel_tol=1e-6)  # merge decision still projection-based
    assert math.isclose(merged.guide_value, 0.99, rel_tol=1e-6)  # PUCT guidance uses the rollout value
    assert len(rollout_calls) == 1  # only the first (node-creating) candidate paid for a rollout;
                                     # the other K-1 merged into it and never called score_rollouts


def test_expand_node_marks_exhausted_when_all_candidates_dead(monkeypatch):
    lm = _FakeLM(num_hidden_layers=4)
    prompt_ids = torch.tensor(_text_to_ids(""))
    graph = create_root(prompt_ids, prompt_len=0)

    dead_text = "no boundary here at all"  # no "Step N:" or "Answer:" line, no trailing newline
    candidate_ids = torch.tensor(_text_to_ids(dead_text))

    calls = {"generate": 0}

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        calls["generate"] += 1
        return [candidate_ids for _ in range(K)]

    monkeypatch.setattr(search_mod, "generate_traces", fake_generate_traces)
    monkeypatch.setattr(search_mod, "hidden_states_for_sequence",
                         lambda lm_, full_ids: _fake_hidden_states_constant(0.5, lm_.num_hidden_layers))

    config = SearchConfig(K=2, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                           projection=_trivial_projection(), merge_enabled=False)
    expand_node(lm, graph, graph.root_id, instance=None, verifier_fn=lambda *a: (False, {}), config=config)

    root = graph.nodes[graph.root_id]
    assert root.kind == "exhausted"
    assert root.children == []
    assert calls["generate"] == 1

    # re-selecting an exhausted node must not attempt to expand it again
    path = select_leaf(graph, config)
    assert path == [graph.root_id]
    assert root.kind == "exhausted"  # select_leaf itself never calls generate_traces


# ---------- run_random_search ----------

def test_run_random_search_solves_within_a_walk(monkeypatch):
    lm = _FakeLM(num_hidden_layers=4)
    prompt_ids = torch.tensor(_text_to_ids(""))

    calls = {"n": 0}

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        calls["n"] += 1
        if calls["n"] == 1:
            return [torch.tensor(_text_to_ids("Step 1: 2 + 3 = 5\n"))]
        return [torch.tensor(_text_to_ids("Step 1: 2 + 3 = 5\nAnswer: 5\n"))]

    monkeypatch.setattr(search_mod, "generate_traces", fake_generate_traces)

    def verifier(instance, step_bodies, answer_body):
        return answer_body == "5", {}

    solved = run_random_search(lm, prompt_ids, prompt_len=0, instance=None, verifier_fn=verifier,
                                budget=1, max_new_tokens_step=20, temperature=0.8, max_depth=6)
    assert solved is True
    assert calls["n"] == 2  # one call for the step, one for the answer -- no wasted extra walk


def test_run_random_search_dead_end_returns_false_without_exhausting_budget(monkeypatch):
    lm = _FakeLM(num_hidden_layers=4)
    prompt_ids = torch.tensor(_text_to_ids(""))
    dead_text = "no boundary here at all"

    calls = {"n": 0}

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        calls["n"] += 1
        return [torch.tensor(_text_to_ids(dead_text))]

    monkeypatch.setattr(search_mod, "generate_traces", fake_generate_traces)

    solved = run_random_search(lm, prompt_ids, prompt_len=0, instance=None,
                                verifier_fn=lambda *a: (False, {}),
                                budget=1, max_new_tokens_step=20, temperature=0.8, max_depth=6)
    assert solved is False
    assert calls["n"] == 1  # dead candidate ends the walk immediately, no further steps


def test_run_random_search_tries_full_budget_when_never_correct(monkeypatch):
    lm = _FakeLM(num_hidden_layers=4)
    prompt_ids = torch.tensor(_text_to_ids(""))
    wrong_answer_text = "Answer: 5\n"

    calls = {"n": 0}

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        calls["n"] += 1
        return [torch.tensor(_text_to_ids(wrong_answer_text))]

    monkeypatch.setattr(search_mod, "generate_traces", fake_generate_traces)

    solved = run_random_search(lm, prompt_ids, prompt_len=0, instance=None,
                                verifier_fn=lambda *a: (False, {}),
                                budget=3, max_new_tokens_step=20, temperature=0.8, max_depth=6)
    assert solved is False
    assert calls["n"] == 3  # every one of the 3 independent walks got tried
