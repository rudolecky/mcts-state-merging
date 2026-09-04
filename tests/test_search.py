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
import pytest
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


# ---------- admission filter (reject_duplicates) ----------

def _reject_fixture(monkeypatch, proj_value=0.42, step_text="Step 1: 2 + 3 = 5\n"):
    lm = _FakeLM(num_hidden_layers=4)
    candidate_ids = torch.tensor(_text_to_ids(step_text))

    monkeypatch.setattr(search_mod, "generate_traces",
                        lambda lm_, prefix, K, mx, temp: [candidate_ids for _ in range(K)])
    monkeypatch.setattr(search_mod, "hidden_states_for_sequence",
                        lambda lm_, ids: _fake_hidden_states_constant(proj_value, lm_.num_hidden_layers))
    return lm


def _dummy_verifier(instance, step_bodies, answer_body):
    return False, {}


def test_reject_duplicates_keeps_one_node_and_never_pools(monkeypatch):
    """Same K identical candidates that merge mode collapses into one
    multi-parent node: the filter also yields one node, but by discarding
    the duplicates -- so no node ever holds statistics from two paths."""
    lm = _reject_fixture(monkeypatch)
    graph = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)
    config = SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                          projection=_trivial_projection(), merge_enabled=False,
                          reject_duplicates=True, tau=0.05)

    expand_node(lm, graph, graph.root_id, instance=None, verifier_fn=_dummy_verifier, config=config)

    step_nodes = [n for n in graph.nodes.values() if n.kind == "step"]
    assert len(step_nodes) == 1
    assert graph.rejected == 2  # the other two candidates were dropped
    assert graph.merge_hits == 0
    assert graph.new_nodes == 1
    # The surviving node has exactly one parent and one incoming edge: still a tree.
    assert step_nodes[0].parents == [graph.root_id]
    assert len(graph.nodes[graph.root_id].children) == 1


def _snapshot(graph):
    """Everything about a graph that a behavioural change could move --
    node counts alone would let a divergence through."""
    return {
        "next_id": graph._next_id,
        "depth_index": {k: list(v) for k, v in sorted(graph.depth_index.items())},
        "counters": (graph.new_nodes, graph.merge_hits, graph.rejected),
        "nodes": {
            nid: (n.depth, n.kind, tuple(n.full_ids.tolist()), n.proj_value, n.guide_value,
                  n.n_visits, n.w_value, n.is_terminal_correct, n.has_been_expanded,
                  n.no_admissible_children, tuple(n.parents),
                  tuple((e.child_id, e.prior, e.n_edge) for e in n.children))
            for nid, n in sorted(graph.nodes.items())
        },
    }


def test_reject_duplicates_is_inert_when_tau_is_zero(monkeypatch):
    """tau=0 disables the equivalence relation, so the filter must reduce to
    the baseline EXACTLY -- compared on a full graph snapshot, not on node
    counts, which would pass under any divergence that preserves them."""
    lm = _reject_fixture(monkeypatch)
    snaps = {}
    for name, cfg in (
        ("baseline", SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                                  projection=_trivial_projection(), merge_enabled=False)),
        ("reject", SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                                projection=_trivial_projection(), merge_enabled=False,
                                reject_duplicates=True, tau=0.0)),
    ):
        g = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)
        expand_node(lm, g, g.root_id, instance=None, verifier_fn=_dummy_verifier, config=cfg)
        snaps[name] = _snapshot(g)
    assert snaps["baseline"] == snaps["reject"]
    assert len(snaps["reject"]["nodes"]) == 4  # root + 3 distinct step nodes


def test_reject_duplicates_declines_a_live_relation_without_diverging(monkeypatch):
    """The complement of the tau=0 control: a tau large enough to be live,
    but candidates spread wider than it, so the filter is genuinely invoked
    and genuinely declines every time. Must still match the baseline."""
    lm = _FakeLM(num_hidden_layers=4)
    texts = ["Step 1: a = 1\n", "Step 1: b = 2\n", "Step 1: c = 3\n"]
    cand = [torch.tensor(_text_to_ids(t)) for t in texts]
    values = iter([0.10, 0.50, 0.90] * 4)
    monkeypatch.setattr(search_mod, "generate_traces", lambda lm_, pre, K, mx, tp: cand[:K])
    monkeypatch.setattr(search_mod, "hidden_states_for_sequence",
                        lambda lm_, ids: _fake_hidden_states_constant(next(values), lm_.num_hidden_layers))

    snaps = {}
    for name, cfg in (
        ("baseline", SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                                  projection=_trivial_projection(), merge_enabled=False)),
        ("reject", SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                                projection=_trivial_projection(), merge_enabled=False,
                                reject_duplicates=True, tau=0.05)),  # live, but < the 0.4 gaps
    ):
        g = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)
        expand_node(lm, g, g.root_id, instance=None, verifier_fn=_dummy_verifier, config=cfg)
        snaps[name] = _snapshot(g)
    assert snaps["baseline"] == snaps["reject"]


def test_reject_duplicates_spares_answer_candidates_by_default(monkeypatch):
    """An answer can be verified, so filtering it on an approximate
    projected-value match would discard a possibly-correct solution on the
    weakest available signal. Default must leave answers alone."""
    lm = _reject_fixture(monkeypatch, step_text="Answer: 5\n")
    verified = []

    def counting_verifier(instance, step_bodies, answer_body):
        verified.append(answer_body)
        return False, {}

    graph = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)
    config = SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                          projection=_trivial_projection(), merge_enabled=False,
                          reject_duplicates=True, tau=0.05)
    expand_node(lm, graph, graph.root_id, instance=None, verifier_fn=counting_verifier, config=config)

    answers = [n for n in graph.nodes.values() if n.kind == "answer"]
    assert len(answers) == 3  # none filtered
    assert graph.rejected == 0
    assert len(verified) == 3  # and each one actually got checked
    # Critically, sparing them must not mean pooling them: a spared answer
    # gets its own node, never an extra parent on the existing one.
    assert graph.merge_hits == 0
    assert all(len(n.parents) == 1 for n in answers)

    # The symmetric-with-merge variant does filter them.
    graph2 = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)
    config2 = SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                           projection=_trivial_projection(), merge_enabled=False,
                           reject_duplicates=True, reject_answer_nodes=True, tau=0.05)
    expand_node(lm, graph2, graph2.root_id, instance=None, verifier_fn=_dummy_verifier, config=config2)
    assert len([n for n in graph2.nodes.values() if n.kind == "answer"]) == 1
    assert graph2.rejected == 2


def test_reject_duplicates_marks_parent_alive_not_exhausted(monkeypatch):
    """A parent whose every candidate was filtered still did real work --
    marking it "exhausted" would back up 0.0 and tell its own parent that a
    merely-duplicate continuation is worthless, which is the value
    corruption this arm exists to avoid. Set up so EVERY candidate matches a
    pre-registered node."""
    lm = _reject_fixture(monkeypatch)
    graph = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)
    graph.register(_mk_node(99, depth=1, kind="step", proj_value=0.42))
    config = SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                          projection=_trivial_projection(), merge_enabled=False,
                          reject_duplicates=True, tau=0.05)

    expand_node(lm, graph, graph.root_id, instance=None, verifier_fn=_dummy_verifier, config=config)

    assert graph.rejected == 3  # all K filtered
    assert graph.new_nodes == 0
    assert graph.nodes[graph.root_id].kind == "root"  # not "exhausted"
    assert graph.nodes[graph.root_id].has_been_expanded is True
    # And nothing was pooled into the pre-existing node.
    assert graph.nodes[99].parents == []
    assert graph.merge_hits == 0


def test_run_search_never_expands_the_same_node_twice(monkeypatch):
    """The invariant the budget leak violated.

    Pre-existing arms satisfy it structurally: `select_leaf` descends only
    through nodes that are expanded AND have children, so it can only ever
    return an unexpanded node or a terminal one. A filtering arm broke it --
    an expansion that admits nothing leaves its node `has_been_expanded=True`
    with no children, `select_leaf` kept returning it, and `run_search` kept
    paying K fresh LLM generations to have them filtered again for the same
    reason (measured elsewhere at up to 58% of the arm's budget, invisible in
    every recorded statistic).

    Asserted directly, per node id, and across all four arms -- so it also
    guards the two arms that were always correct.
    """
    for kwargs in ({"merge_enabled": False},
                   {"merge_enabled": True},
                   {"merge_enabled": False, "reject_duplicates": True},
                   {"merge_enabled": False, "reject_duplicates": True, "reject_answer_nodes": True}):
        lm = _reject_fixture(monkeypatch)
        expanded: list[int] = []
        inner = search_mod.expand_node

        def recording(lm_, graph, node_id, instance, verifier_fn, config):
            expanded.append(node_id)
            return inner(lm_, graph, node_id, instance, verifier_fn, config)

        monkeypatch.setattr(search_mod, "expand_node", recording)
        search_mod.run_search(
            lm, torch.tensor(_text_to_ids("")), 0, None, _dummy_verifier,
            SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                         projection=_trivial_projection(), tau=0.05, **kwargs),
            budget=15,
        )
        assert len(expanded) == len(set(expanded)), (
            f"arm {kwargs} expanded a node more than once: {expanded}")
        monkeypatch.undo()


def test_a_fully_filtered_node_gets_marked_and_stops_the_descent(monkeypatch):
    """The mechanism behind the invariant above, checked on the state itself
    rather than through a full search: a node whose every candidate was
    filtered is marked, and `select_leaf` then refuses to return it while
    anything else is admissible."""
    lm = _reject_fixture(monkeypatch)
    graph = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)
    graph.register(_mk_node(99, depth=1, kind="step", proj_value=0.42))
    config = SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                          projection=_trivial_projection(), merge_enabled=False,
                          reject_duplicates=True, tau=0.05)

    expand_node(lm, graph, graph.root_id, instance=None, verifier_fn=_dummy_verifier, config=config)
    root = graph.nodes[graph.root_id]
    assert root.has_been_expanded is True and root.children == []  # the state in question

    # run_search's guard is what marks it; simulate the one iteration that
    # would follow, then confirm the loop cannot spend another expansion here.
    path = select_leaf(graph, config)
    assert path == [graph.root_id]
    root.no_admissible_children = True
    assert select_leaf(graph, config) == [graph.root_id]  # nowhere else to go -- run_search breaks


def test_blocked_children_are_skipped_by_selection():
    """A blocked node must not be selected while an admissible sibling
    exists, however attractive its Q looks."""
    root = Node(node_id=0, depth=0, kind="root", full_ids=torch.tensor([0]),
                n_visits=4, has_been_expanded=True)
    blocked = Node(node_id=1, depth=1, kind="step", full_ids=torch.tensor([0]),
                   proj_value=0.9, guide_value=0.9, no_admissible_children=True)
    ok = Node(node_id=2, depth=1, kind="step", full_ids=torch.tensor([0]),
              proj_value=0.1, guide_value=0.1)
    root.children = [Edge(child_id=1, prior=0.5, n_edge=1), Edge(child_id=2, prior=0.5, n_edge=1)]
    graph = SearchGraph(problem_prompt_len=0, nodes={0: root, 1: blocked, 2: ok}, root_id=0)
    config = SearchConfig(K=2, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                          projection=_trivial_projection(), merge_enabled=False,
                          reject_duplicates=True, tau=0.05)
    assert select_leaf(graph, config) == [0, 2]  # the 0.9-value blocked node is skipped


def test_merge_and_reject_cannot_be_combined():
    with pytest.raises(ValueError):
        SearchConfig(K=3, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                     projection=_trivial_projection(), merge_enabled=True,
                     reject_duplicates=True, tau=0.05)


def test_reject_duplicates_handles_a_mixed_step_and_answer_batch(monkeypatch):
    """The only case where the per-candidate answer exemption branches both
    ways inside one expansion. Steps get filtered, answers do not, and
    neither is ever pooled."""
    lm = _FakeLM(num_hidden_layers=4)
    cands = [torch.tensor(_text_to_ids(t)) for t in
             ("Step 1: a\n", "Step 1: b\n", "Answer: 5\n", "Answer: 6\n")]

    monkeypatch.setattr(search_mod, "generate_traces", lambda lm_, pre, K, mx, tp: cands[:K])
    monkeypatch.setattr(search_mod, "hidden_states_for_sequence",
                        lambda lm_, ids: _fake_hidden_states_constant(0.42, lm_.num_hidden_layers))

    graph = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)
    config = SearchConfig(K=4, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                          projection=_trivial_projection(), merge_enabled=False,
                          reject_duplicates=True, tau=0.05)
    expand_node(lm, graph, graph.root_id, instance=None, verifier_fn=_dummy_verifier, config=config)

    steps = [n for n in graph.nodes.values() if n.kind == "step"]
    answers = [n for n in graph.nodes.values() if n.kind == "answer"]
    assert len(steps) == 1  # second step candidate filtered
    assert len(answers) == 2  # both answers spared, despite matching each other
    assert graph.rejected == 1
    assert graph.merge_hits == 0
    assert all(len(n.parents) == 1 for n in steps + answers)


def test_reject_duplicates_filters_across_parents_without_ever_pooling(monkeypatch):
    """The multi-parent risk this feature exists to eliminate: a candidate
    under parent B matching a node already hanging under parent A. Merge
    mode would give that node a second parent; the filter must discard the
    candidate and leave the node with exactly one."""
    # Candidates carry TWO step boundaries, so a depth-1 parent can produce a
    # depth-2 child (the harness reads boundary index `parent.depth`).
    lm = _reject_fixture(monkeypatch, step_text="Step 1: x\nStep 2: y\n")
    graph = create_root(torch.tensor(_text_to_ids("")), prompt_len=0)

    # Two sibling parents at depth 1, created directly.
    for nid in (1, 2):
        graph.register(_mk_node(nid, depth=1, kind="step", proj_value=0.1 * nid))
        graph.nodes[nid].parents = [graph.root_id]
    config = SearchConfig(K=2, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                          projection=_trivial_projection(), merge_enabled=False,
                          reject_duplicates=True, tau=0.05)

    # Both parents' candidates land on the same depth-2 projected value.
    expand_node(lm, graph, 1, instance=None, verifier_fn=_dummy_verifier, config=config)
    depth2_after_first = [n for n in graph.nodes.values() if n.depth == 2]
    assert len(depth2_after_first) == 1

    expand_node(lm, graph, 2, instance=None, verifier_fn=_dummy_verifier, config=config)
    depth2 = [n for n in graph.nodes.values() if n.depth == 2]
    assert len(depth2) == 1  # parent 2's candidates all filtered against parent 1's child
    assert depth2[0].parents == [1]  # NOT [1, 2] -- no cross-parent pooling
    assert graph.merge_hits == 0
    # And parent 2 got no edge to it.
    assert graph.nodes[2].children == []


def test_uses_tau_matches_which_arms_actually_depend_on_tau(monkeypatch):
    """The load-bearing assumption of the checkpoint's shard naming.

    `run_search_experiment._uses_tau` infers tau-dependence from which flags an
    arm sets, but the real invariant lives in `expand_node`'s guard in this
    module -- and nothing linked them. If a new arm ever reaches
    `find_merge_target` by a route those flags do not describe, its shards
    would drop the tau tag and every point of a tau sweep would silently
    report the first point's result for that arm. So assert the property
    itself: for each arm, does running at two different taus actually change
    the search?
    """
    from mcts_phase0.run_search_experiment import _ARM_KWARGS, _uses_tau

    step_a, step_b = "Step 1: a = 1\n", "Step 1: b = 2\n"

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        prefix = lm_.tokenizer.decode(prefix_ids.tolist())
        if not prefix:
            return [torch.tensor(_text_to_ids(t)) for t in (step_a, step_b)][:K]
        return [torch.tensor(_text_to_ids(prefix + "Answer: 7\n")) for _ in range(K)]

    def fake_hidden_states(lm_, full_ids):
        text = lm_.tokenizer.decode(full_ids.tolist())
        # Two depth-1 steps 0.2 apart: within reach of a wide tau, not a narrow one.
        value = 0.5 if "Answer" in text else (0.4 if "a = 1" in text else 0.6)
        return _fake_hidden_states_constant(value, lm_.num_hidden_layers)

    monkeypatch.setattr(search_mod, "generate_traces", fake_generate_traces)
    monkeypatch.setattr(search_mod, "hidden_states_for_sequence", fake_hidden_states)

    def digest(arm, tau):
        graph = search_mod.run_search(
            _FakeLM(num_hidden_layers=4), torch.tensor(_text_to_ids("")), 0, None,
            _dummy_verifier,
            SearchConfig(K=2, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                         projection=_trivial_projection(), tau=tau, max_depth=3,
                         **_ARM_KWARGS[arm]),
            budget=12,
        )
        return (len(graph.nodes), graph.merge_hits, graph.rejected, graph.new_nodes)

    for arm in _ARM_KWARGS:
        changed = digest(arm, 0.0) != digest(arm, 0.5)
        assert changed == _uses_tau(arm), (
            f"arm {arm!r}: _uses_tau says {_uses_tau(arm)} but running at tau=0.0 vs 0.5 "
            f"{'changed' if changed else 'did not change'} the search. The checkpoint's shard "
            "naming depends on these agreeing.")
