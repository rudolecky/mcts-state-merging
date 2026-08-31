"""Verification for search_oracle.py, independent of any real model:
- oracle_value: hand-computed against a small fact graph (on-track,
  wrong-fact, broken-chain, wrong-start, dead-end, already-solved cases)
- expand_node_oracle: harness neutrality, mirroring test_search.py's own
  monkeypatched generate_traces / hidden_states_for_sequence convention
"""

import numpy as np
import torch

import mcts_phase0.search_oracle as search_oracle_mod
from mcts_phase0.datasets.prosqa import ProsQAInstance
from mcts_phase0.projection import FrozenProjection
from mcts_phase0.search import SearchConfig, create_root
from mcts_phase0.search_oracle import expand_node_oracle, oracle_value

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


def _fake_hidden_states_constant(value: float, num_hidden_layers: int, seq_len: int = 40):
    tup = tuple(torch.full((1, seq_len, 2), 0.0) for _ in range(num_hidden_layers + 1))
    for t in tup:
        t[0, :, 0] = value
    return tup


def _trivial_projection(layer="mid") -> FrozenProjection:
    return FrozenProjection(layer=layer, mean=np.array([0.0, 0.0]), std=np.array([1.0, 1.0]),
                             coef=np.array([1.0, 0.0]), intercept=0.0, alpha=1.0)


def _mk_instance(facts, start, target):
    return ProsQAInstance(facts=tuple(facts), start=start, target=target, answer="yes",
                           path_count=1, correct_path=None)


# ---------- oracle_value ----------

def test_oracle_value_on_track_partial_chain():
    inst = _mk_instance([("a", "b"), ("b", "c")], start="a", target="c")
    text = "Step 1: a is a b\n"
    full_ids = torch.tensor(_text_to_ids(text))
    assert oracle_value(inst, _FakeTokenizer(), full_ids, prompt_len=0) == 1.0


def test_oracle_value_already_reached_target():
    inst = _mk_instance([("a", "b"), ("b", "c")], start="a", target="c")
    text = "Step 1: a is a b\nStep 2: b is a c\n"
    full_ids = torch.tensor(_text_to_ids(text))
    assert oracle_value(inst, _FakeTokenizer(), full_ids, prompt_len=0) == 1.0


def test_oracle_value_dead_end_cannot_reach_target():
    inst = _mk_instance([("a", "b"), ("b", "c"), ("a", "d")], start="a", target="c")
    text = "Step 1: a is a d\n"  # valid fact, but d is a dead end -- can't reach c
    full_ids = torch.tensor(_text_to_ids(text))
    assert oracle_value(inst, _FakeTokenizer(), full_ids, prompt_len=0) == 0.0


def test_oracle_value_fact_not_given():
    inst = _mk_instance([("a", "b"), ("b", "c")], start="a", target="c")
    text = "Step 1: a is a c\n"  # not a real edge in this graph
    full_ids = torch.tensor(_text_to_ids(text))
    assert oracle_value(inst, _FakeTokenizer(), full_ids, prompt_len=0) == 0.0


def test_oracle_value_broken_chain():
    inst = _mk_instance([("a", "b"), ("b", "c"), ("x", "c")], start="a", target="c")
    text = "Step 1: a is a b\nStep 2: x is a c\n"  # step 2 doesn't connect to step 1's end
    full_ids = torch.tensor(_text_to_ids(text))
    assert oracle_value(inst, _FakeTokenizer(), full_ids, prompt_len=0) == 0.0


def test_oracle_value_wrong_start():
    inst = _mk_instance([("a", "b"), ("b", "c"), ("z", "b")], start="a", target="c")
    text = "Step 1: z is a b\n"  # valid fact, but doesn't start from instance.start
    full_ids = torch.tensor(_text_to_ids(text))
    assert oracle_value(inst, _FakeTokenizer(), full_ids, prompt_len=0) == 0.0


def test_oracle_value_no_steps_yet_is_neutral():
    inst = _mk_instance([("a", "b"), ("b", "c")], start="a", target="c")
    full_ids = torch.tensor(_text_to_ids(""))
    assert oracle_value(inst, _FakeTokenizer(), full_ids, prompt_len=0) == 1.0


# ---------- expand_node_oracle ----------

def test_expand_node_oracle_sets_guide_value_from_ground_truth_not_projection(monkeypatch):
    lm = _FakeLM(num_hidden_layers=4)
    inst = _mk_instance([("a", "b"), ("b", "c"), ("a", "d")], start="a", target="c")
    prompt_ids = torch.tensor(_text_to_ids(""))
    graph = create_root(prompt_ids, prompt_len=0)

    on_track_text = "Step 1: a is a b\n"
    off_track_text = "Step 1: a is a d\n"
    on_track_ids = torch.tensor(_text_to_ids(on_track_text))
    off_track_ids = torch.tensor(_text_to_ids(off_track_text))

    def fake_generate_traces(lm_, prefix_ids, K, max_new_tokens, temperature):
        return [on_track_ids, off_track_ids]

    def fake_hidden_states(lm_, full_ids):
        # deliberately identical, far-from-truth proj_value for both candidates --
        # if guide_value came from this instead of the oracle, both nodes would
        # score the same, which the assertions below would catch.
        return _fake_hidden_states_constant(0.5, lm_.num_hidden_layers)

    monkeypatch.setattr(search_oracle_mod, "generate_traces", fake_generate_traces)
    monkeypatch.setattr(search_oracle_mod, "hidden_states_for_sequence", fake_hidden_states)

    config = SearchConfig(K=2, max_new_tokens_step=20, temperature=0.8, c_puct=1.0,
                           projection=_trivial_projection(), merge_enabled=False)
    expand_node_oracle(lm, graph, graph.root_id, instance=inst, verifier_fn=lambda *a: (False, {}), config=config)

    step_nodes = [n for n in graph.nodes.values() if n.kind == "step"]
    assert len(step_nodes) == 2
    guide_values = sorted(n.guide_value for n in step_nodes)
    assert guide_values == [0.0, 1.0]  # ground truth split, despite identical proj_value/hidden states
