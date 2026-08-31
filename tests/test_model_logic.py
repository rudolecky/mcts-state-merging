"""Unit tests for the pure, model-free logic inside model.py: step-boundary
detection (decode-only), pad-trimming, and layer-index resolution. The one
piece that genuinely needs the real model (hidden-state-reproducibility)
lives in test_model_consistency.py, marked slow/manual.
"""

import torch

from mcts_phase0.model import (
    _trim_trailing_pad,
    find_answer_boundary,
    find_position_based_boundaries,
    find_step_boundaries,
    resolve_layers,
)

_VOCAB = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:\n=+-*/,.")
_CHAR_TO_ID = {c: i for i, c in enumerate(_VOCAB)}


class _FakeTokenizer:
    """Character-level fake tokenizer: one 'token' per character. This is a
    harder stress test than real BPE, since token boundaries never align
    with line boundaries at all.
    """

    def decode(self, ids, skip_special_tokens=True):
        return "".join(_VOCAB[i] for i in ids)


def _text_to_ids(text: str) -> list[int]:
    return [_CHAR_TO_ID[c] for c in text]


def test_find_step_boundaries_locates_each_step_line_end():
    line1 = "Step 1: 2 + 3 = 5"
    line2 = "Step 2: 5 * 4 = 20"
    line3 = "Answer: 20"
    gen_text = f"{line1}\n{line2}\n{line3}"
    prompt_len = 5
    full_ids = torch.tensor(list(range(1000, 1000 + prompt_len)) + _text_to_ids(gen_text))

    boundaries = find_step_boundaries(_FakeTokenizer(), prompt_len, full_ids)

    assert len(boundaries) == 2
    assert boundaries[0] == prompt_len + len(line1) - 1
    expected_offset_2 = len(line1) + 1 + len(line2)
    assert boundaries[1] == prompt_len + expected_offset_2 - 1


def test_find_answer_boundary_locates_the_line():
    line1 = "Step 1: 2 + 3 = 5"
    line2 = "Answer: 5"
    gen_text = f"{line1}\n{line2}"
    prompt_len = 5
    full_ids = torch.tensor(list(range(1000, 1000 + prompt_len)) + _text_to_ids(gen_text))

    idx = find_answer_boundary(_FakeTokenizer(), prompt_len, full_ids)

    expected_offset = len(line1) + 1 + len(line2)
    assert idx == prompt_len + expected_offset - 1


def test_find_answer_boundary_returns_none_when_absent():
    gen_text = "Step 1: 2 + 3 = 5\nStep 2: 5 * 4 = 20"
    prompt_len = 5
    full_ids = torch.tensor(list(range(1000, 1000 + prompt_len)) + _text_to_ids(gen_text))
    assert find_answer_boundary(_FakeTokenizer(), prompt_len, full_ids) is None


def test_step_and_answer_boundary_tie_break_earlier_index_wins():
    """expand_node's tie-break: whichever of the next step line or an answer
    line completes at the EARLIER token index decides the new node's kind.
    Scripted case where a candidate's very next line is the answer, not
    another step -- the answer boundary must come out earlier than any
    (nonexistent) next step boundary.
    """
    line1 = "Step 1: 2 + 3 = 5"
    line2 = "Answer: 5"
    gen_text = f"{line1}\n{line2}"
    prompt_len = 5
    full_ids = torch.tensor(list(range(1000, 1000 + prompt_len)) + _text_to_ids(gen_text))

    steps = find_step_boundaries(_FakeTokenizer(), prompt_len, full_ids)
    answer_idx = find_answer_boundary(_FakeTokenizer(), prompt_len, full_ids)

    # parent was already at depth 1 (one step line already accounted for);
    # there is no *new* step boundary beyond that -- only the answer is new.
    parent_depth = 1
    new_step_idx = steps[parent_depth] if len(steps) > parent_depth else None
    assert new_step_idx is None
    assert answer_idx is not None


def test_find_step_boundaries_ignores_non_step_lines():
    gen_text = "Facts:\nStep 1: a is a b\nblah blah\nStep 2: b is a c\nAnswer: yes"
    prompt_len = 3
    full_ids = torch.tensor(list(range(2000, 2000 + prompt_len)) + _text_to_ids(gen_text))

    boundaries = find_step_boundaries(_FakeTokenizer(), prompt_len, full_ids)

    assert len(boundaries) == 2  # only the two genuine "Step N:" lines


def test_find_step_boundaries_empty_generation():
    full_ids = torch.tensor([1, 2, 3])
    assert find_step_boundaries(_FakeTokenizer(), prompt_len=3, full_ids=full_ids) == []


def test_find_step_boundaries_no_step_lines_at_all():
    gen_text = "Answer: no"
    prompt_len = 4
    full_ids = torch.tensor(list(range(4)) + _text_to_ids(gen_text))
    assert find_step_boundaries(_FakeTokenizer(), prompt_len, full_ids) == []


def test_trim_trailing_pad_cuts_at_first_eos():
    prompt_len = 2
    eos_id = 99
    full_ids = torch.tensor([1, 2, 5, 6, 99, 99, 99])
    trimmed = _trim_trailing_pad(full_ids, prompt_len, eos_id)
    assert trimmed.tolist() == [1, 2, 5, 6]


def test_trim_trailing_pad_no_eos_present():
    full_ids = torch.tensor([1, 2, 5, 6, 7])
    trimmed = _trim_trailing_pad(full_ids, prompt_len=2, eos_id=99)
    assert trimmed.tolist() == [1, 2, 5, 6, 7]


def test_find_position_based_boundaries_evenly_spaced_and_ascending():
    full_ids = torch.arange(200)
    boundaries = find_position_based_boundaries(prompt_len=10, full_ids=full_ids, num_snapshots=3)
    assert len(boundaries) == 3
    assert boundaries == sorted(boundaries)
    assert all(10 <= b < 200 for b in boundaries)
    # roughly quartile-spaced through the 190-token generated span
    assert boundaries[0] < boundaries[1] < boundaries[2]


def test_find_position_based_boundaries_too_short_returns_empty():
    full_ids = torch.arange(12)
    assert find_position_based_boundaries(prompt_len=10, full_ids=full_ids, num_snapshots=5) == []


def test_resolve_layers_arithmetic():
    layers = resolve_layers(28)
    assert layers == {"mid": 14, "3/4": 21, "final": 28}


def test_resolve_layers_rounding_on_odd_depth():
    layers = resolve_layers(27)
    assert layers["final"] == 27
    assert layers["mid"] == 13  # integer division
    assert layers["3/4"] == round(27 * 3 / 4)
