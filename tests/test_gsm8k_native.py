"""Verification for gsm8k_native.py, no network/model involved."""

from mcts_phase0.datasets.gsm8k import GSM8KInstance
from mcts_phase0.datasets.gsm8k_native import build_prompt, native_split, parse_and_verify


def _inst(answer_value=42.0):
    return GSM8KInstance(
        question="q", reference_solution="s", answer_value=answer_value, split="train", split_index=0,
    )


# ---------- native_split ----------

def test_native_split_finds_boxed_answer():
    text = "Let me think... 20+22=42. So the answer is \\boxed{42}."
    step_bodies, answer_body = native_split(text)
    assert step_bodies == []
    assert answer_body == "42"


def test_native_split_finds_last_boxed_when_multiple():
    # models sometimes second-guess themselves and re-box a corrected answer
    text = "First I thought \\boxed{40}, but rechecking: \\boxed{42}."
    _, answer_body = native_split(text)
    assert answer_body == "42"


def test_native_split_falls_back_to_answer_is_pattern():
    text = "<think>reasoning...</think>\nThe final answer is 42."
    _, answer_body = native_split(text)
    assert answer_body == "42."  # tolerance for the trailing period is parse_and_verify's job


def test_native_split_returns_none_when_nothing_found():
    text = "I got confused and never concluded."
    _, answer_body = native_split(text)
    assert answer_body is None


def test_native_split_step_bodies_always_empty():
    # no per-step structure exists in this mode by construction
    text = "Step 1: something\n\\boxed{42}"
    step_bodies, _ = native_split(text)
    assert step_bodies == []


# ---------- build_prompt ----------

def test_build_prompt_asks_for_boxed_not_step_lines():
    inst = _inst()
    prompt = build_prompt(inst)
    assert "\\boxed{}" in prompt
    assert "Step N:" not in prompt
    assert inst.question in prompt


# ---------- parse_and_verify ----------

def test_parse_and_verify_accepts_correct_boxed_answer():
    ok, info = parse_and_verify(_inst(42.0), *native_split("reasoning \\boxed{42}"))
    assert ok is True
    assert info["well_formed"] is True


def test_parse_and_verify_rejects_wrong_answer_but_well_formed():
    ok, info = parse_and_verify(_inst(42.0), *native_split("reasoning \\boxed{41}"))
    assert ok is False
    assert info["well_formed"] is True


def test_parse_and_verify_rejects_missing_answer():
    ok, info = parse_and_verify(_inst(42.0), *native_split("no conclusion reached"))
    assert ok is False
    assert info["well_formed"] is False


def test_parse_and_verify_never_requires_step_bodies():
    # the key contrast with gsm8k.parse_and_verify: well_formed doesn't
    # depend on step_bodies being non-empty, since it's always [] here
    ok, info = parse_and_verify(_inst(42.0), [], "42")
    assert ok is True
    assert info["well_formed"] is True
