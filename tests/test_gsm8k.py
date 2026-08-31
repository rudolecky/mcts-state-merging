"""Verification for gsm8k.py, no network/model involved (except the one
`slow`-marked test at the bottom, which does a real fetch).
"""

import pytest

from mcts_phase0.datasets.gsm8k import (
    GSM8KInstance,
    _extract_number,
    _load_split,
    _parse_final_answer,
    build_prompt,
    generate_sample,
    parse_and_verify,
)

_FIXTURE_RECORDS = [
    {"question": "Natalia sold 48 clips in April and half as many in May. How many total?",
     "answer": "Natalia sold 48/2 = <<48/2=24>>24 in May.\nAltogether 48+24=<<48+24=72>>72.\n#### 72"},
    {"question": "A bakery has 40 muffins and sells 15. How many are left?",
     "answer": "40-15=<<40-15=25>>25 muffins left.\n#### 25"},
    {"question": "A price is $1,234 after a discount. Round trip cost is double that.",
     "answer": "1234*2=<<1234*2=2468>>2468.\n#### 2,468"},
]


# ---------- _parse_final_answer ----------

def test_parse_final_answer_extracts_number_and_strips_marker():
    solution, value = _parse_final_answer("step one.\nstep two.\n#### 42")
    assert value == 42.0
    assert "####" not in solution


def test_parse_final_answer_handles_comma_thousands_separator():
    _, value = _parse_final_answer("some steps\n#### 2,468")
    assert value == 2468.0


def test_parse_final_answer_raises_loudly_on_malformed_record():
    with pytest.raises(ValueError):
        _parse_final_answer("no marker here at all")


# ---------- GSM8KInstance / generate_sample ----------

def test_instance_path_count_is_fixed_sentinel():
    instances = generate_sample(n=3, seed=0, records=_FIXTURE_RECORDS)
    assert all(inst.path_count == -1 for inst in instances)


def test_instance_id_stable_and_tied_to_split_index():
    instances = generate_sample(n=3, seed=0, split="train", records=_FIXTURE_RECORDS)
    ids = [inst.id for inst in instances]
    assert len(set(ids)) == len(ids)  # unique
    assert all(i.startswith("gsm8k_train_") for i in ids)


def test_generate_sample_deterministic_given_seed():
    a = generate_sample(n=2, seed=7, records=_FIXTURE_RECORDS)
    b = generate_sample(n=2, seed=7, records=_FIXTURE_RECORDS)
    assert [inst.id for inst in a] == [inst.id for inst in b]


def test_generate_sample_caps_at_available_records_without_crashing():
    instances = generate_sample(n=100, seed=0, records=_FIXTURE_RECORDS)
    assert len(instances) == len(_FIXTURE_RECORDS)


# ---------- build_prompt ----------

def test_build_prompt_contains_instructions_and_fabricated_example():
    inst = generate_sample(n=1, seed=0, records=_FIXTURE_RECORDS)[0]
    prompt = build_prompt(inst)
    assert "Step N:" in prompt
    assert "Answer: <just the number>" in prompt
    assert "bakery makes 3 trays" in prompt  # the fabricated example, not a real GSM8K problem
    assert inst.question in prompt


# ---------- parse_and_verify ----------

def _inst(answer_value=72.0):
    return GSM8KInstance(
        question="q", reference_solution="s", answer_value=answer_value, split="train", split_index=0,
    )


def test_parse_and_verify_accepts_correct_answer():
    ok, info = parse_and_verify(_inst(72.0), ["Step 1: 48/2=24", "Step 2: 48+24=72"], "72")
    assert ok is True
    assert info["well_formed"] is True


def test_parse_and_verify_rejects_wrong_answer_but_well_formed():
    ok, info = parse_and_verify(_inst(72.0), ["Step 1: 48/2=24"], "71")
    assert ok is False
    assert info["well_formed"] is True  # structurally graded fine, just the wrong number


def test_parse_and_verify_rejects_missing_answer_line():
    ok, info = parse_and_verify(_inst(72.0), ["Step 1: 48/2=24"], None)
    assert ok is False
    assert info["well_formed"] is False


def test_parse_and_verify_rejects_no_step_lines():
    ok, info = parse_and_verify(_inst(72.0), [], "72")
    assert ok is False
    assert info["well_formed"] is False


def test_parse_and_verify_handles_dollar_sign_and_commas():
    assert _extract_number("$1,234") == 1234.0
    assert _extract_number("1,234.5") == 1234.5


def test_parse_and_verify_tolerates_trailing_period():
    ok, _ = parse_and_verify(_inst(72.0), ["Step 1: x"], "72.")
    assert ok is True


def test_parse_and_verify_rejects_trailing_prose():
    # strict parse, matching countdown's own convention -- "14 dollars" is not accepted
    # just because a number appears somewhere in the text.
    assert _extract_number("14 dollars") is None
    ok, info = parse_and_verify(_inst(14.0), ["Step 1: x"], "14 dollars")
    assert ok is False
    assert info["well_formed"] is False


# ---------- real network fetch (excluded from default -m "not slow" runs) ----------

@pytest.mark.slow
def test_load_split_real_fetch_counts_and_disjointness():
    train = _load_split("train")
    test = _load_split("test")
    assert len(train) == 7473
    assert len(test) == 1319
    train_qs = {r["question"] for r in train}
    test_qs = {r["question"] for r in test}
    assert train_qs.isdisjoint(test_qs)

    a = generate_sample(n=5, seed=3, split="train")
    b = generate_sample(n=5, seed=3, split="train")
    assert [inst.id for inst in a] == [inst.id for inst in b]
