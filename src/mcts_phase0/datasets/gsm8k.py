"""GSM8K: OpenAI's real grade-school math word-problem benchmark, as the
first non-synthetic dataset in this project. Fetched as plain-text jsonl
directly from the original publisher's repo (openai/grade-school-math on
GitHub), pinned to a fixed commit, rather than the HF-hosted mirror (which
only ships parquet and would need a new dependency to read) -- zero new
dependencies, and this project's whole methodology leans on seeded-RNG
determinism, so an unpinned external fetch would be its first
non-reproducible input surface.

IMPORTANT, more than for any other dataset in this project: `parse_and_verify`
below can only check the FINAL numeric answer against ground truth. GSM8K's
solutions are free natural language with an informal "<<a op b=c>>"
calculator-annotation convention, not an enforced grammar the way
countdown's "a OP b = result" or connect_four's "drop in column C" are --
there is nothing to mechanically re-derive a claimed line against. Every
other verifier in this project treats "never trust the model's own claimed
intermediate values" as load-bearing; this one structurally cannot. A trace
that reaches the right final number via non-sequitur or internally
inconsistent reasoning is scored `is_correct=True` here. "Solved" on this
dataset means "got the final answer right," not "produced a verified-sound
derivation" -- weaker than every other dataset in this project, and that
gap must travel with any result reported from it.
"""

from __future__ import annotations

import json
import math
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_COMMIT_SHA = "3101c7d5072418e28b9008a6636bde82a006892c"  # openai/grade-school-math @ master, 2021-11-19
_EXPECTED_COUNTS = {"train": 7473, "test": 1319}
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass(frozen=True)
class GSM8KInstance:
    question: str
    reference_solution: str  # verbatim, including the "#### N" tail -- for debugging only
    answer_value: float
    split: str
    split_index: int
    # Fixed sentinel, not a real quantity: GSM8K has no derivation-multiplicity
    # concept analogous to countdown's grammar enumeration, prosqa's DAG path
    # count, or connect_four's first-move count. -1 (not None) keeps this
    # field type-stable and unambiguous against every real dataset's values
    # (0 = prosqa's "no path"; 1+ = real multiplicity elsewhere).
    path_count: int = -1

    @property
    def id(self) -> str:
        return f"gsm8k_{self.split}_{self.split_index}"


def _parse_final_answer(answer_field: str) -> tuple[str, float]:
    """Split GSM8K's "<solution text>\\n#### <number>" convention. Every
    official record is guaranteed this format -- fails loudly on a
    malformed record (a canary that the wrong file was fetched), never
    swallows the error.
    """
    if "#### " not in answer_field:
        raise ValueError(f"GSM8K record missing '#### ' final-answer marker: {answer_field!r}")
    solution_text, _, tail = answer_field.partition("#### ")
    return solution_text.strip(), float(tail.strip().replace(",", ""))


def _instance_from_record(record: dict, split: str, split_index: int) -> GSM8KInstance:
    solution_text, answer_value = _parse_final_answer(record["answer"])
    return GSM8KInstance(
        question=record["question"], reference_solution=solution_text,
        answer_value=answer_value, split=split, split_index=split_index,
    )


def _parse_jsonl_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def _fetch_split_text(split: str, cache_dir: str = "results/gsm8k_cache") -> str:
    cache_path = Path(cache_dir) / f"{split}.jsonl"
    if cache_path.exists():
        return cache_path.read_text()
    url = f"https://raw.githubusercontent.com/openai/grade-school-math/{_COMMIT_SHA}/grade_school_math/data/{split}.jsonl"
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text)
    return text


def _load_split(split: str) -> list[dict]:
    records = _parse_jsonl_lines(_fetch_split_text(split))
    expected = _EXPECTED_COUNTS.get(split)
    if expected is not None and len(records) != expected:
        raise ValueError(f"GSM8K {split} split: expected {expected} records, got {len(records)} -- wrong file fetched?")
    return records


def generate_sample(
    n: int, seed: int = 0, split: str = "train", records: list[dict] | None = None,
) -> list[GSM8KInstance]:
    """Flat deterministic draw -- no low/high stratification, unlike
    countdown/connect_four: GSM8K has no a-priori difficulty proxy to
    stratify on (no path_count analog), which is exactly what calibrate.py's
    pass-rate pre-pass exists to discover empirically instead.

    `records` is an injection point for tests (bypasses the real network
    fetch entirely) -- production callers pass split only.
    """
    if records is None:
        records = _load_split(split)
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), min(n, len(records)))
    return [_instance_from_record(records[i], split, i) for i in indices]


def build_prompt(instance: GSM8KInstance) -> str:
    example = (
        "Example:\n"
        "A bakery makes 3 trays of muffins with 8 muffins per tray, then sells 10 of them. "
        "How many muffins are left?\n"
        "Step 1: 3 * 8 = 24\n"
        "Step 2: 24 - 10 = 14\n"
        "Answer: 14\n\n"
    )
    body = (
        f"{instance.question}\n"
        "Work through this step by step. Write one computation or reasoning fact per line as\n"
        "Step N: <computation or fact>\n"
        "Then write\n"
        "Answer: <just the number>\n"
    )
    return example + body


def _extract_number(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.strip().rstrip(".")
    if t.startswith("$"):
        t = t[1:]
    t = t.replace(",", "")
    if not _NUMBER_RE.match(t):
        return None
    return float(t)


def parse_and_verify(
    instance: GSM8KInstance, step_bodies: list[str], answer_body: str | None
) -> tuple[bool, dict]:
    extracted = _extract_number(answer_body)
    # well_formed here means "structurally parseable enough to grade" only --
    # this verifier never re-derives step_bodies' arithmetic (see module
    # docstring: no fixed grammar exists to check natural-language reasoning
    # against, unlike countdown/prosqa/connect_four's verifiers).
    well_formed = bool(step_bodies) and extracted is not None
    if not well_formed:
        reason = "no parseable final answer" if extracted is None else "no step lines"
        return False, {"well_formed": False, "reason": reason}
    is_correct = math.isclose(extracted, instance.answer_value, rel_tol=1e-6, abs_tol=1e-6)
    reason = "ok" if is_correct else f"wrong answer: {extracted} != {instance.answer_value}"
    return is_correct, {"well_formed": True, "reason": reason}
