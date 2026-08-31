"""GSM8K probed at a reasoning-RL model's OWN natural boundaries, instead of
this project's usual `Step N:`/`Answer:` line convention.

Why this exists: three literature-grounded hypotheses were tested for why
gsm8k.py's within-problem value probe never replicated on Qwen2.5-7B-Instruct
(see results/GSM8K_FINDINGS.md) -- among them, that the self-verification
geometry found in "Reasoning Models Know When They're Right" (NYU,
arXiv:2504.05419) was demonstrated specifically on reasoning-RL-trained
models (DeepSeek-R1-Distill, QwQ), not plain instruction-tuned ones. Testing
that directly by just swapping the model produced worse raw accuracy and
5-10x slower generation than the instruct model got -- because forcing a
model trained to think in free-form `<think>...</think>` prose into discrete
`Step N:` lines fights its trained behavior, rather than genuinely testing
the hypothesis.

This module drops that forced format entirely: `build_prompt` asks only for
a `\\boxed{...}` final answer (the standard DeepSeek-R1 training/eval
convention the model already complies with, not an invented phrase),
nothing about intermediate structure. Snapshot boundaries then come from
`model.py`'s existing `find_position_based_boundaries` fallback (which
already fires automatically whenever zero "Step N:" lines are found -- no
changes needed there), not from parsing content.

Reuses `GSM8KInstance`/`generate_sample`/`_extract_number` from gsm8k.py
directly -- same underlying question/answer data and `.id` scheme, only the
prompt and parsing contract differ.
"""

from __future__ import annotations

import math
import re

from .gsm8k import GSM8KInstance, _extract_number, generate_sample  # noqa: F401  (re-exported)

_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_IS_RE = re.compile(r"(?:final answer is|answer is|final answer:?)\s*:?\s*(\S+)", re.IGNORECASE)


def build_prompt(instance: GSM8KInstance) -> str:
    return (
        f"{instance.question}\n\n"
        "Put your final answer as a single number in \\boxed{}."
    )


def native_split(text: str) -> tuple[list[str], str | None]:
    """No per-step structure exists in this mode by construction --
    step_bodies is always [] (snapshot boundaries come from
    find_position_based_boundaries, not from parsing content here).
    answer_body is the last \\boxed{...} in the text, falling back to an
    "answer is X" pattern for the rare sample that doesn't use \\boxed{}.
    """
    boxed_matches = _BOXED_RE.findall(text)
    if boxed_matches:
        return [], boxed_matches[-1].strip()
    last_match = None
    for last_match in _ANSWER_IS_RE.finditer(text):
        pass
    if last_match is not None:
        return [], last_match.group(1).strip()
    return [], None


def parse_and_verify(
    instance: GSM8KInstance, step_bodies: list[str], answer_body: str | None
) -> tuple[bool, dict]:
    """Cannot reuse gsm8k.parse_and_verify as-is: its `well_formed =
    bool(step_bodies) and ...` check would always be False here, since
    step_bodies is always [] by construction in this mode (no per-step
    structure exists to check). well_formed here means only "a final answer
    was found," not "the response had discrete steps."
    """
    extracted = _extract_number(answer_body)
    well_formed = extracted is not None
    if not well_formed:
        return False, {"well_formed": False, "reason": "no \\boxed{} or 'answer is' found"}
    is_correct = math.isclose(extracted, instance.answer_value, rel_tol=1e-6, abs_tol=1e-6)
    reason = "ok" if is_correct else f"wrong answer: {extracted} != {instance.answer_value}"
    return is_correct, {"well_formed": True, "reason": reason}
