"""Shared step-delimiter convention used by both datasets.

model.py's step-boundary parsing is dataset-agnostic and relies on every
prompt using this exact convention: one fact/operation per line, prefixed
with "Step N:", followed by a final "Answer: ..." line.
"""

import re

STEP_LINE_RE = re.compile(r"^Step (\d+):\s*(.*)$")
ANSWER_LINE_RE = re.compile(r"^Answer:\s*(.*)$")


def step_line(step_num: int, body: str) -> str:
    return f"Step {step_num}: {body}"


def answer_line(body: str) -> str:
    return f"Answer: {body}"


def split_steps(text: str) -> tuple[list[str], str | None]:
    """Split generated trace text into step bodies plus the final answer body.

    Returns (step_bodies, answer_body). answer_body is None if no Answer
    line was found (malformed/truncated generation).
    """
    step_bodies: list[str] = []
    answer_body: str | None = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = STEP_LINE_RE.match(line)
        if m:
            step_bodies.append(m.group(2).strip())
            continue
        m = ANSWER_LINE_RE.match(line)
        if m:
            answer_body = m.group(1).strip()
            break  # ignore anything the model rambles after the answer
    return step_bodies, answer_body
