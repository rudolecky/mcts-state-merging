"""Countdown puzzle generator + verifier.

Rule variant (locked, enforced identically in generation and verification):
- Each starting number is used exactly once.
- Operators: +, -, *, /.
- Intermediate (and final) results must be positive integers.
- Subtraction must stay positive; division only when exact.

Distinct-solution counting treats a solution as a canonical bracketed
expression string over the starting numbers. Commutative operators (+, *)
canonicalize operand order (so a+b and b+a count once); subtraction and
division are directional and counted separately. This is a path-multiplicity
count over *derivations* (parenthesization/order matters), not just distinct
final values -- deliberately, since different derivations are different
reasoning traces even when arithmetically equivalent by associativity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

from .common import answer_line, step_line

OPS = ("+", "-", "*", "/")


def _combine(v1: int, e1: str, v2: int, e2: str, op: str) -> tuple[int, str] | None:
    if op == "+":
        a, b = _canon_order(e1, e2)
        return v1 + v2, f"({a}+{b})"
    if op == "*":
        a, b = _canon_order(e1, e2)
        return v1 * v2, f"({a}*{b})"
    if op == "-":
        if v1 - v2 <= 0:
            return None
        return v1 - v2, f"({e1}-{e2})"
    if op == "/":
        if v2 == 0 or v1 % v2 != 0:
            return None
        return v1 // v2, f"({e1}/{e2})"
    raise ValueError(op)


def _canon_order(e1: str, e2: str) -> tuple[str, str]:
    """Canonicalize operand order for commutative ops by lexicographic expr string."""
    return (e1, e2) if e1 <= e2 else (e2, e1)


def _all_expressions(numbers: tuple[int, ...]) -> dict[int, set[str]]:
    """All (value -> set of canonical derivation strings) using every number exactly once."""
    n = len(numbers)

    @lru_cache(maxsize=None)
    def solve(indices: tuple[int, ...]) -> dict[int, frozenset[str]]:
        if len(indices) == 1:
            v = numbers[indices[0]]
            return {v: frozenset({str(v)})}
        results: dict[int, set[str]] = {}
        idx_set = indices
        # iterate over every non-empty proper subset as the "left" split;
        # the complementary subset is visited separately (as its own `indices`
        # tuple elsewhere in the recursion), which covers the reversed
        # direction for non-commutative ops without extra bookkeeping here.
        for r in range(1, len(idx_set)):
            for left in combinations(idx_set, r):
                right = tuple(i for i in idx_set if i not in left)
                left_res = solve(tuple(sorted(left)))
                right_res = solve(tuple(sorted(right)))
                for v1, exprs1 in left_res.items():
                    for v2, exprs2 in right_res.items():
                        for e1 in exprs1:
                            for e2 in exprs2:
                                for op in OPS:
                                    combined = _combine(v1, e1, v2, e2, op)
                                    if combined is None:
                                        continue
                                    rv, rexpr = combined
                                    results.setdefault(rv, set()).add(rexpr)
        return {k: frozenset(v) for k, v in results.items()}

    full = solve(tuple(range(n)))
    return {k: set(v) for k, v in full.items()}


@dataclass(frozen=True)
class CountdownInstance:
    numbers: tuple[int, ...]
    target: int
    path_count: int
    example_solutions: tuple[str, ...] = field(default_factory=tuple)

    @property
    def id(self) -> str:
        return f"cd_{'-'.join(map(str, self.numbers))}_t{self.target}"


def generate_instance(
    rng: random.Random,
    num_numbers: int,
    number_low: int,
    number_high: int,
    max_examples: int = 3,
) -> CountdownInstance | None:
    numbers = tuple(rng.randint(number_low, number_high) for _ in range(num_numbers))
    expr_map = _all_expressions(numbers)
    # a "trivial" solution is just one of the starting numbers on its own,
    # with no combination at all -- exclude those from both the candidate
    # pool and the path-count/example-solutions of whichever target is picked.
    nontrivial: dict[int, list[str]] = {}
    for v, exprs in expr_map.items():
        real = [e for e in exprs if not e.lstrip("-").isdigit()]
        if real:
            nontrivial[v] = real
    if not nontrivial:
        return None
    target = rng.choice(list(nontrivial))
    solutions = tuple(sorted(nontrivial[target]))
    return CountdownInstance(
        numbers=numbers,
        target=target,
        path_count=len(solutions),
        example_solutions=solutions[:max_examples],
    )


def _parse_expr(expr: str) -> tuple[int, list[str]]:
    """Recursively evaluate a canonical bracketed expression, returning its
    value and the bottom-up list of step bodies ("a op b = result") that
    derive it. Leaves are non-negative integer literals; every intermediate
    result is positive per the generation rules, so no unary-minus case
    can arise at a top level operator scan.
    """
    expr = expr.strip()
    if expr.isdigit():
        return int(expr), []
    assert expr[0] == "(" and expr[-1] == ")", f"malformed expression: {expr!r}"
    inner = expr[1:-1]
    depth = 0
    op_idx = None
    op_char = None
    for i, ch in enumerate(inner):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and ch in "+-*/":
            op_idx, op_char = i, ch
            break
    if op_idx is None:
        raise ValueError(f"no top-level operator found in {expr!r}")
    v1, steps1 = _parse_expr(inner[:op_idx])
    v2, steps2 = _parse_expr(inner[op_idx + 1 :])
    if op_char == "+":
        result = v1 + v2
    elif op_char == "*":
        result = v1 * v2
    elif op_char == "-":
        result = v1 - v2
    else:
        result = v1 // v2
    return result, steps1 + steps2 + [f"{v1} {op_char} {v2} = {result}"]


def expr_to_steps(expr: str) -> list[str]:
    """Convert one canonical solution string into ordered step-body strings,
    suitable for `common.step_line` / feeding straight into `parse_and_verify`.
    """
    _, steps = _parse_expr(expr)
    return steps


def generate_stratified(
    n_low: int,
    n_high: int,
    seed: int = 0,
    num_numbers: int = 4,
    number_low: int = 1,
    number_high: int = 15,
    low_threshold: int = 2,
    exclude_ids: set[str] | None = None,
) -> list[CountdownInstance]:
    """Generate instances stratified into low-path-count (<= low_threshold)
    and high-path-count (> low_threshold) buckets, n_low and n_high of each.

    `exclude_ids` gives a mechanical guarantee of zero overlap with a
    previously-generated set (e.g. problems already used to fit a learned
    metric) -- checked against `.id`, which is a different key than the
    internal `seen` dedup set (keyed by (numbers, target)), so it can't just
    seed `seen` directly.
    """
    exclude_ids = exclude_ids or set()
    rng = random.Random(seed)
    seen: set[tuple[tuple[int, ...], int]] = set()
    low_bucket: list[CountdownInstance] = []
    high_bucket: list[CountdownInstance] = []
    attempts = 0
    max_attempts = 200 * (n_low + n_high) + 1000
    while (len(low_bucket) < n_low or len(high_bucket) < n_high) and attempts < max_attempts:
        attempts += 1
        inst = generate_instance(rng, num_numbers, number_low, number_high)
        if inst is None:
            continue
        key = (inst.numbers, inst.target)
        if key in seen:
            continue
        seen.add(key)
        if inst.id in exclude_ids:
            continue
        if inst.path_count <= low_threshold and len(low_bucket) < n_low:
            low_bucket.append(inst)
        elif inst.path_count > low_threshold and len(high_bucket) < n_high:
            high_bucket.append(inst)
    return low_bucket + high_bucket


def build_prompt(instance: CountdownInstance) -> str:
    example = (
        "Example:\n"
        "Numbers: [2, 3, 4]. Target: 20.\n"
        "Step 1: 4 * 3 = 12\n"
        "Step 2: 12 + 2 = 14\n"
        "Answer: 14\n\n"
        "(That example fell short of its target -- always keep combining "
        "until exactly one number remains, and report that number.)\n\n"
    )
    body = (
        f"Numbers: {list(instance.numbers)}. Target: {instance.target}.\n"
        "Use each number exactly once. Combine two numbers at a time with "
        "+, -, *, or /, writing one combination per line as\n"
        "Step N: a OP b = result\n"
        "Keep combining until exactly one number is left, then write\n"
        "Answer: <that number>\n"
    )
    return example + body


def parse_and_verify(
    instance: CountdownInstance, step_bodies: list[str], answer_body: str | None
) -> tuple[bool, dict]:
    """Verify a parsed trace's derivation and final answer against the rules.

    Returns (is_correct, info) where info carries a well_formed flag and any
    parse failure reason, for debugging/logging.
    """
    import re as _re

    # trailing period tolerated: a formatting artifact of the model imitating
    # prose, not a reasoning error (same rationale as in prosqa's verifier).
    line_re = _re.compile(r"^\s*(-?\d+)\s*([+\-*/])\s*(-?\d+)\s*=\s*(-?\d+)\s*\.?\s*$")
    available: dict[int, int] = {}
    for n in instance.numbers:
        available[n] = available.get(n, 0) + 1

    def take(v: int) -> bool:
        if available.get(v, 0) <= 0:
            return False
        available[v] -= 1
        return True

    for body in step_bodies:
        m = line_re.match(body)
        if not m:
            return False, {"well_formed": False, "reason": f"unparseable step: {body!r}"}
        a, op, b, claimed = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))
        if not take(a):
            return False, {"well_formed": False, "reason": f"operand {a} not available"}
        if not take(b):
            available[a] = available.get(a, 0) + 1  # restore before failing
            return False, {"well_formed": False, "reason": f"operand {b} not available"}
        if op == "+":
            actual = a + b
        elif op == "*":
            actual = a * b
        elif op == "-":
            if a - b <= 0:
                return False, {"well_formed": False, "reason": "subtraction non-positive"}
            actual = a - b
        elif op == "/":
            if b == 0 or a % b != 0:
                return False, {"well_formed": False, "reason": "division not exact"}
            actual = a // b
        else:
            return False, {"well_formed": False, "reason": f"bad op {op}"}
        if actual != claimed:
            return False, {"well_formed": False, "reason": f"arithmetic error: {a}{op}{b}!={claimed}"}
        available[actual] = available.get(actual, 0) + 1

    remaining = [v for v, c in available.items() for _ in range(c)]
    well_formed = len(remaining) == 1 and remaining[0] == instance.target
    if answer_body is None:
        return False, {"well_formed": well_formed, "reason": "no answer line"}
    try:
        claimed_answer = int(answer_body.strip().rstrip("."))
    except ValueError:
        return False, {"well_formed": well_formed, "reason": f"unparseable answer: {answer_body!r}"}
    is_correct = well_formed and claimed_answer == instance.target
    return is_correct, {"well_formed": well_formed, "reason": "ok" if is_correct else "answer/derivation mismatch"}
