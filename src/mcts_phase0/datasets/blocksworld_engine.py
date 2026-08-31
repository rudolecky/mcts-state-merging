"""Blocksworld STRIPS planning, reusing pyperplan's own `Task`/`Operator`/
`breadth_first_search` primitives directly -- the exact planner
arXiv:2405.18248 ("Extreme Value Monte Carlo Tree Search for Classical
Planning") itself uses -- rather than a hand-rolled parallel engine. No
PDDL text is parsed: Blocksworld's grounded STRIPS encoding is standard
and unambiguous enough to generate directly as `Operator` objects.

State: a `frozenset` of STRIPS fact strings -- already canonical
(`Operator.apply` already returns a frozenset, no separate normalization
step, same "the state representation already is the key" principle every
other engine in this project follows).

Blocksworld's four actions are genuine inverse pairs (pickup/putdown,
stack/unstack), and independent block-pairs commute trivially (stacking
A-on-B then C-on-D reaches the identical world-state as C-on-D then
A-on-B) -- reaching the same state via different action orders is common
here, unlike Connect Four's irreversible drops.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from pyperplan.search import breadth_first_search
from pyperplan.task import Operator, Task


def _facts(num_blocks: int) -> set[str]:
    facts = {"handempty"}
    for b in range(1, num_blocks + 1):
        facts.add(f"ontable({b})")
        facts.add(f"clear({b})")
        facts.add(f"holding({b})")
        for c in range(1, num_blocks + 1):
            if b != c:
                facts.add(f"on({b},{c})")
    return facts


def _operators(num_blocks: int) -> list[Operator]:
    ops = []
    for b in range(1, num_blocks + 1):
        ops.append(Operator(
            f"pickup({b})",
            frozenset({f"clear({b})", f"ontable({b})", "handempty"}),
            frozenset({f"holding({b})"}),
            frozenset({f"clear({b})", f"ontable({b})", "handempty"}),
        ))
        ops.append(Operator(
            f"putdown({b})",
            frozenset({f"holding({b})"}),
            frozenset({f"ontable({b})", f"clear({b})", "handempty"}),
            frozenset({f"holding({b})"}),
        ))
        for c in range(1, num_blocks + 1):
            if b == c:
                continue
            ops.append(Operator(
                f"stack({b},{c})",
                frozenset({f"holding({b})", f"clear({c})"}),
                frozenset({f"on({b},{c})", f"clear({b})", "handempty"}),
                frozenset({f"holding({b})", f"clear({c})"}),
            ))
            ops.append(Operator(
                f"unstack({b},{c})",
                frozenset({f"on({b},{c})", f"clear({b})", "handempty"}),
                frozenset({f"holding({b})", f"clear({c})"}),
                frozenset({f"on({b},{c})", f"clear({b})", "handempty"}),
            ))
    return ops


def goal_state(num_blocks: int) -> frozenset[str]:
    """The standard Blocksworld goal shape: every block stacked in one
    tower, in numeric order (1 on the table, 2 on 1, 3 on 2, ...)."""
    facts = {"handempty", "ontable(1)"}
    if num_blocks == 1:
        facts.add("clear(1)")
    else:
        for b in range(2, num_blocks + 1):
            facts.add(f"on({b},{b - 1})")
        facts.add(f"clear({num_blocks})")
    return frozenset(facts)


def scrambled_state(num_blocks: int, rng: random.Random) -> frozenset[str]:
    """A random valid Blocksworld state. Each block, processed in a random
    order, either goes on the table or on top of an *already-processed*
    not-yet-occupied block -- restricting supports to already-placed
    blocks guarantees a forest of simple towers, never a cycle."""
    order = list(range(1, num_blocks + 1))
    rng.shuffle(order)
    on_top_of: dict[int, int | None] = {}
    occupied: set[int] = set()
    placed: list[int] = []
    for b in order:
        candidates = [c for c in placed if c not in occupied]
        if candidates and rng.random() < 0.6:
            support = rng.choice(candidates)
            on_top_of[b] = support
            occupied.add(support)
        else:
            on_top_of[b] = None
        placed.append(b)

    facts = {"handempty"}
    for b in range(1, num_blocks + 1):
        if on_top_of[b] is None:
            facts.add(f"ontable({b})")
        else:
            facts.add(f"on({b},{on_top_of[b]})")
        if b not in occupied:
            facts.add(f"clear({b})")
    return frozenset(facts)


def make_task(num_blocks: int) -> Task:
    """`initial_state`/`goals` are placeholders -- callers overwrite them
    per query (e.g. before a `breadth_first_search` call); the reusable
    part is `facts`/`operators`, which don't depend on any specific
    instance."""
    placeholder = goal_state(num_blocks)
    return Task(f"blocksworld-{num_blocks}", _facts(num_blocks), placeholder, placeholder, _operators(num_blocks))


def legal_moves(state: frozenset[str], task: Task) -> list[Operator]:
    return [op for op in task.operators if op.applicable(state)]


def apply_move(state: frozenset[str], op: Operator) -> frozenset[str]:
    return op.apply(state)


def is_goal(state: frozenset[str], goal: frozenset[str]) -> bool:
    return goal <= state


@dataclass(frozen=True)
class BlocksworldInstance:
    id: str
    start_state: frozenset[str]
    num_blocks: int
    plan_length: int


def generate_puzzles(
    n: int, seed: int, num_blocks: int, target_plan_length: int, max_attempts: int = 20_000,
) -> list[BlocksworldInstance]:
    """Seeded rejection sampling: scramble a random valid state, verify its
    *exact* optimal plan length via `pyperplan.search.breadth_first_search`
    (the same exact-difficulty-by-construction principle as every other
    domain's puzzle generator here, just reusing pyperplan's own oracle
    instead of a hand-rolled one)."""
    task = make_task(num_blocks)
    goal = goal_state(num_blocks)
    rng = random.Random(seed)
    found: list[BlocksworldInstance] = []
    seen: set[frozenset[str]] = set()
    for _ in range(max_attempts):
        if len(found) >= n:
            break
        state = scrambled_state(num_blocks, rng)
        if state in seen or is_goal(state, goal):
            continue
        seen.add(state)
        task.initial_state = state
        task.goals = goal
        plan = breadth_first_search(task)
        if plan is None or len(plan) != target_plan_length:
            continue
        found.append(BlocksworldInstance(
            id=f"bw{num_blocks}_pl{target_plan_length}_{len(found)}",
            start_state=state, num_blocks=num_blocks, plan_length=target_plan_length,
        ))
    if len(found) < n:
        raise ValueError(
            f"only found {len(found)}/{n} puzzles at num_blocks={num_blocks}, "
            f"target_plan_length={target_plan_length} within {max_attempts} attempts"
        )
    return found
