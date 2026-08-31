"""ProsQA-style synthetic DAG entailment reasoning generator + verifier.

Each instance presents a set of "X is a Y" facts forming a DAG (edges only
ever go from an earlier-created entity to a later-created one, which makes
the graph acyclic by construction -- see `_Universe`), then asks whether one
entity transitively implies another. Positive instances are built from a
single connected component (a main chain plus dead-end distractor branches,
and -- for a fraction of instances -- one reconverging shortcut edge that
creates a second genuine derivation path). Negative instances pair entities
from two disjoint components, so no path can exist.

Path-count is computed generically via DAG dynamic programming over
whatever facts are actually shown, not assumed from the construction
recipe -- this doubles as a check that the generator does what it intends.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

_WORD_BANK = (
    "wampus", "yumpus", "rompus", "zhorpus", "dumpus", "vumpus", "tumpus",
    "impus", "numpus", "gorpus", "shumpus", "jelpus", "brimpus", "quimpus",
    "florpus", "clanpus", "drampus", "wibpus", "snorpus", "plimpus",
)


class _Universe:
    """Tracks a strictly increasing entity-creation order so every edge we
    add is guaranteed forward (creation_order[u] < creation_order[v]),
    making the resulting graph acyclic by construction.
    """

    def __init__(self, rng: random.Random):
        self._rng = rng
        self._counter = 0
        self._used_names: set[str] = set()

    def fresh_entity(self) -> str:
        while True:
            name = f"{self._rng.choice(_WORD_BANK)}{self._counter}"
            self._counter += 1
            if name not in self._used_names:
                self._used_names.add(name)
                return name


def _path_count(facts: list[tuple[str, str]], start: str, target: str) -> int:
    """Count distinct directed paths start -> target over the given edges."""
    successors: dict[str, list[str]] = {}
    for u, v in facts:
        successors.setdefault(u, []).append(v)
    memo: dict[str, int] = {}

    def count(node: str, visiting: frozenset[str]) -> int:
        if node == target:
            return 1
        if node in memo:
            return memo[node]
        total = 0
        for nxt in successors.get(node, []):
            if nxt in visiting:
                continue  # defensive; construction guarantees acyclicity anyway
            total += count(nxt, visiting | {nxt})
        memo[node] = total
        return total

    return count(start, frozenset({start}))


def _build_component(
    rng: random.Random,
    universe: _Universe,
    chain_len: int,
    num_distractors: int,
    reconverge: bool,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Build one component: a main chain of `chain_len` edges, plus dead-end
    distractor branches and (if reconverge) one shortcut edge creating a
    second start->end path. Returns (facts, chain_entities).
    """
    chain = [universe.fresh_entity() for _ in range(chain_len + 1)]
    facts: list[tuple[str, str]] = [(chain[i], chain[i + 1]) for i in range(chain_len)]

    for _ in range(num_distractors):
        branch_point = rng.randint(0, chain_len - 1)  # not the last node
        dead_end = universe.fresh_entity()
        facts.append((chain[branch_point], dead_end))
        # occasionally extend the dead end one more hop so distractors read
        # as genuine (if irrelevant) reasoning chains, not obviously inert leaves
        if rng.random() < 0.5:
            dead_end_2 = universe.fresh_entity()
            facts.append((dead_end, dead_end_2))

    if reconverge and chain_len >= 2:
        j = rng.randint(2, chain_len)  # shortcut lands >=2 hops in, a genuine skip
        facts.append((chain[0], chain[j]))

    return facts, chain


@dataclass(frozen=True)
class ProsQAInstance:
    facts: tuple[tuple[str, str], ...]  # every fact shown in the prompt, shuffled order
    start: str
    target: str
    answer: str  # "yes" or "no"
    path_count: int  # distinct start->target derivations; 0 for "no" instances
    correct_path: tuple[str, ...] | None  # one example entity chain, if answer == "yes"

    @property
    def id(self) -> str:
        return f"pq_{self.start}_{self.target}_{self.answer}"


def _shortest_path(facts: list[tuple[str, str]], start: str, target: str) -> list[str] | None:
    successors: dict[str, list[str]] = {}
    for u, v in facts:
        successors.setdefault(u, []).append(v)
    from collections import deque

    q = deque([[start]])
    seen = {start}
    while q:
        path = q.popleft()
        node = path[-1]
        if node == target:
            return path
        for nxt in successors.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(path + [nxt])
    return None


def generate_dataset(
    n_positive: int,
    n_negative: int,
    seed: int = 0,
    reconverge_fraction: float = 0.4,
    chain_len_range: tuple[int, int] = (3, 5),
    num_distractors_range: tuple[int, int] = (1, 3),
) -> list[ProsQAInstance]:
    if n_negative > 0 and n_positive < 2:
        raise ValueError(
            "negative instances need at least 2 positive components to pair "
            f"across (need n_positive >= 2 when n_negative > 0; got n_positive={n_positive})"
        )

    rng = random.Random(seed)
    universe = _Universe(rng)
    instances: list[ProsQAInstance] = []

    positive_components: list[tuple[list[tuple[str, str]], list[str]]] = []
    for i in range(n_positive):
        chain_len = rng.randint(*chain_len_range)
        num_distractors = rng.randint(*num_distractors_range)
        reconverge = rng.random() < reconverge_fraction
        facts, chain = _build_component(rng, universe, chain_len, num_distractors, reconverge)
        positive_components.append((facts, chain))
        start, target = chain[0], chain[-1]
        pc = _path_count(facts, start, target)
        path = _shortest_path(facts, start, target)
        shuffled = list(facts)
        rng.shuffle(shuffled)
        instances.append(
            ProsQAInstance(
                facts=tuple(shuffled),
                start=start,
                target=target,
                answer="yes",
                path_count=pc,
                correct_path=tuple(path) if path else None,
            )
        )

    # negative instances: pair start of one component with the end of a
    # *different* component, combining both components' facts so the model
    # has to actually reason across the full shown fact set to say "no".
    for i in range(n_negative):
        a_idx, b_idx = rng.sample(range(len(positive_components)), 2)
        facts_a, chain_a = positive_components[a_idx]
        facts_b, chain_b = positive_components[b_idx]
        combined = facts_a + facts_b
        start, target = chain_a[0], chain_b[-1]
        pc = _path_count(combined, start, target)  # expect 0; assert this holds
        assert pc == 0, "negative instance unexpectedly has a cross-component path"
        shuffled = list(combined)
        rng.shuffle(shuffled)
        instances.append(
            ProsQAInstance(
                facts=tuple(shuffled),
                start=start,
                target=target,
                answer="no",
                path_count=0,
                correct_path=None,
            )
        )

    rng.shuffle(instances)
    return instances


def average_path_count(instances: list[ProsQAInstance]) -> float:
    positives = [i for i in instances if i.answer == "yes"]
    if not positives:
        return 0.0
    return sum(i.path_count for i in positives) / len(positives)


def build_prompt(instance: ProsQAInstance) -> str:
    example = (
        "Example facts:\n"
        "- foo0 is a bar1.\n"
        "- bar1 is a baz2.\n\n"
        "Question: Is every foo0 a baz2?\n"
        "Step 1: foo0 is a bar1\n"
        "Step 2: bar1 is a baz2\n"
        "Answer: yes\n\n"
    )
    facts_block = "\n".join(f"- {x} is a {y}." for x, y in instance.facts)
    body = (
        f"Facts:\n{facts_block}\n\n"
        f"Question: Is every {instance.start} a {instance.target}?\n"
        "Chain the facts above to reach an answer, one fact per line:\n"
        "Step 1: <entity> is a <entity>\n"
        "...\n"
        "Answer: yes or no\n"
        "(If no chain connects them, skip straight to the Answer line.)\n"
    )
    return example + body


def parse_and_verify(
    instance: ProsQAInstance, step_bodies: list[str], answer_body: str | None
) -> tuple[bool, dict]:
    import re as _re

    fact_set = set(instance.facts)
    # tolerate a trailing period: the Facts block in the prompt lists facts as
    # "- x is a y." so the model reliably copies that punctuation into its
    # step lines, which is a formatting artifact and not a reasoning error.
    line_re = _re.compile(r"^\s*(.+?)\s+is a\s+(.+?)\s*\.?\s*$")

    chain: list[tuple[str, str]] = []
    for body in step_bodies:
        m = line_re.match(body)
        if not m:
            return False, {"well_formed": False, "reason": f"unparseable step: {body!r}"}
        x, y = m.group(1).strip().rstrip("."), m.group(2).strip().rstrip(".")
        if (x, y) not in fact_set:
            return False, {"well_formed": False, "reason": f"fact not given: {x} is a {y}"}
        chain.append((x, y))

    if answer_body is None:
        return False, {"well_formed": False, "reason": "no answer line"}
    claimed = answer_body.strip().lower().rstrip(".")
    if claimed not in ("yes", "no"):
        return False, {"well_formed": False, "reason": f"unparseable answer: {answer_body!r}"}

    if claimed == "no":
        well_formed = True  # no derivation required to justify a "no"
    else:
        well_formed = (
            len(chain) > 0
            and chain[0][0] == instance.start
            and chain[-1][1] == instance.target
            and all(chain[i][1] == chain[i + 1][0] for i in range(len(chain) - 1))
        )

    is_correct = (claimed == instance.answer) and (instance.answer == "no" or well_formed)
    return is_correct, {"well_formed": well_formed, "reason": "ok" if is_correct else "answer/chain mismatch"}
