"""Program-synthesis search engine over the real, vendored arc-dsl -- Stage
1 of the ARC-AGI plan. A `ProgramState` is a per-index-aligned set of typed
"context" values (grid, object, integer, closure, ...), one context tuple
per training example, since one program applies to every example in
lockstep (exactly how `solvers.py`'s real `solve_X(I)` functions work --
called once per example with that example's own input substituted for `I`).

Two states are the SAME for merge purposes if they hold the same set of
(type, per-example-values) entries, regardless of the order those entries
were produced in (`canonical_key`) -- the direct ARC analogue of
Blocksworld's frozenset-of-facts state, and of a real transposition here:
two different primitive orderings landing on the same available values.

A context only ever grows -- no primitive removes a value -- but this does
NOT make the search cycle-free: `canonical_key` is a *set* of the context's
(type, value) entries, and a move whose result duplicates a value already
present doesn't grow that set at all, making the "new" node identical to
its own parent. Confirmed directly (a hung real run, traced via SIGINT to
`select()` spinning through a one-node `children` dict forever), not
assumed away -- `classical_mcts_arc_program.py`'s `select()` needs the same
cycle-safe path restriction every reversible domain in this project uses.

Branching factor is controlled by (a) restricting to a curated subset of
the 160 real primitives (the ~60 most-used across all 400 real reference
solutions -- confirmed by frequency count, not guessed) and (b) sampling a
bounded number of valid (function, argument) combinations per call to
`legal_moves` rather than exhaustively enumerating them -- an explicit,
documented tradeoff, not silent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import arc_types_registry as reg
from .arc_types_registry import arc_constants, call_base, dsl, to_callable

CLOSURE_BUILDERS = frozenset({"lbind", "rbind", "compose", "chain", "fork", "power", "matcher"})

# (callable-argument positions, return type) for higher-order functions that
# CONSUME a callable and produce a concrete result (not build a new closure).
# Several of these are genuinely loosely-typed in the real DSL (their exact
# output shape depends on runtime content, not just static argument types) --
# tagged ANY where that's the case, a documented simplification, not a guess
# dressed up as precision.
CLOSURE_CONSUMERS = {
    "apply": ([0], frozenset({"ANY"})),
    "mapply": ([0], frozenset({"ANY"})),
    "mfilter": ([1], frozenset({"ANY"})),
    "sfilter": ([1], frozenset({"ANY"})),
    "extract": ([1], frozenset({"ANY"})),
    "argmax": ([1], frozenset({"ANY"})),
    "argmin": ([1], frozenset({"ANY"})),
    "valmax": ([1], frozenset({"INTEGER"})),
    "valmin": ([1], frozenset({"INTEGER"})),
    "order": ([1], frozenset({"ANY"})),
    "papply": ([0], frozenset({"ANY"})),
    "mpapply": ([0], frozenset({"ANY"})),
}

HIGHER_ORDER_IN_USE = CLOSURE_BUILDERS | frozenset(CLOSURE_CONSUMERS)

# The closure-builders' OWN parameter signatures -- i.e. what they need as
# arguments to be *constructed* (real dsl.py signatures: `lbind(function:
# Callable, fixed: Any) -> Callable`, etc.), NOT what the closure they
# produce will itself need when later called (that's computed dynamically
# by `arc_types_registry.build_closure`). Two genuinely different things
# this project's own code has to track, since arc-dsl's bare `Callable`
# hints don't distinguish them either.
_CALLABLE = frozenset({"CALLABLE"})
_ANY = frozenset({"ANY"})
_INTEGER = frozenset({"INTEGER"})
CLOSURE_BUILDER_OWN_SIGNATURES = {
    "lbind": (_CALLABLE, _ANY),
    "rbind": (_CALLABLE, _ANY),
    "compose": (_CALLABLE, _CALLABLE),
    "chain": (_CALLABLE, _CALLABLE, _CALLABLE),
    "fork": (_CALLABLE, _CALLABLE, _CALLABLE),
    "power": (_CALLABLE, _INTEGER),
    "matcher": (_CALLABLE, _ANY),
}

# The ~60 most-used real dsl.py functions across all 400 arc-dsl reference
# solutions (confirmed by frequency count over solvers.py, not guessed).
CURATED_FUNCTIONS = (
    "fork", "compose", "fill", "lbind", "objects", "rbind", "mapply", "ofcolor",
    "apply", "paint", "chain", "astuple", "branch", "first", "sfilter", "merge",
    "replace", "shift", "colorfilter", "argmax", "subgrid", "canvas", "matcher",
    "difference", "leastcolor", "vconcat", "crop", "mfilter", "combine", "hconcat",
    "interval", "underfill", "shoot", "vmirror", "width", "subtract", "ulcorner",
    "remove", "hmirror", "sizefilter", "height", "extract", "insert", "equality",
    "asobject", "upscale", "order", "size", "decrement", "cover", "shape", "argmin",
    "rot90", "add", "normalize", "color", "power", "asindices", "papply", "mostcolor",
)


def _param_tags_for(func: str) -> tuple:
    """What a call to `func` itself needs as arguments (for legal_moves
    grounding) -- NOT what a closure it might construct will later need."""
    if func in reg.BASE_SIGNATURES:
        return reg.BASE_SIGNATURES[func].param_tags
    if func in CLOSURE_BUILDER_OWN_SIGNATURES:
        return CLOSURE_BUILDER_OWN_SIGNATURES[func]
    if func in CLOSURE_CONSUMERS:
        callable_pos, _return_tag = CLOSURE_CONSUMERS[func]
        import inspect
        n_params = len(inspect.signature(getattr(dsl, func)).parameters)
        return tuple(_CALLABLE if i in callable_pos else _ANY for i in range(n_params))
    raise ValueError(f"no signature known for {func!r}")


def _constants_by_type() -> dict:
    by_type = {}
    for name in dir(arc_constants):
        if name.startswith("_"):
            continue
        value = getattr(arc_constants, name)
        if isinstance(value, bool):
            tag = "BOOLEAN"
        elif isinstance(value, int):
            tag = "INTEGER"
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
            tag = "INTEGERTUPLE"
        else:
            continue
        by_type.setdefault(tag, []).append(name)
    return by_type


CONSTANTS_BY_TYPE = _constants_by_type()


@dataclass(frozen=True)
class ProgramState:
    contexts: tuple  # tuple[tuple[value, ...], ...] -- N per-example context tuples, index-aligned
    type_schema: tuple  # tuple[frozenset[str], ...] -- one tag set per index, shared across examples


def create_initial_state(train_inputs: tuple) -> ProgramState:
    """`train_inputs`: one grid per training example. Index 0 of the context
    is always the raw input grid, matching `I` in a real solvers.py program."""
    contexts = tuple((grid,) for grid in train_inputs)
    return ProgramState(contexts=contexts, type_schema=(frozenset({"GRID"}),))


def _tag_matches(candidate_tags: frozenset, needed_tags: frozenset) -> bool:
    if "ANY" in candidate_tags or "ANY" in needed_tags:
        return True
    return bool(candidate_tags & needed_tags)


def _candidate_args_for_param(state: ProgramState, needed_tags: frozenset, rng: random.Random):
    """Yields ('ctx', i) / ('const', name) / ('fnref', name) options whose
    type matches `needed_tags`."""
    options = []
    if "CALLABLE" in needed_tags:
        for i, tag in enumerate(state.type_schema):
            if "CALLABLE" in tag:
                options.append(("ctx", i))
        for fname in CURATED_FUNCTIONS:
            if fname in reg.BASE_SIGNATURES and reg.BASE_SIGNATURES[fname].param_tags:
                options.append(("fnref", fname))
        return options
    for i, tag in enumerate(state.type_schema):
        if _tag_matches(tag, needed_tags):
            options.append(("ctx", i))
    for concrete_tag in needed_tags:
        for name in CONSTANTS_BY_TYPE.get(concrete_tag, []):
            options.append(("const", name))
    return options


def legal_moves(state: ProgramState, rng: random.Random, sample_size: int = 25) -> list:
    """A bounded RANDOM SAMPLE of valid (function, arg_specs) moves, not an
    exhaustive enumeration -- branching factor with the full curated
    function set and a growing context would otherwise explode. See this
    module's docstring."""
    moves = set()  # deduplicates; ORDER comes from the sort below, never from the set
    attempts = 0
    max_attempts = sample_size * 20
    funcs = list(CURATED_FUNCTIONS)
    while len(moves) < sample_size and attempts < max_attempts:
        attempts += 1
        func = rng.choice(funcs)
        arg_specs = []
        ok = True
        for needed_tags in _param_tags_for(func):
            options = _candidate_args_for_param(state, needed_tags, rng)
            if not options:
                ok = False
                break
            arg_specs.append(rng.choice(options))
        if not ok:
            continue
        moves.add((func, tuple(arg_specs)))
    # Sorted, not `list(moves)`. A set of tuples containing strings iterates
    # in hash order, and Python randomizes string hashes per process
    # (PYTHONHASHSEED), so `list(moves)` made every ARC search depend on
    # which process it ran in: identical seeds gave different move orders,
    # hence different `rng.choice` picks and different `rng.shuffle` results
    # in `simulate`. Confirmed directly -- the same state and seed under two
    # hash seeds returned two different orderings, and two runs of the same
    # experiment config disagreed on 2 of 12 per-task solve rates.
    # `key=repr` because arg_specs mix ('ctx', int) with ('const', str),
    # which are not mutually orderable.
    return sorted(moves, key=repr)


def apply_move(state: ProgramState, move) -> ProgramState:
    func, arg_specs = move
    new_contexts = []
    new_value_tag = None
    for ex_idx, ctx in enumerate(state.contexts):
        args = []
        for kind, val in arg_specs:
            if kind == "ctx":
                args.append(ctx[val])
            elif kind == "const":
                args.append(getattr(arc_constants, val))
            elif kind == "fnref":
                args.append(val)
        if func in CLOSURE_BUILDERS:
            closure = reg.build_closure(func, *args)
            new_value = closure
            new_value_tag = closure.return_tag
        elif func in CLOSURE_CONSUMERS:
            callable_pos, return_tag = CLOSURE_CONSUMERS[func]
            call_args = list(args)
            for pos in callable_pos:
                call_args[pos] = to_callable(call_args[pos])
            new_value = getattr(dsl, func)(*call_args)
            new_value_tag = return_tag
        else:
            new_value = call_base(func, *args)
            sig = reg.BASE_SIGNATURES[func]
            # Several real dsl.py functions are "shape-preserving polymorphic"
            # (e.g. hmirror/vmirror/dmirror/cmirror: Piece -> Piece, the same
            # Grid|Object|Indices union on both sides) -- their bare type hint
            # can't say "output type == input type", so the hint alone widens
            # a Grid-in-Grid-out call to the full 3-way union. Narrowed here to
            # whatever the actual argument's own (already-known) tag was,
            # whenever the function is single-parameter and its declared
            # parameter type equals its declared return type -- a real,
            # general refinement (not per-function hardcoding), not a guess.
            if (len(sig.param_tags) == 1 and sig.param_tags[0] == sig.return_tag
                    and arg_specs[0][0] == "ctx"):
                new_value_tag = state.type_schema[arg_specs[0][1]]
            else:
                new_value_tag = sig.return_tag
        new_contexts.append(ctx + (new_value,))
    return ProgramState(contexts=tuple(new_contexts), type_schema=state.type_schema + (new_value_tag,))


def goal_index(state: ProgramState, target_outputs: tuple) -> int | None:
    """The lowest context index whose per-example values are exactly the
    target outputs, or None if no index matches.

    Checked at EVERY index, not just the last one, so that goal-ness is a
    function of the merge key. `canonical_key` is order-independent, so a
    state holding the target at index i and one holding it at index j can
    share a key; a last-value-only test would therefore depend on
    information the key does not carry, while the merge arm caches
    `is_terminal` from whichever ordering created the node first.

    That inconsistency is real but, as it stands, **unreachable** -- this is
    an invariant repair, not a bug fix, and it was checked rather than
    assumed. A state holding the target at a non-final index can only be
    produced by expanding its own prefix, and that prefix holds the target
    at *its* last index, so even the old test marks the prefix terminal and
    gives it no untried moves. It is never expanded, so the non-final-goal
    state is never created. Measured: over 60k node creations across all 32
    curated tasks, both goal tests agreed every time, and no experiment
    result moved.

    It is kept because it costs nothing (see `last_value_is_goal` for the
    hot path), because the invariant should not depend on an expansion-order
    argument holding forever, and because it is the right test on its own
    terms: a program that produced the target output has solved the task
    whether or not it computed something else afterwards. The same principle
    applied to `guct_uniform_arc._arc_heuristic`, where the equivalent
    inconsistency IS reachable, did change results.
    """
    for i in range(len(state.type_schema)):
        if all(ctx[i] == target for ctx, target in zip(state.contexts, target_outputs)):
            return i
    return None


def is_goal(state: ProgramState, target_outputs: tuple) -> bool:
    return goal_index(state, target_outputs) is not None


def last_value_is_goal(state: ProgramState, target_outputs: tuple) -> bool:
    """Whether the MOST RECENTLY produced value matches every target -- the
    O(1)-index incremental form of `is_goal`.

    Sound wherever the caller already knows the state's other indices do
    not match, which is exactly the situation inside a rollout: the walk
    checks after every step, `apply_move` appends exactly one value, so
    goal-ness can only newly arise from that value. Using the full
    `is_goal` scan per step instead is equivalent but costs O(context
    length) grid comparisons on every step of every rollout, which is a
    large constant on deep nodes."""
    return all(ctx[-1] == target for ctx, target in zip(state.contexts, target_outputs))


def last_value_progress(state: ProgramState, target_outputs: tuple) -> float:
    """`goal_progress` restricted to the most recently produced value. The
    running maximum over an incrementally extended walk is exactly
    `goal_progress`, since each step adds one index and the max over all
    indices is the max of (max so far, this index)."""
    return sum(
        grid_overlap_fraction(ctx[-1], target)
        for ctx, target in zip(state.contexts, target_outputs)
    ) / len(state.contexts)


def grid_overlap_fraction(candidate, target) -> float:
    """Cell-level agreement between a candidate context value and a target
    output grid: 1.0 on an exact match of any value type, else the fraction
    of matching cells if the two are same-shape grids, else 0.0. No
    shape-mismatch metric is invented -- a wrong-shape value scores 0, the
    same as a non-grid one -- so the only partial credit given is the part
    that is unambiguous to measure.

    Moved here from `guct_uniform_arc._grid_overlap_fraction`, which now
    delegates to it, so the graded ARC signal has one implementation.
    Real bug it already carries, caught during that module's calibration:
    checking only that `candidate[0]` is a tuple isn't enough to confirm
    candidate is a Grid (Tuple[Tuple[Integer]]) -- other DSL types also
    look tuple-of-tuples in their first element (e.g. a jagged structure
    whose row 0 happens to be a nested tuple but a later "row" is a bare
    int), and that crashed `zip(r1, r2)` with 'int is not iterable' the
    first time a non-Grid tuple value reached the function. Every row is
    validated as its own tuple, and every row's length is checked against
    target's corresponding row, not just row 0's."""
    if candidate == target:
        return 1.0
    if not isinstance(candidate, tuple) or not candidate:
        return 0.0
    if not all(isinstance(row, tuple) for row in candidate):
        return 0.0
    # `target` gets the same validation as `candidate`. Real ARC targets are
    # always grids, so this is unreachable today, but this is now a shared
    # engine helper with two callers and a docstring that advertises
    # robustness -- `len(target)` on a non-grid would raise TypeError.
    if not isinstance(target, tuple) or not all(isinstance(row, tuple) for row in target):
        return 0.0
    if len(candidate) != len(target) or any(len(r1) != len(r2) for r1, r2 in zip(candidate, target)):
        return 0.0
    total = sum(len(row) for row in candidate)
    matches = sum(1 for r1, r2 in zip(candidate, target) for c1, c2 in zip(r1, r2) if c1 == c2)
    return matches / total


def goal_progress(state: ProgramState, target_outputs: tuple) -> float:
    """Graded counterpart to `is_goal`: the best over context indices of
    the mean per-example similarity to that example's target output. Equals
    1.0 exactly when `is_goal` is True.

    Both axes the ARC verifier literature uses are present: the mean over
    examples is "fraction of training examples matched" when every score is
    0 or 1, and `grid_overlap_fraction` supplies the within-example
    cell-level term. The motivation is that binary exact-match is the
    sparsest reward signal of any domain in this repo, and merging works by
    pooling statistics -- pooling zeros carries no information, which is a
    candidate explanation for ARC showing this project's smallest merge
    effect.

    Maxed over context indices rather than read off `ctx[-1]`, for the same
    reason `goal_index` scans every index: anything the merge key cannot
    see must not change a state's value, or the merge and tree arms are
    scoring different things."""
    n = len(state.contexts)
    best = 0.0
    for i in range(len(state.type_schema)):
        score = sum(
            grid_overlap_fraction(ctx[i], target)
            for ctx, target in zip(state.contexts, target_outputs)
        ) / n
        best = max(best, score)
    return best


def canonical_key(state: ProgramState, key_mode: str = "typed"):
    """Order-independent merge key -- two states are the same if they hold
    the same entries, however they were produced. The direct ARC analogue
    of Blocksworld's frozenset state.

    `key_mode` selects the equivalence relation:

    * "typed" (default, original behavior): entries are
      (type-tag-set, per-example-values) pairs.
    * "values": entries are per-example-value tuples alone. Strictly
      coarser, and closer to observational equivalence as bottom-up program
      synthesis means it -- two states holding the same concrete values are
      interchangeable for everything observable, whereas "typed" keeps them
      apart whenever the same value was reached under different static tags
      (which happens here: `apply_move`'s return-tag refinement assigns
      different tag sets to the same value depending on the route taken,
      and the higher-order consumers are tagged ANY). The approximation
      this introduces is that the surviving node keeps the first arrival's
      `type_schema`, so type-driven `legal_moves` sampling downstream is
      the representative's, not the newcomer's -- the standard
      representative-parent merge approximation, measured rather than
      assumed.
    """
    if key_mode == "typed":
        return frozenset(
            (tag, tuple(ctx[i] for ctx in state.contexts))
            for i, tag in enumerate(state.type_schema)
        )
    if key_mode == "values":
        return frozenset(
            tuple(ctx[i] for ctx in state.contexts)
            for i in range(len(state.type_schema))
        )
    raise ValueError(f"unknown key_mode {key_mode!r}")
