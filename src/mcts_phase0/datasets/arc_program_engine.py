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
    moves = set()
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
    return list(moves)


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


def is_goal(state: ProgramState, target_outputs: tuple) -> bool:
    return all(ctx[-1] == target for ctx, target in zip(state.contexts, target_outputs))


def canonical_key(state: ProgramState):
    """Order-independent merge key: two states are the same if they hold
    the same set of (type, per-example-values) entries, however they were
    produced -- the direct ARC analogue of Blocksworld's frozenset state."""
    return frozenset(
        (tag, tuple(ctx[i] for ctx in state.contexts))
        for i, tag in enumerate(state.type_schema)
    )
