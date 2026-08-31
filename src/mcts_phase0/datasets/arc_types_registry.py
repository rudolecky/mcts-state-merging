"""A typed-signature registry layered on top of the vendored `arc-dsl`
(`src/mcts_phase0/vendor/arc_dsl/`), since that library's own type hints for
its 19 higher-order functions are bare `Callable` -- not statically
parameterized with what a Callable's own input/output types are. This
module infers/assigns that itself, mechanically for the 141 base (non-
higher-order) functions (via `typing.get_type_hints` against `arc_types.py`,
no manual work), and via hand-derived propagation rules for the 19
higher-order ones (read directly from their implementations in `dsl.py`).

Every value flowing through a search state in `arc_program_engine.py` is
tagged with its type from HERE, not inferred from Python's own runtime
representation -- deliberately, since several ARC value types are
structurally indistinguishable at runtime (`Object`, `Indices`, and
`IntegerSet` are all "a frozenset of tuples/ints" and collide for empty
sets). A value's tag is always known because it's always produced by
calling a function whose signature is in this registry.

`Closure` values (the result of `lbind`/`rbind`/`compose`/`chain`/`fork`/
`power`/`matcher`) don't reimplement those combinators -- `execute()` always
resolves down to real vendored `dsl.py` calls, so runtime *behavior* reuses
the library's own tested correctness. Only the *type bookkeeping* needed to
know which later primitives a constructed closure can feed into is this
project's own code, and it's the highest-risk part of this whole ARC
pilot -- verified in `tests/test_arc_types_registry.py` by replaying real
`solvers.py` reference programs through this exact machinery and checking
every intermediate value against calling the real functions directly.
"""

from __future__ import annotations

import inspect
import os
import sys
import typing
from dataclasses import dataclass

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "..", "vendor", "arc_dsl")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

import arc_types  # noqa: E402
import constants as arc_constants  # noqa: E402
import dsl  # noqa: E402

HIGHER_ORDER_NAMES = frozenset({
    "apply", "argmax", "argmin", "chain", "compose", "extract", "fork", "lbind",
    "mapply", "matcher", "mfilter", "mpapply", "order", "papply", "power", "rbind",
    "sfilter", "valmax", "valmin",
})


# ---------- structural type resolution ----------

def resolve_type_tag(hint) -> frozenset:
    """Maps a resolved `typing` hint to a set of atomic tags (a union type
    resolves to more than one tag). Verified against all 16 of arc_types.py's
    own named aliases and all 141 base functions' real signatures in
    tests/test_arc_types_registry.py -- not just against the small set of
    constructs anticipated ahead of time."""
    if hint is bool:
        return frozenset({"BOOLEAN"})
    if hint is int:
        return frozenset({"INTEGER"})
    if hint is typing.Any:
        return frozenset({"ANY"})
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin is typing.Union:
        out = set()
        for a in args:
            out |= resolve_type_tag(a)
        return frozenset(out)
    if hint is typing.Tuple or origin is tuple:
        if not args:
            return frozenset({"ANY"})  # bare, unparameterized Tuple
        if args == (int, int):
            return frozenset({"INTEGERTUPLE"})
        if len(args) == 1 and typing.get_origin(args[0]) is tuple and typing.get_args(args[0]) == (int,):
            return frozenset({"GRID"})
        if len(args) == 1 and typing.get_origin(args[0]) is tuple and typing.get_args(args[0]) == ():
            return frozenset({"TUPLETUPLE"})
        if len(args) == 2 and args[0] is int and typing.get_origin(args[1]) is tuple and typing.get_args(args[1]) == (int, int):
            return frozenset({"CELL"})
        return frozenset({"ANY"})
    if hint is typing.FrozenSet or origin is frozenset:
        if not args:
            return frozenset({"ANY"})
        inner = resolve_type_tag(args[0])
        mapping = {
            frozenset({"INTEGER"}): "INTEGERSET",
            frozenset({"CELL"}): "OBJECT",
            frozenset({"INTEGERTUPLE"}): "INDICES",
            frozenset({"OBJECT"}): "OBJECTS",
            frozenset({"INDICES"}): "INDICESSET",
        }
        if inner in mapping:
            return frozenset({mapping[inner]})
        return frozenset({"ANY"})
    if hint is typing.Container or origin is typing.Container:
        return frozenset({"ANY"})
    return frozenset({"ANY"})


# ---------- base (non-higher-order) function signatures, extracted mechanically ----------

@dataclass(frozen=True)
class FunctionSignature:
    param_tags: tuple  # tuple[frozenset[str], ...], one per positional parameter
    return_tag: frozenset


def _build_base_signatures() -> dict:
    registry = {}
    for name in dir(dsl):
        if name.startswith("_") or name in HIGHER_ORDER_NAMES:
            continue
        fn = getattr(dsl, name)
        if not inspect.isfunction(fn):
            continue
        hints = typing.get_type_hints(fn, globalns=vars(arc_types))
        params = list(inspect.signature(fn).parameters)
        param_tags = tuple(resolve_type_tag(hints[p]) for p in params if p in hints)
        return_tag = resolve_type_tag(hints["return"])
        registry[name] = FunctionSignature(param_tags=param_tags, return_tag=return_tag)
    return registry


BASE_SIGNATURES = _build_base_signatures()


# ---------- closures over higher-order functions ----------

@dataclass(frozen=True)
class Closure:
    """A constructed callable value. `parts` holds either nested `Closure`s,
    base-function-name strings (a bare reference to a dsl.py function, used
    directly wherever a Callable-typed value is needed), or bound concrete
    values (all ARC values are frozensets/tuples/ints, always hashable).
    `param_tags`/`return_tag` are computed once at construction, not
    recomputed on each use."""
    kind: str
    parts: tuple
    param_tags: tuple
    return_tag: frozenset


def signature_of(value) -> FunctionSignature:
    """`value` is either a base-function-name string or a Closure."""
    if isinstance(value, str):
        return BASE_SIGNATURES[value]
    return FunctionSignature(param_tags=value.param_tags, return_tag=value.return_tag)


def build_closure(kind: str, *parts) -> Closure:
    """Hand-derived propagation rules, read directly from each higher-order
    function's real implementation in dsl.py (not from its bare `Callable`
    hint, which carries no signature information)."""
    if kind == "lbind":
        inner, _bound = parts
        sig = signature_of(inner)
        return Closure(kind, parts, sig.param_tags[1:], sig.return_tag)
    if kind == "rbind":
        inner, _bound = parts
        sig = signature_of(inner)
        return Closure(kind, parts, sig.param_tags[:-1], sig.return_tag)
    if kind == "compose":
        outer, inner = parts
        return Closure(kind, parts, signature_of(inner).param_tags, signature_of(outer).return_tag)
    if kind == "chain":
        h, g, f = parts
        return Closure(kind, parts, signature_of(f).param_tags, signature_of(h).return_tag)
    if kind == "fork":
        outer, a, _b = parts
        return Closure(kind, parts, signature_of(a).param_tags, signature_of(outer).return_tag)
    if kind == "power":
        inner, _n = parts
        sig = signature_of(inner)
        return Closure(kind, parts, sig.param_tags, sig.return_tag)
    if kind == "matcher":
        inner, _target = parts
        sig = signature_of(inner)
        return Closure(kind, parts, (sig.param_tags[0],), frozenset({"BOOLEAN"}))
    raise ValueError(f"unknown closure kind: {kind!r}")


def to_callable(value):
    """Resolves a base-function-name string or a Closure down to a real,
    directly-callable Python function -- always built from vendored dsl.py
    combinators (`dsl.lbind`, `dsl.compose`, ...), never reimplemented."""
    if isinstance(value, str):
        return getattr(dsl, value)
    kind, parts = value.kind, value.parts
    if kind == "lbind":
        inner, bound = parts
        return dsl.lbind(to_callable(inner), bound)
    if kind == "rbind":
        inner, bound = parts
        return dsl.rbind(to_callable(inner), bound)
    if kind == "compose":
        outer, inner = parts
        return dsl.compose(to_callable(outer), to_callable(inner))
    if kind == "chain":
        h, g, f = parts
        return dsl.chain(to_callable(h), to_callable(g), to_callable(f))
    if kind == "fork":
        outer, a, b = parts
        return dsl.fork(to_callable(outer), to_callable(a), to_callable(b))
    if kind == "power":
        inner, n = parts
        return dsl.power(to_callable(inner), n)
    if kind == "matcher":
        inner, target = parts
        return dsl.matcher(to_callable(inner), target)
    raise ValueError(f"unknown closure kind: {kind!r}")


def call_base(name: str, *args):
    """Calls a base (non-higher-order) dsl.py function directly by name."""
    return getattr(dsl, name)(*args)
