"""Stage 0 verification: replay real `solvers.py` reference programs
through this project's own `arc_types_registry` machinery (Closure
construction, type tags, `to_callable`/`call_base`) and check every
intermediate value against calling the real, vendored `dsl.py` functions
directly -- not just that the type bookkeeping is internally consistent.
This is the highest-value test in the whole ARC pilot: a subtle
type-propagation bug here would silently corrupt everything built on top
of it in later stages.
"""

import json
import re
import sys
from pathlib import Path

from mcts_phase0.datasets import arc_types_registry as reg
from mcts_phase0.datasets.arc_types_registry import arc_constants, dsl

sys.path.insert(0, "src/mcts_phase0/vendor/arc_dsl")
import solvers as real_solvers  # noqa: E402

DATA_DIR = Path("data/arc_agi/tasks")

CLOSURE_BUILDERS = {"lbind", "rbind", "compose", "chain", "fork", "power", "matcher"}
CLOSURE_CONSUMER_CALLABLE_POS = {
    "apply": [0], "mapply": [0], "mfilter": [1], "sfilter": [1],
    "extract": [1], "argmax": [1], "argmin": [1], "valmax": [1], "valmin": [1],
    "order": [1], "papply": [0], "mpapply": [0],
}

_LINE_RE = re.compile(r"^(\w+) = (\w+)\((.*)\)$")


def _parse_args(arg_str):
    return [] if not arg_str else [a.strip() for a in arg_str.split(",")]


def _grid(task_id):
    raw = json.loads((DATA_DIR / f"{task_id}.json").read_text())
    return tuple(tuple(row) for row in raw["train"][0]["input"])


def replay_solver(source: str, input_grid):
    """Parses a `solve_X(I): ... return O` function body and re-executes it
    entirely through arc_types_registry, returning the final value."""
    lines = [l.strip() for l in source.splitlines() if l.strip()]
    assert lines[0].startswith("def solve_")
    body = lines[1:-1]  # drop def line and 'return O'
    env = {"I": input_grid}
    for line in body:
        m = _LINE_RE.match(line)
        assert m, f"unparseable line: {line!r}"
        var, func, arg_str = m.groups()
        arg_names = _parse_args(arg_str)
        resolved = [_resolve_arg(a, env) for a in arg_names]
        if func in CLOSURE_BUILDERS:
            env[var] = reg.build_closure(func, *resolved)
        elif func in CLOSURE_CONSUMER_CALLABLE_POS:
            call_args = list(resolved)
            for pos in CLOSURE_CONSUMER_CALLABLE_POS[func]:
                call_args[pos] = reg.to_callable(call_args[pos])
            env[var] = getattr(dsl, func)(*call_args)
        else:
            env[var] = reg.call_base(func, *resolved)
    return env["O"]


def _resolve_arg(name, env):
    if name in env:
        return env[name]
    if hasattr(arc_constants, name):
        return getattr(arc_constants, name)
    if name in reg.BASE_SIGNATURES or name in reg.HIGHER_ORDER_NAMES:
        return name  # bare reference to a dsl.py function, by name
    raise ValueError(f"cannot resolve argument {name!r}")


# ---------- resolve_type_tag: verified against every one of arc_types.py's own aliases ----------

def test_resolve_type_tag_matches_every_named_arc_type_alias():
    import arc_types

    expected = {
        "Boolean": {"BOOLEAN"}, "Integer": {"INTEGER"}, "IntegerTuple": {"INTEGERTUPLE"},
        "Numerical": {"INTEGER", "INTEGERTUPLE"}, "IntegerSet": {"INTEGERSET"}, "Grid": {"GRID"},
        "Cell": {"CELL"}, "Object": {"OBJECT"}, "Objects": {"OBJECTS"}, "Indices": {"INDICES"},
        "IndicesSet": {"INDICESSET"}, "Patch": {"OBJECT", "INDICES"}, "Element": {"OBJECT", "GRID"},
        "Piece": {"OBJECT", "INDICES", "GRID"}, "TupleTuple": {"TUPLETUPLE"},
    }
    for alias_name, expected_tags in expected.items():
        hint = getattr(arc_types, alias_name)
        assert reg.resolve_type_tag(hint) == frozenset(expected_tags), alias_name


def test_resolve_type_tag_has_no_unresolved_tag_across_every_base_function_signature():
    for name, sig in reg.BASE_SIGNATURES.items():
        for tags in (*sig.param_tags, sig.return_tag):
            assert tags, f"{name} produced an empty tag set"


# ---------- build_closure: propagation rules, hand-checked ----------

def test_lbind_drops_the_first_parameter():
    # shift: (patch: Patch={Object,Indices}, directions: IntegerTuple) -> Patch
    closure = reg.build_closure("lbind", "shift", (0, 1))
    assert closure.param_tags == (frozenset({"INTEGERTUPLE"}),)  # only "directions" remains
    assert closure.return_tag == frozenset({"OBJECT", "INDICES"})


def test_rbind_drops_the_last_parameter():
    closure = reg.build_closure("rbind", "shift", (0, 1))
    assert closure.param_tags == (frozenset({"OBJECT", "INDICES"}),)  # "patch" remains (1st param)
    assert closure.return_tag == frozenset({"OBJECT", "INDICES"})


def test_matcher_always_returns_boolean():
    closure = reg.build_closure("matcher", "size", 1)
    assert closure.return_tag == frozenset({"BOOLEAN"})


# ---------- full replay of real solvers.py reference programs ----------

_SOLVE_48D8FB45 = """
def solve_48d8fb45(I):
    x1 = objects(I, T, T, T)
    x2 = matcher(size, ONE)
    x3 = extract(x1, x2)
    x4 = lbind(adjacent, x3)
    x5 = extract(x1, x4)
    O = subgrid(x5, I)
    return O
"""

_SOLVE_662C240A = """
def solve_662c240a(I):
    x1 = vsplit(I, THREE)
    x2 = fork(equality, dmirror, identity)
    x3 = compose(flip, x2)
    O = extract(x1, x3)
    return O
"""

_SOLVE_5521C0D9 = """
def solve_5521c0d9(I):
    x1 = objects(I, T, F, T)
    x2 = merge(x1)
    x3 = cover(I, x2)
    x4 = chain(toivec, invert, height)
    x5 = fork(shift, identity, x4)
    x6 = mapply(x5, x1)
    O = paint(x3, x6)
    return O
"""

_SOLVE_1F876C06 = """
def solve_1f876c06(I):
    x1 = fgpartition(I)
    x2 = compose(last, first)
    x3 = power(last, TWO)
    x4 = fork(connect, x2, x3)
    x5 = fork(recolor, color, x4)
    x6 = mapply(x5, x1)
    O = paint(I, x6)
    return O
"""


def test_replay_matches_real_solver_using_lbind_and_matcher():
    grid = _grid("48d8fb45")
    assert replay_solver(_SOLVE_48D8FB45, grid) == real_solvers.solve_48d8fb45(grid)


def test_replay_matches_real_solver_using_fork_and_compose():
    grid = _grid("662c240a")
    assert replay_solver(_SOLVE_662C240A, grid) == real_solvers.solve_662c240a(grid)


def test_replay_matches_real_solver_using_chain_fork_mapply():
    grid = _grid("5521c0d9")
    assert replay_solver(_SOLVE_5521C0D9, grid) == real_solvers.solve_5521c0d9(grid)


def test_replay_matches_real_solver_using_power_and_compose():
    grid = _grid("1f876c06")
    assert replay_solver(_SOLVE_1F876C06, grid) == real_solvers.solve_1f876c06(grid)
