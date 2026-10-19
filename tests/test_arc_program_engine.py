"""Verification for arc_program_engine.py: initial-state shape, canonical
merge-key order-independence, a genuine transposition using REAL vendored
primitives (hmirror and vmirror commute -- both reach the same grid as a
180-degree rotation regardless of order, the ARC analogue of Blocksworld's
independent-block-pair test), legal_moves type-checking, and apply_move
correctness against a real base function.
"""

from mcts_phase0.datasets import arc_program_engine as engine
from mcts_phase0.datasets.arc_types_registry import dsl


def _grid(rows):
    return tuple(tuple(row) for row in rows)


_TRAIN_INPUTS = (_grid([[1, 2], [3, 4]]),)


def test_create_initial_state_shape():
    state = engine.create_initial_state(_TRAIN_INPUTS)
    assert state.contexts == ((_grid([[1, 2], [3, 4]]),),)
    assert state.type_schema == (frozenset({"GRID"}),)


def test_apply_move_hmirror_matches_real_dsl_function():
    state = engine.create_initial_state(_TRAIN_INPUTS)
    move = ("hmirror", (("ctx", 0),))
    new_state = engine.apply_move(state, move)
    assert new_state.contexts[0][-1] == dsl.hmirror(_TRAIN_INPUTS[0])
    assert new_state.type_schema[-1] == frozenset({"GRID"})


def test_canonical_key_is_order_independent():
    state = engine.create_initial_state(_TRAIN_INPUTS)
    order_a = engine.apply_move(engine.apply_move(state, ("hmirror", (("ctx", 0),))), ("width", (("ctx", 0),)))
    order_b = engine.apply_move(engine.apply_move(state, ("width", (("ctx", 0),))), ("hmirror", (("ctx", 0),)))
    # Different insertion order of the SAME two independent values -> same set.
    assert engine.canonical_key(order_a) == engine.canonical_key(order_b)


def test_two_independent_transforms_of_the_same_value_commute():
    # Both hmirror and vmirror are applied independently to ctx[0] (the
    # original input), not chained onto each other's output -- the direct
    # ARC analogue of Blocksworld's independent-block-pair transposition
    # (two independent operations on the same base state, order doesn't
    # matter). canonical_key is order-independent over the *full* set of
    # context entries, so applying them in either order reaches the same
    # available toolkit of values, just at swapped indices.
    state = engine.create_initial_state(_TRAIN_INPUTS)
    order_a = engine.apply_move(engine.apply_move(state, ("hmirror", (("ctx", 0),))), ("vmirror", (("ctx", 0),)))
    order_b = engine.apply_move(engine.apply_move(state, ("vmirror", (("ctx", 0),))), ("hmirror", (("ctx", 0),)))
    assert engine.canonical_key(order_a) == engine.canonical_key(order_b)


def test_is_goal_detects_the_target_at_any_context_index():
    state = engine.create_initial_state(_TRAIN_INPUTS)
    mirrored = engine.apply_move(state, ("hmirror", (("ctx", 0),)))
    target = (dsl.hmirror(_TRAIN_INPUTS[0]),)
    assert engine.is_goal(mirrored, target) is True
    assert engine.is_goal(state, target) is False
    # The target is no longer the most recent value, but the program did
    # produce it, so it is still solved.
    kept_going = engine.apply_move(mirrored, ("width", (("ctx", 0),)))
    assert kept_going.contexts[0][-1] != target[0]
    assert engine.is_goal(kept_going, target) is True
    assert engine.goal_index(kept_going, target) == 1


def test_is_goal_is_a_function_of_the_merge_key():
    """Pins the invariant that goal-ness depends only on what the merge key
    carries. `canonical_key` is order-independent while the old `is_goal`
    read `ctx[-1]`, so two states sharing a key could disagree on goal-ness
    and the merge arm would keep whichever ordering arrived first.

    The states that would have exposed it are unreachable in the current
    search (a non-final-goal state's own prefix is already terminal, so it
    is never expanded), so this guards an invariant rather than fixing an
    observed failure -- see `engine.goal_index`."""
    state = engine.create_initial_state(_TRAIN_INPUTS)
    target = (dsl.hmirror(_TRAIN_INPUTS[0]),)
    # Same two values, two orderings: goal-last and goal-not-last.
    goal_last = engine.apply_move(
        engine.apply_move(state, ("width", (("ctx", 0),))), ("hmirror", (("ctx", 0),)))
    goal_first = engine.apply_move(
        engine.apply_move(state, ("hmirror", (("ctx", 0),))), ("width", (("ctx", 0),)))
    assert engine.canonical_key(goal_last) == engine.canonical_key(goal_first)
    assert engine.is_goal(goal_last, target) is True
    assert engine.is_goal(goal_first, target) is True


def test_goal_progress_is_one_exactly_when_is_goal():
    state = engine.create_initial_state(_TRAIN_INPUTS)
    mirrored = engine.apply_move(state, ("hmirror", (("ctx", 0),)))
    target = (dsl.hmirror(_TRAIN_INPUTS[0]),)
    assert engine.goal_progress(mirrored, target) == 1.0
    assert engine.goal_progress(state, target) < 1.0


def test_goal_progress_gives_cell_level_partial_credit():
    target = (_grid([[1, 2], [3, 4]]),)
    # 2 of 4 cells right, same shape.
    half = engine.ProgramState(contexts=((_grid([[1, 2], [9, 9]]),),), type_schema=(frozenset({"GRID"}),))
    assert engine.goal_progress(half, target) == 0.5
    # Wrong shape gets no partial credit at all.
    wrong_shape = engine.ProgramState(contexts=((_grid([[1, 2, 3]]),),), type_schema=(frozenset({"GRID"}),))
    assert engine.goal_progress(wrong_shape, target) == 0.0


def test_goal_progress_means_over_training_examples():
    targets = (_grid([[1, 1]]), _grid([[2, 2]]))
    # First example exact, second example completely wrong -> mean 0.5.
    st = engine.ProgramState(
        contexts=((_grid([[1, 1]]),), (_grid([[9, 9]]),)),
        type_schema=(frozenset({"GRID"}),),
    )
    assert engine.goal_progress(st, targets) == 0.5


def test_values_key_mode_is_coarser_than_typed():
    """Observational-equivalence key: two states holding the same concrete
    values merge even when the static tags differ, which the typed key
    keeps apart."""
    grid = _grid([[1, 2], [3, 4]])
    a = engine.ProgramState(contexts=((grid,),), type_schema=(frozenset({"GRID"}),))
    b = engine.ProgramState(contexts=((grid,),), type_schema=(frozenset({"ANY"}),))
    assert engine.canonical_key(a, "typed") != engine.canonical_key(b, "typed")
    assert engine.canonical_key(a, "values") == engine.canonical_key(b, "values")


def test_unknown_key_mode_is_rejected():
    import pytest
    state = engine.create_initial_state(_TRAIN_INPUTS)
    with pytest.raises(ValueError):
        engine.canonical_key(state, "nope")


def test_legal_moves_are_type_checked_and_bounded():
    import random
    state = engine.create_initial_state(_TRAIN_INPUTS)
    rng = random.Random(0)
    moves = engine.legal_moves(state, rng, sample_size=15)
    assert 0 < len(moves) <= 15
    for func, arg_specs in moves:
        needed = engine.CLOSURE_BUILDER_OWN_SIGNATURES.get(func) or engine._param_tags_for(func)
        assert len(arg_specs) == len(needed)


def test_legal_moves_never_offers_a_context_index_out_of_range():
    import random
    state = engine.create_initial_state(_TRAIN_INPUTS)  # only index 0 exists
    rng = random.Random(1)
    moves = engine.legal_moves(state, rng, sample_size=25)
    for _func, arg_specs in moves:
        for kind, val in arg_specs:
            if kind == "ctx":
                assert val == 0


def test_legal_moves_order_is_process_independent():
    """`legal_moves` used to return `list(<set of tuples containing
    strings>)`, whose iteration order follows randomized string hashes, so
    the same state and seed produced different move orders in different
    processes -- silently making every ARC search unreproducible. The order
    must be a function of the moves alone."""
    import random
    import subprocess
    import sys

    state = engine.create_initial_state(_TRAIN_INPUTS)
    local = [m[0] for m in engine.legal_moves(state, random.Random(0), 8)]

    code = (
        "import random;"
        "from mcts_phase0.datasets import arc_program_engine as e;"
        f"s = e.create_initial_state((({tuple(tuple(r) for r in _TRAIN_INPUTS[0])!r}),));"
        "print([m[0] for m in e.legal_moves(s, random.Random(0), 8)])"
    )
    for hash_seed in ("1", "2"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env={"PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"})
        assert out.returncode == 0, out.stderr
        assert eval(out.stdout) == local, f"order differs under PYTHONHASHSEED={hash_seed}"
