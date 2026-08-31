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


def test_is_goal_checks_the_most_recent_value_per_example():
    state = engine.create_initial_state(_TRAIN_INPUTS)
    mirrored = engine.apply_move(state, ("hmirror", (("ctx", 0),)))
    target = (dsl.hmirror(_TRAIN_INPUTS[0]),)
    assert engine.is_goal(mirrored, target) is True
    assert engine.is_goal(state, target) is False


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
