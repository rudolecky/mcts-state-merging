"""Verification for blocksworld_engine.py, no model/LLM involved:
- correct operator grounding (count, preconditions/effects for a hand-
  picked pickup/stack pair)
- a genuine transposition: two independent-block-pair action sequences
  commute (simpler to construct here than Connect Four's own fixture --
  no alternating-mover constraint to trip over, since this is single-agent)
- generate_puzzles returns instances at the exact requested plan_length,
  checked directly against pyperplan's own breadth_first_search, not assumed
"""

from pyperplan.search import breadth_first_search

from mcts_phase0.datasets.blocksworld_engine import (
    apply_move,
    generate_puzzles,
    goal_state,
    is_goal,
    legal_moves,
    make_task,
    scrambled_state,
)


def test_operator_count_matches_the_grounding_formula():
    # 2N pickup/putdown pairs + N*(N-1) stack/unstack pairs
    task = make_task(4)
    assert len(task.operators) == 2 * 4 + 4 * 3 * 2


def test_pickup_operator_preconditions_and_effects():
    task = make_task(3)
    pickup_1 = next(op for op in task.operators if op.name == "pickup(1)")
    assert pickup_1.preconditions == frozenset({"clear(1)", "ontable(1)", "handempty"})
    assert pickup_1.add_effects == frozenset({"holding(1)"})
    assert pickup_1.del_effects == frozenset({"clear(1)", "ontable(1)", "handempty"})


def test_stack_operator_preconditions_and_effects():
    task = make_task(3)
    stack_1_2 = next(op for op in task.operators if op.name == "stack(1,2)")
    assert stack_1_2.preconditions == frozenset({"holding(1)", "clear(2)"})
    assert stack_1_2.add_effects == frozenset({"on(1,2)", "clear(1)", "handempty"})
    assert stack_1_2.del_effects == frozenset({"holding(1)", "clear(2)"})


def test_goal_state_is_a_single_numeric_tower():
    goal = goal_state(3)
    assert goal == frozenset({"handempty", "ontable(1)", "on(2,1)", "on(3,2)", "clear(3)"})


def test_scrambled_state_is_always_a_valid_configuration():
    import random

    rng = random.Random(0)
    for _ in range(50):
        state = scrambled_state(4, rng)
        # every block is either ontable or on exactly one other block, never both
        for b in range(1, 5):
            on_table = f"ontable({b})" in state
            on_something = any(f"on({b},{c})" in state for c in range(1, 5) if c != b)
            assert on_table != on_something  # exactly one, not both, not neither


def test_two_independent_block_pair_sequences_commute():
    # 4 blocks, all on the table to start. Stacking 1-on-2 then 3-on-4 touches
    # entirely disjoint block pairs from stacking 3-on-4 then 1-on-2 -- a
    # genuine transposition, simpler than Connect Four's needed fixture since
    # there's no alternating-mover constraint here at all.
    task = make_task(4)
    start = frozenset({"ontable(1)", "ontable(2)", "ontable(3)", "ontable(4)", "clear(1)", "clear(2)", "clear(3)", "clear(4)", "handempty"})

    def _play(state, names):
        for name in names:
            op = next(o for o in legal_moves(state, task) if o.name == name)
            state = apply_move(state, op)
        return state

    result_a = _play(start, ["pickup(1)", "stack(1,2)", "pickup(3)", "stack(3,4)"])
    result_b = _play(start, ["pickup(3)", "stack(3,4)", "pickup(1)", "stack(1,2)"])
    assert result_a == result_b


def test_generate_puzzles_returns_states_at_exact_plan_length():
    puzzles = generate_puzzles(n=5, seed=0, num_blocks=5, target_plan_length=10)
    assert len(puzzles) == 5
    ids = {inst.id for inst in puzzles}
    assert len(ids) == 5

    task = make_task(5)
    goal = goal_state(5)
    for inst in puzzles:
        assert inst.plan_length == 10
        task.initial_state = inst.start_state
        task.goals = goal
        plan = breadth_first_search(task)
        assert plan is not None
        assert len(plan) == 10
