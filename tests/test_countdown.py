import random

from mcts_phase0.datasets.countdown import (
    expr_to_steps,
    generate_instance,
    generate_stratified,
    parse_and_verify,
)


def _leaves(expr: str) -> list[int]:
    """Extract the leaf integers from a canonical bracketed expression."""
    return [int(tok) for tok in expr.replace("(", " ").replace(")", " ")
            .replace("+", " ").replace("-", " ").replace("*", " ").replace("/", " ").split()]


def test_generate_instance_solutions_are_valid_and_use_each_number_once():
    rng = random.Random(1234)
    checked = 0
    for _ in range(200):
        inst = generate_instance(rng, num_numbers=4, number_low=1, number_high=15)
        if inst is None:
            continue
        assert inst.path_count >= len(inst.example_solutions)
        for expr in inst.example_solutions:
            value = eval(expr)  # noqa: S307 -- generated numeric expression, safe
            assert value == inst.target
            assert sorted(_leaves(expr)) == sorted(inst.numbers)
        checked += 1
    assert checked > 50  # sanity: generation isn't silently failing every time


def test_known_24_reachable_two_ways_from_2_3_4():
    inst = generate_instance(random.Random(0), num_numbers=3, number_low=2, number_high=2)
    # force a known draw instead of relying on random range collapse
    from mcts_phase0.datasets.countdown import _all_expressions

    expr_map = _all_expressions((2, 3, 4))
    assert 24 in expr_map
    assert "((2*4)*3)" in expr_map[24] or "(3*(2*4))" in expr_map[24]
    assert "((3*4)*2)" in expr_map[24] or "(2*(3*4))" in expr_map[24]
    assert len(expr_map[24]) >= 2


def test_expr_to_steps_matches_parse_and_verify():
    from mcts_phase0.datasets.countdown import CountdownInstance

    rng = random.Random(99)
    checked = 0
    for _ in range(100):
        inst = generate_instance(rng, num_numbers=4, number_low=1, number_high=12)
        if inst is None or not inst.example_solutions:
            continue
        expr = inst.example_solutions[0]
        steps = expr_to_steps(expr)
        step_bodies = [s for s in steps]
        ok, info = parse_and_verify(inst, step_bodies, str(inst.target))
        assert ok, (expr, steps, info)
        checked += 1
    assert checked > 20


def test_verifier_tolerates_trailing_periods():
    """Same formatting-artifact tolerance as prosqa's verifier: a trailing
    period on a step or answer line is prose habit, not a reasoning error.
    """
    from mcts_phase0.datasets.countdown import CountdownInstance

    inst = CountdownInstance(numbers=(2, 4, 3), target=24, path_count=1, example_solutions=("((2*4)*3)",))
    ok, info = parse_and_verify(inst, ["2 * 4 = 8.", "8 * 3 = 24."], "24.")
    assert ok, info


def test_verifier_rejects_bad_arithmetic():
    from mcts_phase0.datasets.countdown import CountdownInstance

    inst = CountdownInstance(numbers=(2, 3, 4), target=24, path_count=1, example_solutions=("((2*4)*3)",))
    ok, info = parse_and_verify(inst, ["2 * 4 = 9", "9 * 3 = 27"], "27")
    assert not ok
    assert not info["well_formed"]


def test_verifier_rejects_reusing_unavailable_number():
    from mcts_phase0.datasets.countdown import CountdownInstance

    inst = CountdownInstance(numbers=(2, 3, 4), target=24, path_count=1, example_solutions=("((2*4)*3)",))
    # 5 was never available
    ok, info = parse_and_verify(inst, ["5 * 4 = 20", "20 + 3 = 23"], "23")
    assert not ok
    assert "not available" in info["reason"]


def test_verifier_rejects_non_exact_division():
    from mcts_phase0.datasets.countdown import CountdownInstance

    inst = CountdownInstance(numbers=(2, 3, 5), target=1, path_count=0, example_solutions=())
    ok, info = parse_and_verify(inst, ["5 / 2 = 2"], "2")
    assert not ok
    assert "division" in info["reason"]


def test_verifier_rejects_negative_subtraction():
    from mcts_phase0.datasets.countdown import CountdownInstance

    inst = CountdownInstance(numbers=(2, 5), target=-3, path_count=0, example_solutions=())
    ok, info = parse_and_verify(inst, ["2 - 5 = -3"], "-3")
    assert not ok
    assert "subtraction" in info["reason"]


def test_verifier_rejects_missing_answer_line():
    from mcts_phase0.datasets.countdown import CountdownInstance

    inst = CountdownInstance(numbers=(2, 4, 3), target=24, path_count=1, example_solutions=("((2*4)*3)",))
    ok, info = parse_and_verify(inst, ["2 * 4 = 8", "8 * 3 = 24"], None)
    assert not ok
    assert info["well_formed"] is True  # derivation was fine, just no answer stated
    assert info["reason"] == "no answer line"


def test_generate_stratified_buckets_and_dedup():
    instances = generate_stratified(n_low=5, n_high=5, seed=42, num_numbers=4, number_low=1, number_high=12)
    assert len(instances) == 10
    low = [i for i in instances if i.path_count <= 2]
    high = [i for i in instances if i.path_count > 2]
    assert len(low) == 5
    assert len(high) == 5
    keys = [(i.numbers, i.target) for i in instances]
    assert len(keys) == len(set(keys))  # no duplicates


def test_generate_stratified_is_deterministic_given_seed():
    a = generate_stratified(n_low=3, n_high=3, seed=7)
    b = generate_stratified(n_low=3, n_high=3, seed=7)
    assert [(i.numbers, i.target) for i in a] == [(i.numbers, i.target) for i in b]


def test_generate_stratified_excludes_given_ids():
    baseline = generate_stratified(n_low=5, n_high=5, seed=3, num_numbers=4, number_low=1, number_high=12)
    exclude = {inst.id for inst in baseline[:4]}

    held_out = generate_stratified(
        n_low=5, n_high=5, seed=3, num_numbers=4, number_low=1, number_high=12, exclude_ids=exclude,
    )

    assert len(held_out) == 10
    assert exclude.isdisjoint({inst.id for inst in held_out})
