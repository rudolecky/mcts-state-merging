from mcts_phase0.datasets.prosqa import (
    ProsQAInstance,
    _path_count,
    average_path_count,
    canonical_state_at,
    generate_dataset,
    parse_and_verify,
)


def test_path_count_simple_chain():
    facts = [("a", "b"), ("b", "c")]
    assert _path_count(facts, "a", "c") == 1
    assert _path_count(facts, "a", "b") == 1
    assert _path_count(facts, "b", "a") == 0  # wrong direction


def test_path_count_with_shortcut_reconvergence():
    facts = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "c")]  # shortcut a->c
    assert _path_count(facts, "a", "d") == 2  # a-b-c-d and a-c-d
    assert _path_count(facts, "a", "c") == 2  # a-b-c and a-c directly


def test_path_count_with_dead_end_distractor_does_not_add_paths():
    facts = [("a", "b"), ("b", "c"), ("b", "x")]  # x is a dead end off b
    assert _path_count(facts, "a", "c") == 1
    assert _path_count(facts, "a", "x") == 1


def test_generate_dataset_positive_instances_have_valid_reachable_path():
    instances = generate_dataset(n_positive=30, n_negative=0, seed=1)
    assert len(instances) == 30
    for inst in instances:
        assert inst.answer == "yes"
        assert inst.correct_path is not None
        assert inst.correct_path[0] == inst.start
        assert inst.correct_path[-1] == inst.target
        # recompute path count independently from the shown facts
        recomputed = _path_count(list(inst.facts), inst.start, inst.target)
        assert recomputed == inst.path_count
        assert recomputed >= 1


def test_generate_dataset_negative_instances_are_unreachable():
    instances = generate_dataset(n_positive=10, n_negative=10, seed=2)
    negatives = [i for i in instances if i.answer == "no"]
    assert len(negatives) == 10
    for inst in negatives:
        assert inst.path_count == 0
        assert inst.correct_path is None
        recomputed = _path_count(list(inst.facts), inst.start, inst.target)
        assert recomputed == 0


def test_average_path_count_lands_near_expected_ballpark():
    instances = generate_dataset(n_positive=200, n_negative=0, seed=3, reconverge_fraction=0.4)
    avg = average_path_count(instances)
    # Coconut's original ProsQA measured ~1.6 avg shortest paths/query;
    # we're not trying to hit that exactly, just land in a plausible,
    # deliberately-modest-multiplicity ballpark rather than 1.0 flat or
    # something wildly inflated like Countdown's.
    assert 1.1 <= avg <= 2.2


def test_generate_dataset_is_deterministic_given_seed():
    a = generate_dataset(n_positive=5, n_negative=5, seed=42)
    b = generate_dataset(n_positive=5, n_negative=5, seed=42)
    a_keys = [(i.start, i.target, i.answer, i.path_count) for i in a]
    b_keys = [(i.start, i.target, i.answer, i.path_count) for i in b]
    assert a_keys == b_keys


def test_parse_and_verify_accepts_correct_chain():
    instances = generate_dataset(n_positive=20, n_negative=0, seed=5)
    checked = 0
    for inst in instances:
        path = inst.correct_path
        step_bodies = [f"{path[i]} is a {path[i + 1]}" for i in range(len(path) - 1)]
        ok, info = parse_and_verify(inst, step_bodies, "yes")
        assert ok, info
        checked += 1
    assert checked == 20


def test_parse_and_verify_rejects_fabricated_fact():
    instances = generate_dataset(n_positive=5, n_negative=0, seed=6)
    inst = instances[0]
    ok, info = parse_and_verify(inst, [f"{inst.start} is a nonexistent_entity_xyz"], "yes")
    assert not ok
    assert "not given" in info["reason"]


def test_parse_and_verify_rejects_disconnected_chain():
    instances = generate_dataset(n_positive=5, n_negative=0, seed=7)
    inst = instances[0]
    path = inst.correct_path
    if len(path) < 3:
        return  # need at least 2 hops to drop the middle one meaningfully
    # cite only the first fact, skipping the rest -- doesn't reach target
    step_bodies = [f"{path[0]} is a {path[1]}"]
    ok, info = parse_and_verify(inst, step_bodies, "yes")
    assert not ok
    assert not info["well_formed"]


def test_parse_and_verify_accepts_no_with_no_steps():
    instances = generate_dataset(n_positive=5, n_negative=5, seed=8)
    negatives = [i for i in instances if i.answer == "no"]
    assert negatives
    ok, info = parse_and_verify(negatives[0], [], "no")
    assert ok
    assert info["well_formed"] is True


def test_parse_and_verify_tolerates_trailing_periods():
    """The prompt's Facts block lists facts as "- x is a y." so the model
    reliably copies that trailing period into its step lines. That's a
    formatting artifact, not a reasoning error, and must not be scored wrong
    -- this bug drove every V-hat to 0 in the first calibration run.
    """
    instances = generate_dataset(n_positive=5, n_negative=0, seed=11)
    inst = instances[0]
    path = inst.correct_path
    step_bodies = [f"{path[i]} is a {path[i + 1]}." for i in range(len(path) - 1)]
    ok, info = parse_and_verify(inst, step_bodies, "yes.")
    assert ok, info


def test_parse_and_verify_rejects_missing_answer():
    instances = generate_dataset(n_positive=5, n_negative=0, seed=9)
    inst = instances[0]
    ok, info = parse_and_verify(inst, [], None)
    assert not ok
    assert info["reason"] == "no answer line"


# ---------- canonical_state_at: ground-truth "same state" key ----------

def _reconverging_instance() -> ProsQAInstance:
    # Same shortcut-reconvergence fixture as test_path_count_with_shortcut_reconvergence:
    # a->b->c->d is the main chain, a->c is a genuinely different second route to c.
    facts = (("a", "b"), ("b", "c"), ("c", "d"), ("a", "c"))
    return ProsQAInstance(
        facts=facts, start="a", target="d", answer="yes",
        path_count=2, correct_path=("a", "b", "c", "d"),
    )


def test_canonical_state_at_empty_prefix_is_start():
    inst = _reconverging_instance()
    assert canonical_state_at(inst, []) == "a"


def test_canonical_state_at_follows_a_valid_prefix():
    inst = _reconverging_instance()
    assert canonical_state_at(inst, ["a is a b"]) == "b"
    assert canonical_state_at(inst, ["a is a b", "b is a c"]) == "c"
    assert canonical_state_at(inst, ["a is a b", "b is a c", "c is a d"]) == "d"


def test_canonical_state_at_recognizes_the_real_transposition():
    # The main chain (2 hops) and the reconverging shortcut (1 hop) are a
    # genuine, by-construction transposition: two different fact-sequences
    # reaching the identical current entity. This property is the whole
    # reason canonical_state_at keys on current entity, not full sequence.
    inst = _reconverging_instance()
    via_chain = canonical_state_at(inst, ["a is a b", "b is a c"])
    via_shortcut = canonical_state_at(inst, ["a is a c"])
    assert via_chain == via_shortcut == "c"


def test_canonical_state_at_none_on_fact_not_given():
    inst = _reconverging_instance()
    assert canonical_state_at(inst, ["a is a zzz"]) is None


def test_canonical_state_at_none_on_unparseable_step():
    inst = _reconverging_instance()
    assert canonical_state_at(inst, ["not a valid line at all"]) is None


def test_canonical_state_at_none_on_non_contiguous_jump():
    # "b is a c" is a real fact, but the prefix never established "b" as the
    # current entity (current starts at "a") -- a disconnected step, not a
    # continuation, so this must fail closed rather than silently landing on "c".
    inst = _reconverging_instance()
    assert canonical_state_at(inst, ["b is a c"]) is None


def test_canonical_state_at_none_after_an_already_bad_step():
    inst = _reconverging_instance()
    assert canonical_state_at(inst, ["a is a zzz", "a is a b"]) is None
