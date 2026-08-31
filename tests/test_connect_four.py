"""Verification for connect_four.py -- no model/LLM involved.

Uses the exact hand-built double-threat scenario from
test_connect_four_engine.py (X already at columns 1,2; column 3 opens a
double threat at columns 0/4) as a known-good forced-win-in-3 fixture, so
parse_and_verify's accept/reject behavior is checked against a position
whose correct answer was already independently established there.
"""

from mcts_phase0.datasets.connect_four import (
    ConnectFourInstance,
    build_prompt,
    canonical_state_at,
    generate_puzzles,
    parse_and_verify,
)
from mcts_phase0.datasets.connect_four_engine import apply_move, board_key, shortest_forced_win, replay


def _fixture_instance(k_plies: int = 3) -> ConnectFourInstance:
    # pre_moves: X->col1, O->col5, X->col2, O->col5 (O's moves dumped in a
    # column that doesn't interfere with the 0..4 double-threat setup)
    return ConnectFourInstance(pre_moves=(1, 5, 2, 5), width=6, height=2, k_plies=k_plies, path_count=1)


def test_to_move_and_id_are_consistent_with_pre_moves_parity():
    inst = _fixture_instance()
    assert inst.to_move == "X"  # 4 pre_moves played -> X to move next
    assert isinstance(inst.id, str) and inst.id  # non-empty, stable string


def test_two_transposed_pre_move_orders_get_the_same_id():
    a = ConnectFourInstance(pre_moves=(1, 5, 2, 5), width=6, height=2, k_plies=3, path_count=1)
    b = ConnectFourInstance(pre_moves=(2, 5, 1, 5), width=6, height=2, k_plies=3, path_count=1)
    assert a.id == b.id  # same resulting board + side to move -> same puzzle


def test_build_prompt_contains_grid_and_instructions():
    prompt = build_prompt(_fixture_instance())
    assert "Step N: drop in column C" in prompt
    assert "Answer: win" in prompt
    assert "X to move" in prompt


def test_build_prompt_scratch_board_variant_adds_extra_instruction():
    strict = build_prompt(_fixture_instance(), encourage_scratch_board=False)
    loose = build_prompt(_fixture_instance(), encourage_scratch_board=True)
    assert "sketch the board" not in strict
    assert "sketch the board" in loose


def test_parse_and_verify_accepts_the_correct_forcing_line():
    inst = _fixture_instance()
    steps = ["drop in column 3", "drop in column 0", "drop in column 4"]
    ok, info = parse_and_verify(inst, steps, "win")
    assert ok is True
    assert info["well_formed"] is True


def test_parse_and_verify_tolerates_trailing_period():
    inst = _fixture_instance()
    steps = ["drop in column 3.", "drop in column 0.", "drop in column 4."]
    ok, info = parse_and_verify(inst, steps, "win.")
    assert ok is True


def test_parse_and_verify_rejects_non_forcing_first_move():
    inst = _fixture_instance()
    steps = ["drop in column 0", "drop in column 3", "drop in column 4"]
    ok, info = parse_and_verify(inst, steps, "win")
    assert ok is False
    assert info["well_formed"] is False
    assert "forcing" in info["reason"]


def test_parse_and_verify_rejects_illegal_move_on_full_column():
    inst = _fixture_instance()  # column 5 already has 2 pieces, height=2 -> full
    ok, info = parse_and_verify(inst, ["drop in column 5"], "win")
    assert ok is False
    assert "illegal" in info["reason"]


def test_parse_and_verify_rejects_unparseable_step():
    inst = _fixture_instance()
    ok, info = parse_and_verify(inst, ["move to column 3"], "win")
    assert ok is False
    assert "unparseable" in info["reason"]


def test_parse_and_verify_well_formed_but_incorrect_on_wrong_final_answer():
    inst = _fixture_instance()
    steps = ["drop in column 3", "drop in column 0", "drop in column 4"]
    ok, info = parse_and_verify(inst, steps, "lose")
    assert ok is False
    assert info["well_formed"] is True  # the line itself was a genuine forced win


def test_parse_and_verify_rejects_extra_moves_after_win():
    inst = _fixture_instance()
    steps = ["drop in column 3", "drop in column 0", "drop in column 4", "drop in column 1"]
    ok, info = parse_and_verify(inst, steps, "win")
    assert ok is False
    assert "extra moves" in info["reason"]


def test_generate_puzzles_are_all_genuinely_exact_k_ply_forced_wins():
    # k_plies=3 exact forced-wins are genuinely rare among random openings
    # (an empirical Stage-0 finding -- see generate_puzzles' docstring), so
    # n_low/max_pre_moves are picked from measured yield, not guessed.
    instances = generate_puzzles(n_low=3, n_high=0, seed=0, width=5, height=4, k_plies=3, max_pre_moves=8)
    assert len(instances) >= 1  # generation found at least something at this budget
    for inst in instances:
        board = replay(inst.pre_moves, inst.width, inst.height)
        assert shortest_forced_win(board, inst.to_move, max_plies=inst.k_plies, height=inst.height) == inst.k_plies
        assert inst.path_count >= 1  # every kept instance genuinely has a preserving first move


def test_generate_puzzles_is_deterministic_given_a_seed():
    a = generate_puzzles(n_low=3, n_high=0, seed=7, width=5, height=4, k_plies=3, max_pre_moves=8)
    b = generate_puzzles(n_low=3, n_high=0, seed=7, width=5, height=4, k_plies=3, max_pre_moves=8)
    assert [inst.id for inst in a] == [inst.id for inst in b]


def test_canonical_state_at_matches_direct_engine_replay():
    inst = _fixture_instance()
    key = canonical_state_at(inst, ["drop in column 3", "drop in column 0"])

    board = replay(inst.pre_moves, inst.width, inst.height)
    board = apply_move(board, 3, "X")
    board = apply_move(board, 0, "O")
    expected = board_key(board, "X")
    assert key == expected


def test_canonical_state_at_none_on_unparseable_or_illegal_move():
    inst = _fixture_instance()
    assert canonical_state_at(inst, ["not a move"]) is None
    assert canonical_state_at(inst, ["drop in column 5"]) is None  # column 5 already full


def test_generate_puzzles_respects_exclude_ids():
    first_batch = generate_puzzles(n_low=3, n_high=0, seed=3, width=5, height=4, k_plies=3, max_pre_moves=8)
    exclude = {inst.id for inst in first_batch}
    second_batch = generate_puzzles(
        n_low=3, n_high=0, seed=3, width=5, height=4, k_plies=3, max_pre_moves=8, exclude_ids=exclude,
    )
    assert exclude.isdisjoint({inst.id for inst in second_batch})
