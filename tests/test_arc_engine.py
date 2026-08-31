"""Verification for arc_engine.py: primitive correctness against hand-
checked grids, a genuine transposition (two independent recolors commute --
the ARC analogue of Blocksworld's independent-block-pair test), and the
brute-force oracle's correctness on a tiny synthetic task with a known
short solution.
"""

from mcts_phase0.datasets.arc_engine import (
    ArcTask,
    apply_move,
    find_short_program,
    is_goal,
    legal_moves,
    load_task,
    same_shape_task,
)


def _grid(rows):
    return tuple(tuple(row) for row in rows)


# ---------- primitive correctness, hand-checked ----------

def test_rot90_matches_a_hand_checked_example():
    grid = _grid([[1, 2, 3], [4, 5, 6]])
    # 90 degrees clockwise: top row becomes right column
    expected = _grid([[4, 1], [5, 2], [6, 3]])
    state = (apply_move((grid,), "rot90"))[0]
    assert state == expected


def test_rot90_applied_four_times_is_identity():
    grid = _grid([[1, 2, 3], [4, 5, 6]])
    state = (grid,)
    for _ in range(4):
        state = apply_move(state, "rot90")
    assert state == (grid,)


def test_flip_h_reverses_each_row():
    grid = _grid([[1, 2, 3], [4, 5, 6]])
    expected = _grid([[3, 2, 1], [6, 5, 4]])
    assert apply_move((grid,), "flip_h") == (expected,)


def test_flip_v_reverses_row_order():
    grid = _grid([[1, 2, 3], [4, 5, 6]])
    expected = _grid([[4, 5, 6], [1, 2, 3]])
    assert apply_move((grid,), "flip_v") == (expected,)


def test_transpose_swaps_rows_and_columns():
    grid = _grid([[1, 2, 3], [4, 5, 6]])
    expected = _grid([[1, 4], [2, 5], [3, 6]])
    assert apply_move((grid,), "transpose") == (expected,)


def test_recolor_replaces_only_the_named_color():
    grid = _grid([[1, 2], [2, 1]])
    expected = _grid([[1, 9], [9, 1]])
    assert apply_move((grid,), ("recolor", 2, 9)) == (expected,)


# ---------- legal_moves grounding ----------

def test_legal_moves_grounds_recolor_over_colors_actually_present():
    state = (_grid([[1, 2], [2, 1]]),)
    moves = legal_moves(state)
    recolor_moves = [m for m in moves if isinstance(m, tuple)]
    assert set(recolor_moves) == {("recolor", 1, 2), ("recolor", 2, 1)}
    assert 3 not in {c for _, c1, c2 in recolor_moves for c in (c1, c2)}


# ---------- genuine transposition: independent recolors commute ----------

def test_two_independent_recolors_commute():
    grid = _grid([[1, 2, 3], [3, 2, 1]])
    state = (grid,)
    order_a = apply_move(apply_move(state, ("recolor", 1, 5)), ("recolor", 2, 6))
    order_b = apply_move(apply_move(state, ("recolor", 2, 6)), ("recolor", 1, 5))
    assert order_a == order_b


# ---------- same_shape_task filter ----------

def test_same_shape_task_rejects_a_dimension_changing_pair():
    task = ArcTask(
        task_id="t",
        train_inputs=(_grid([[1, 2], [3, 4]]),),
        train_outputs=(_grid([[1, 2, 1, 2], [3, 4, 3, 4]]),),  # tiled, different shape
    )
    assert same_shape_task(task) is False


def test_same_shape_task_accepts_a_same_shape_pair():
    task = ArcTask(
        task_id="t",
        train_inputs=(_grid([[1, 2], [3, 4]]),),
        train_outputs=(_grid([[3, 1], [4, 2]]),),  # rot90, same shape
    )
    assert same_shape_task(task) is True


# ---------- find_short_program: brute-force oracle correctness ----------

def test_find_short_program_finds_a_known_one_step_solution():
    task = ArcTask(
        task_id="t",
        train_inputs=(_grid([[1, 2], [3, 4]]),),
        train_outputs=(_grid([[3, 1], [4, 2]]),),  # exactly rot90
    )
    program = find_short_program(task, max_depth=3)
    assert program == ["rot90"]


def test_find_short_program_finds_a_known_two_step_solution():
    # rot90 then recolor(1 -> 2) -- recolor is only grounded over colors
    # already present in the current grid, so the target color must already
    # appear post-rotation (both 1 and 2 do here).
    rotated = _grid([[3, 1], [4, 2]])
    target = _grid([[3, 2], [4, 2]])
    task = ArcTask(task_id="t", train_inputs=(_grid([[1, 2], [3, 4]]),), train_outputs=(target,))
    program = find_short_program(task, max_depth=3)
    assert program is not None
    # verify it actually solves it, not just that *a* program of the right length was returned
    state = task.train_inputs
    for move in program:
        state = apply_move(state, move)
    assert is_goal(state, task.train_outputs)


def test_find_short_program_returns_none_when_unreachable_within_depth():
    # a target shape this primitive set can never reach at all (tiling)
    task = ArcTask(
        task_id="t",
        train_inputs=(_grid([[1, 2], [3, 4]]),),
        train_outputs=(_grid([[1, 2, 1, 2], [3, 4, 3, 4]]),),
    )
    assert find_short_program(task, max_depth=2) is None


# ---------- load_task ----------

def test_load_task_parses_the_real_json_schema():
    raw = {"train": [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}], "test": []}
    task = load_task(raw, task_id="abc123")
    assert task.task_id == "abc123"
    assert task.train_inputs == (_grid([[1, 2], [3, 4]]),)
    assert task.train_outputs == (_grid([[3, 1], [4, 2]]),)
