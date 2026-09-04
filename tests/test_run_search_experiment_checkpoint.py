"""Verification for run_search_experiment.py's checkpointing.

A four-arm ProsQA run is ~3.5h, and before this the runner pickled everything
only after the last problem -- so an interruption discarded the entire spend
(it did, twice). These tests pin the three properties that make a restart
trustworthy: shards are keyed per (problem, arm, tau) -- with tau dropped for
the one arm that cannot depend on it, so a tau sweep reuses its baseline -- a
checkpoint from different search arguments is refused rather than mixed in,
and a write interrupted part-way cannot be mistaken for a complete result.
"""

import json

import pytest

from mcts_phase0.run_search_experiment import (
    _ROW_SCHEMA,
    _fingerprint,
    _shard_path,
    load_arm,
    open_checkpoint,
    save_arm,
)

_ARGS = {
    "dataset": "prosqa", "model": "Qwen/Qwen2.5-7B-Instruct", "tau": 0.01, "K": 4,
    "budget": 15, "held_out_seed": 90, "output_dir": "results/x", "tau_percentile": 40.0,
    "arms": ["baseline", "reject"],
}


# ---------- fingerprint ----------

def test_fingerprint_ignores_only_things_that_cannot_change_a_search():
    base = _fingerprint(_ARGS)
    assert _fingerprint({**_ARGS, "output_dir": "somewhere/else"}) == base
    assert _fingerprint({**_ARGS, "tau_percentile": 10.0}) == base
    # Arms are excluded because arm identity lives in the shard filename, which
    # is what lets a finished 3-arm run be extended by a 4th arm for the price
    # of that arm alone. tau likewise -- see test_tau_is_in_the_shard_name_only...
    assert _fingerprint({**_ARGS, "arms": ["baseline", "treatment", "reject"]}) == base
    assert _fingerprint({**_ARGS, "tau": 0.5}) == base


def test_fingerprint_changes_for_anything_that_does():
    base = _fingerprint(_ARGS)
    for key, value in (("K", 8), ("budget", 30),
                       ("held_out_seed", 91), ("model", "Qwen/Qwen2.5-1.5B-Instruct"),
                       ("dataset", "countdown")):
        assert _fingerprint({**_ARGS, key: value}) != base, key


# ---------- shard naming ----------

def test_shard_paths_are_distinct_per_arm_and_per_problem(tmp_path):
    a = _shard_path(tmp_path, "pq_rompus95_yes", "reject", 0.01)
    b = _shard_path(tmp_path, "pq_rompus95_yes", "baseline", 0.01)
    c = _shard_path(tmp_path, "pq_other_yes", "reject", 0.01)
    assert len({a, b, c}) == 3


def test_shard_names_survive_unsafe_problem_ids_without_colliding(tmp_path):
    """Sanitising an id to be filesystem-safe must not merge two different
    problems into one shard -- a digest of the raw id keeps them apart."""
    a = _shard_path(tmp_path, "cd/3+4=7", "reject", 0.01)
    b = _shard_path(tmp_path, "cd:3+4:7", "reject", 0.01)
    assert a != b
    for path in (a, b):
        assert "/" not in path.name and ":" not in path.name
    # And the sanitised name is actually writable.
    a.parent.mkdir(parents=True, exist_ok=True)
    a.write_bytes(b"x")
    assert a.exists()


# ---------- round trip ----------

def test_save_and_load_round_trip(tmp_path):
    payload = {"schema": _ROW_SCHEMA, "solved": True, "unique_nodes": 12, "elapsed_sec": 3.5}
    save_arm(tmp_path, "prob1", "reject", 0.01, payload)
    assert load_arm(tmp_path, "prob1", "reject", 0.01) == payload
    assert load_arm(tmp_path, "prob1", "baseline", 0.01) is None  # per-arm, not per-problem
    assert load_arm(tmp_path, "prob2", "reject", 0.01) is None


def test_save_leaves_no_temp_file_behind(tmp_path):
    save_arm(tmp_path, "prob1", "reject", 0.01, {"schema": _ROW_SCHEMA, "solved": False})
    assert not list(tmp_path.glob("*.tmp"))


def test_a_truncated_shard_is_recomputed_not_trusted(tmp_path):
    """A write interrupted mid-way must read back as absent, so the arm is
    re-run rather than silently contributing a half-result."""
    path = _shard_path(tmp_path, "prob1", "reject", 0.01)
    save_arm(tmp_path, "prob1", "reject", 0.01, {"schema": _ROW_SCHEMA, "solved": True, "unique_nodes": 9})
    good = path.read_bytes()
    path.write_bytes(good[: len(good) // 2])
    assert load_arm(tmp_path, "prob1", "reject", 0.01) is None


# ---------- open_checkpoint ----------

def test_open_checkpoint_creates_the_dir_and_records_the_fingerprint(tmp_path):
    partial = open_checkpoint(tmp_path, _ARGS)
    assert partial.is_dir()
    meta = json.loads((partial / "_meta.json").read_text())
    assert meta["fingerprint"] == _fingerprint(_ARGS)


def test_open_checkpoint_is_idempotent_for_the_same_config(tmp_path):
    first = open_checkpoint(tmp_path, _ARGS)
    save_arm(first, "prob1", "reject", 0.01, {"schema": _ROW_SCHEMA, "solved": True})
    second = open_checkpoint(tmp_path, _ARGS)
    assert second == first
    assert load_arm(second, "prob1", "reject", 0.01) == {"schema": _ROW_SCHEMA, "solved": True}


def test_open_checkpoint_accepts_a_changed_arm_list(tmp_path):
    """The whole point of per-arm shards: extending a run must not invalidate
    the work already done."""
    open_checkpoint(tmp_path, {**_ARGS, "arms": ["baseline", "reject"]})
    partial = open_checkpoint(tmp_path, {**_ARGS, "arms": ["baseline", "treatment", "reject"]})
    assert partial.is_dir()


def test_open_checkpoint_refuses_a_checkpoint_from_a_different_configuration(tmp_path):
    """Reusing it would produce one results file whose rows came from two
    different experiments."""
    open_checkpoint(tmp_path, _ARGS)
    with pytest.raises(SystemExit) as excinfo:
        open_checkpoint(tmp_path, {**_ARGS, "budget": 999})
    assert "different configuration" in str(excinfo.value)


def test_tau_is_in_the_shard_name_only_for_arms_that_read_it(tmp_path):
    """A tau sweep must reuse the baseline (which never consults
    `find_merge_target`, so its result cannot depend on tau) while keeping the
    treatment arms separate per tau."""
    from mcts_phase0.run_search_experiment import _uses_tau

    assert _uses_tau("baseline") is False
    assert all(_uses_tau(a) for a in ("treatment", "reject", "reject_all"))

    base_a = _shard_path(tmp_path, "p1", "baseline", 0.01)
    base_b = _shard_path(tmp_path, "p1", "baseline", 0.05)
    assert base_a == base_b  # one baseline shared across the sweep

    for arm in ("treatment", "reject", "reject_all"):
        assert _shard_path(tmp_path, "p1", arm, 0.01) != _shard_path(tmp_path, "p1", arm, 0.05)


def test_a_tau_sweep_reuses_the_baseline_and_not_the_treatment(tmp_path):
    partial = open_checkpoint(tmp_path, {**_ARGS, "tau": 0.01})
    save_arm(partial, "p1", "baseline", 0.01, {"schema": _ROW_SCHEMA, "solved": True})
    save_arm(partial, "p1", "reject", 0.01, {"schema": _ROW_SCHEMA, "solved": False})

    partial2 = open_checkpoint(tmp_path, {**_ARGS, "tau": 0.05})  # same checkpoint, new tau
    assert load_arm(partial2, "p1", "baseline", 0.05) == {"schema": _ROW_SCHEMA, "solved": True}  # reused
    assert load_arm(partial2, "p1", "reject", 0.05) is None  # recomputed


def test_output_filename_distinguishes_both_tau_and_the_arm_set():
    """Calls the production function, not a copy of it. The previous version
    of this test reimplemented the naming rule in its own body, so deleting
    the rule from `main()` left it passing -- it could not detect the
    regression its docstring was about.

    Both keys matter and both were real bugs: a tau sweep shares one
    --output-dir, and "add a fourth arm cheaply" is a workflow this module
    recommends in its own output."""
    from mcts_phase0.run_search_experiment import results_filename

    assert results_filename(90, 0.01, ["baseline", "reject"]) == \
        "search_results_seed90_tau0.01_baseline-reject.pkl"
    # tau distinguishes sweep points
    assert results_filename(90, 0.01, ["baseline", "reject"]) != \
        results_filename(90, 0.003, ["baseline", "reject"])
    # the arm set distinguishes an extended run from the run it extends
    assert results_filename(90, 0.01, ["baseline", "reject"]) != \
        results_filename(90, 0.01, ["baseline", "reject", "treatment"])
    # arm order must not matter
    assert results_filename(90, 0.01, ["reject", "baseline"]) == \
        results_filename(90, 0.01, ["baseline", "reject"])
    # a baseline-only run has no tau dependence, so no tau tag
    assert results_filename(90, 0.01, ["baseline"]) == "search_results_seed90_baseline.pkl"


def test_arm_row_is_the_single_source_of_the_row_schema():
    """Guards resume against a row-key rename, which the previous version of
    this test could not: it compared a hardcoded dict against itself after a
    round-trip, so renaming a key in `main()` left it green while resume
    silently produced rows the summary could not read."""
    from mcts_phase0.run_search_experiment import _ROW_SCHEMA, arm_row

    class _Node:
        def __init__(self, kind):
            self.kind = kind
            self.is_terminal_correct = kind == "answer"

    class _Graph:
        root_id = 0
        nodes = {0: _Node("root"), 1: _Node("step"), 2: _Node("answer")}
        new_nodes = 2
        merge_hits = 0
        rejected = 3
        expansions_used = 7

    row = arm_row(_Graph(), elapsed_sec=1.5, K=4)
    assert row["schema"] == _ROW_SCHEMA
    assert row["unique_nodes"] == 2  # step + answer, not the root
    assert row["llm_samples"] == 28  # expansions_used * K
    assert row["expansions_used"] == 7 and row["rejected"] == 3
    # Everything the summary and the analyser read must be present.
    for key in ("solved", "unique_nodes", "new_nodes", "merge_hits", "rejected",
                "expansions_used", "llm_samples", "elapsed_sec", "graph"):
        assert key in row, key


def test_a_shard_from_a_different_row_schema_is_recomputed(tmp_path):
    """A stale-schema shard must read back as absent. Otherwise the summary
    raises a KeyError *after* the results file has already been written."""
    from mcts_phase0.run_search_experiment import _ROW_SCHEMA

    save_arm(tmp_path, "prob1", "reject", 0.01, {"schema": _ROW_SCHEMA + 99, "solved": True})
    assert load_arm(tmp_path, "prob1", "reject", 0.01) is None
    # An unversioned row (predating the schema field) is likewise not trusted.
    save_arm(tmp_path, "prob2", "reject", 0.01, {"solved": True})
    assert load_arm(tmp_path, "prob2", "reject", 0.01) is None
    # And a row that is not a dict at all.
    save_arm(tmp_path, "prob3", "reject", 0.01, ["not", "a", "row"])
    assert load_arm(tmp_path, "prob3", "reject", 0.01) is None


def test_fingerprint_covers_every_argument_the_real_parser_defines():
    """The fingerprint tests above use a 9-key stand-in; the real parser has
    ~30. This asserts the "only" in `_FINGERPRINT_IGNORED`'s name against the
    actual argument set, so adding a key to that set by mistake fails here."""
    import sys
    from unittest import mock

    import mcts_phase0.run_search_experiment as mod
    from mcts_phase0.run_search_experiment import _FINGERPRINT_IGNORED, _fingerprint

    captured = {}
    real_parse = None

    class _Stop(Exception):
        pass

    def fake_parse_args(self, *a, **kw):
        captured["args"] = real_parse(self, [])
        raise _Stop

    import argparse
    real_parse = argparse.ArgumentParser.parse_args
    with mock.patch.object(argparse.ArgumentParser, "parse_args", fake_parse_args):
        try:
            mod.main()
        except _Stop:
            pass
    defaults = vars(captured["args"])
    assert len(defaults) > 20, f"only captured {len(defaults)} args"

    # Perturb each key in turn; the ones that do NOT change the hash must be
    # exactly the documented exclusions.
    base = _fingerprint(defaults)
    inert = set()
    for key, value in defaults.items():
        if isinstance(value, bool):
            perturbed = not value
        elif isinstance(value, (int, float)):
            perturbed = value + 7
        elif isinstance(value, list):
            perturbed = list(value) + ["zzz"]
        else:
            perturbed = f"{value}_perturbed"
        if _fingerprint({**defaults, key: perturbed}) == base:
            inert.add(key)
    assert inert == set(_FINGERPRINT_IGNORED), (
        f"keys that do not invalidate the checkpoint: {sorted(inert)}; "
        f"documented exclusions: {sorted(_FINGERPRINT_IGNORED)}")
