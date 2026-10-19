# Republishing the Two ARC Experiments Under Deterministic Move Ordering

## Why

`legal_moves` returned `list(<set of tuples containing strings>)`. Set iteration follows hash
order, Python randomizes string hashes per process, so the move list order — and therefore every
`rng.choice(untried_moves)` and every `rng.shuffle(moves)` inside `simulate` — depended on which
process the search ran in. Every ARC number produced before the fix is process-dependent.

That was demonstrated, not inferred: the same state and seed under two `PYTHONHASHSEED` values
returned different orderings; re-running a completed experiment config from its own saved args and
seeds disagreed on 2 of 12 per-task solve rates (one task 0.4 vs 0.0); and two runs of one config
minutes apart reported tree solve rates of 0.192 and 0.200.

Both affected experiments are re-run here at their published settings with `sorted(moves, key=repr)`
in place. Two independently launched processes now agree to every printed digit, which was
impossible before.

## 1. Classical blind UCB1 merge-vs-tree (`ARC_AGI_FINDINGS.md`)

12 tasks, depth 1–4, grids ≤15×15, seed 0, 10 repeats, rollout depth 5.

| budget | | tree | merge | diff | 95% CI | p |
|---|---|---|---|---|---|---|
| 500 | published | 0.200 | 0.200 | +0.000 | [−0.025, 0.025] | 1.0000 |
| 500 | **deterministic** | 0.208 | 0.208 | **+0.000** | [−0.025, +0.025] | 1.0000 |
| 2000 | published | 0.250 | 0.292 | +0.042 | [0.000, 0.083] | 0.2500 |
| 2000 | **deterministic** | 0.233 | 0.275 | **+0.042** | [−0.033, +0.125] | 0.3750 |
| 5000 | published | 0.317 | 0.383 | +0.067 | [−0.008, 0.158] | 0.3750 |
| 5000 | **deterministic** | 0.325 | 0.333 | **+0.008** | [−0.033, +0.067] | 1.0000 |

The published verdict — "a small, noisy, directionally positive trend, real but modest node-count
reduction, nowhere near any other domain's effect size" — survives. But **the largest number in
that table does not**: the budget-5000 merge effect falls from +0.067 to +0.008, an 8× reduction,
and its CI now sits almost symmetrically around zero. Budgets 500 and 2000 reproduce their diffs
exactly (+0.000, +0.042).

Node reduction is unchanged in character: 412.5→371.3 (10%), 1580.3→1383.3 (12%), 3749.2→3164.1
(16%), against the published 11–13%. The deduplication was always real; it is the accuracy claim at
the top budget that was carrying hash noise.

## 2. GUCT-Uniform ARC (`guct_uniform_arc_experiment`)

30 tasks, seed 0, 10 repeats, moves-per-node 40. Note this arm also carries the heuristic fix
(`_arc_heuristic` now evaluates at the best context index rather than `ctx[-1]`, so it is a function
of the merge key), which was separately shown to move both arms equally and leave the merge-vs-tree
difference unchanged.

| budget | | tree | merge | diff | p |
|---|---|---|---|---|
| 500 | published | 0.130 | 0.107 | −0.023 | 0.0578 |
| 500 | **deterministic** | 0.120 | 0.120 | **−0.000** | 0.8738 |
| 2000 | published | 0.140 | 0.143 | +0.003 | 0.7969 |
| 2000 | **deterministic** | 0.143 | 0.160 | **+0.017** | 0.3571 |

**The published borderline negative at budget 500 does not survive.** −0.023 at p=0.058 — the
closest thing to a significant merge *penalty* under GUCT-Uniform anywhere in this project — becomes
exactly −0.000 at p=0.87. Budget 2000 stays null in both.

Earlier in this work I attributed that shift to the `is_goal` fix. That was wrong, and the
correction is recorded in `ARTICLE_DERIVED_MERGING_FINDINGS.md`: with old and new goal semantics run
side by side under deterministic ordering, the difference is nil. The mover was the hash-order
nondeterminism.

## What changes in the conclusions

Nothing about the project's thesis. ARC remains the domain where merging does approximately nothing,
which is what both tables said before and say now.

What changes is which specific numbers can be quoted:

- `ARC_AGI_FINDINGS.md`'s budget-5000 row (+0.067) should be read as +0.008.
- `BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`'s framing of ARC as showing a marginal merge penalty under
  GUCT-Uniform (p=0.058) no longer has support.
- Any future ARC claim at these effect sizes needs the determinism fix in place first, because the
  noise it injected was the same size as the effects being measured. That is the transferable
  lesson, and it is the same one Phase 0 taught when ρ=0.306 evaporated to 0.145 at 3× the data.

The classical Blocksworld, Sokoban, 8-puzzle, cube and Connect Four results are untouched — their
`legal_moves` are list comprehensions over ordered structures, verified, with no set iteration
anywhere.

## Addendum, 2026-10-02: the GUCT-Uniform numbers above carried an early-termination bug

`guct_uniform_arc.py`'s `run_search` ended the search when every remaining untried move at the
root failed to execute — even though the root still had live children — and, below the root,
closed the incoming edge unconditionally, pruning the leaf's existing subtree. An audit measured it
stopping 23/120 tree and 16/120 merge runs after ~15–24 nodes at budget 500. Fixed by closing (or,
at the root, stopping) only when no live child edge is left, which is the fix the other GUCT ports
already had; a regression test fails on the old code (the search stopped after 1 of 20 expansions).

Re-run with the published arguments (30 tasks, seed 0, 10 repeats, moves-per-node 40). The run was
killed at task 28/30 of budget 2000 because it was thrashing swap alongside a 7B LLM job; the
per-task counts below are recovered from its log, compared task by task with the deterministic run
above:

| budget | tasks | | tree | merge | merge − tree | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| 500 | 30/30 | deterministic (bug) | 0.120 | 0.120 | −0.000 | 0.87 |
| | | **fixed** | 0.133 | 0.130 | **−0.003** | 0.58 |
| 2000 | 27/30 | deterministic (bug) | 0.159 | 0.178 | +0.019 | 0.36 |
| | | **fixed** | 0.200 | 0.167 | **−0.033** | 0.26 |

The fix raises both arms slightly (tree +0.013 / +0.041, merge +0.010 / −0.011; none significant)
and **the GUCT-Uniform ARC null survives**: merge − tree is null at both budgets. Raw log:
`results/guct_uniform_arc_experiment_earlystop_fixed.partial.log`.
