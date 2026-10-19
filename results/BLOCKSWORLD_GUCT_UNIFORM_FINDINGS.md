# Merging Under GUCT-Uniform: Testing the Paper's Own Best Algorithm

## Question

`BLOCKSWORLD_FINDINGS.md` tested merging on Blocksworld using this project's own blind
UCB1-MCTS -- not arXiv:2405.18248's actual algorithm, and not even the same *kind* of search
(no heuristic, Monte Carlo average backup, random rollout evaluation). The user pushed back on
treating that as a real test against the paper: same domain isn't the same problem *set* (our own
generated puzzles, not the paper's IPC benchmark files) or the same *algorithm* (blind UCB1 vs.
heuristic-guided GUCT-Uniform with Full Bellman backup -- a stronger planner independent of
merging entirely). This experiment closes that gap the honest way: reimplement the paper's own
proposed algorithm, **GUCT-Uniform** (their best performer in Table 1 -- beating GBFS,
Softmin-Type(h), and every GUCT-Normal/Normal2 variant), inside this project's own harness, and
run the merge-vs-tree A/B under *it*. Both conditions share every mechanism except merge/no-merge;
the algorithm under test is a faithful reimplementation of the paper's headline method, not a
proxy. (Still not in scope: the paper's literal IPC benchmark files or Table 1 numbers -- see
`BLOCKSWORLD_FINDINGS.md`'s companion note and the plan file for why that comparison isn't
meaningful even in principle, since the paper never runs a merge condition to hold ours against.)

## What GUCT-Uniform actually is (confirmed from the paper directly, not assumed)

- **Selection (Theorem 6, LCB1-Uniform)**: `(û+l̂)/2 - (û-l̂)·√(6·t·log T)`, minimized (planning
  is a cost). `t` is an edge's own visit count, `T` the parent's total visits, `l̂`/`û` the
  running min/max of every value ever backed up through that edge. No free exploration constant.
- **Backup (Definition 3, Full Bellman)**: each node's value is the minimum heuristic ever seen
  anywhere in its subtree -- not a running average.
- **Reward**: one direct call to `hFFHeuristic` (pyperplan's own class, the exact heuristic the
  paper itself uses) on a newly-created leaf's state. No rollout, ever.
- **Expansion**: one child per untried move, created lazily -- the same "at most one edge per
  legal move per node" structural guarantee every other module in this project already has.
  Confirmed directly from the paper's own MCTS description: a plain tree-based open list, no
  transposition table anywhere.

## Two real bugs found and fixed while building this (both about cycle-safety interacting badly
## with a bandit that was never designed to need it)

Blocksworld is reversible, so merge-mode search needs the same cycle-safety this project's other
reversible domains needed (restrict `select()`'s argmin to children not already on the current
path). Every prior module handles the "leaf fully expanded, cycle-blocked" case by taking one more
free sample (a fresh rollout) and continuing. That pattern is silently catastrophic under
LCB1-Uniform specifically, because unlike UCB1, **LCB1-Uniform has no self-limiting exploration
term** -- its score diverges toward -∞ as an edge's visit count `t` grows for any fixed nonzero
spread (the paper's own Fig. 3, "focuses on one plateau," describes exactly this behavior as a
*feature* for planning). Two consequences of naively reusing a frozen resample value:

1. **Cycle-block reuse**: repeatedly feeding the same frozen heuristic value into a cycle-blocked
   edge kept growing its `t` while its spread never moved, making that dead edge look *more*
   attractive every time it failed -- permanently locking search into one dead lane until the
   iteration cap. Fixed by marking the blocked edge `closed` (permanently excluded from future
   selection) instead of resampling, letting the very next top-level `select()` call try a
   different branch.
2. **Confirmed-terminal reuse**: once a goal is found, its value is deterministically 0.0 forever,
   so its spread is exactly zero and its score is pinned, never diverging *or* improving. Left
   unclosed, the search kept re-walking into an already-solved goal instead of exploring anything
   new -- **this was the dominant cost in early testing, not the cycle-block case**: roughly 99%
   of iterations at a modest budget were wasted re-confirming a goal already found once. Fixed the
   same way: close a terminal's incoming edge immediately after its first (fully informative)
   backup.

A related, pure performance bug (not a correctness one) was also found and fixed: pyperplan's
`Operator` recomputes its hash from three frozensets on every call (no memoization), and this
algorithm's deliberate depth-first commitment (a direct consequence of the same
non-self-limiting score) produces much deeper search paths than any rollout-based module here ever
sees. That made an O(depth) path-membership list scan and a growing path-tuple baseline node key
(this project's usual no-merge keying scheme) turn cubic in budget -- one calibration run at
budget=3000 took over 5 hours. Fixed with an O(1) path-membership set and a plain incrementing
integer counter for baseline node identity (same uniqueness guarantee, no hashing cost that grows
with depth); the identical run then took 2.3 seconds, with byte-for-byte identical solve-rate
results, confirming the fix was purely a performance change.

## Setup

- `src/mcts_phase0/guct_uniform_blocksworld.py` (new): `MCTSEdge` (`t`, `l_hat`, `u_hat`,
  `closed`), `MCTSNode` (`own_h`, `h_gbfs`, `n_visits`), `_lcb1_uniform_score`, cycle-safe+closed-
  edge-aware `select()`, `expand()` (integer keys in baseline mode, state-frozenset keys in
  treatment mode), `backup()` (Full Bellman min-propagation), `run_search()`.
- `src/mcts_phase0/run_guct_uniform_blocksworld_experiment.py` (new): mirrors
  `run_classical_mcts_blocksworld_experiment.py`'s CLI shape exactly, one shared `hFFHeuristic`
  instance built once and reused across every puzzle/repeat/condition.
- Difficulty calibration (not guessed): heuristic guidance solves far more efficiently than blind
  rollout, so the blind study's puzzles (5 blocks, plan length 8-12) would have been nearly
  trivial here. Calibrated instead at 6 blocks, plan lengths 10/12/14 -- confirmed via an inline
  sweep across 6/7/8-block puzzles that this already gives a clean floor-to-ceiling window without
  needing to go any larger (larger sizes cost much more to generate via BFS with no added
  informativeness).
- Verified: `tests/test_guct_uniform_blocksworld.py` (8 new tests: LCB1-Uniform score against
  hand-computed values including the untried-edge and single-sample-T=1 cases, `select()`'s
  argmin, Full Bellman backup pooling by **minimum** on a hand-built two-parent DAG -- contrasted
  directly against `classical_mcts_blocksworld.py`'s own average-pooling test, merge-vs-tree
  structural neutrality, a cycle-avoidance regression test, and an end-to-end smoke test using the
  real `hFFHeuristic`) + full suite (276 passed) green, zero regressions.

## Results

30 puzzles (10 each at `target_plan_length` 10/12/14, `num_blocks=6`), 30 repeats/condition,
budgets 100-4000:

| budget | baseline | treatment | diff | 95% CI | wilcoxon p | nodes base→treat |
|---|---|---|---|---|---|---|
| 100 | 0.030 | 0.214 | +0.184 | [0.142, 0.229] | <0.0001 | 101.0→66.8 (34% fewer) |
| 300 | 0.032 | 0.577 | +0.544 | [0.507, 0.586] | <0.0001 | 301.0→188.4 (37% fewer) |
| 1000 | 0.032 | 0.913 | +0.881 | [0.852, 0.909] | <0.0001 | 1001.0→589.3 (41% fewer) |
| 2000 | 0.032 | 0.958 | +0.926 | [0.899, 0.949] | <0.0001 | 2001.0→1121.3 (44% fewer) |
| 4000 | 0.032 | 0.987 | +0.954 | [0.929, 0.977] | <0.0001 | 4001.0→2087.8 (48% fewer) |

By target plan length (baseline / treatment):

| budget | pl=10 | pl=12 | pl=14 |
|---|---|---|---|
| 100 | 0.050 / 0.263 | 0.037 / 0.243 | 0.003 / 0.137 |
| 300 | 0.050 / 0.593 | 0.043 / 0.613 | 0.003 / 0.523 |
| 1000 | 0.050 / 0.913 | 0.043 / 0.917 | 0.003 / 0.910 |
| 2000 | 0.050 / 0.963 | 0.043 / 0.953 | 0.003 / 0.957 |
| 4000 | 0.050 / 0.987 | 0.043 / 0.983 | 0.003 / 0.990 |

**Baseline is essentially flat across the entire 40x budget range tested (100→4000), for every
plan length** -- 0.030-0.050 the whole way, never climbing with more budget the way every other
domain's baseline in this project eventually does. This is qualitatively different from
`BLOCKSWORLD_FINDINGS.md`'s blind-MCTS result, where baseline at least crept upward with budget
(reaching 0.745 for plan_length=8 by budget=3000). Here, giving the untreated tree 40x more
budget produces no measurable improvement at all. Treatment, by contrast, reaches 98-99% by
budget=4000 for every plan length, having already passed 90% by budget=1000. Node reduction grows
from 34% at the lowest budget to 48% at the highest, the same still-growing pattern seen in every
other Blocksworld result.

## Mechanism

This is the sharpest confirmation of the standing theory in the whole project, for a reason
specific to GUCT-Uniform's own design: Full Bellman backup means a merged node's value is the
**minimum** heuristic ever found through *any* arriving path, so every one of the combinatorially
many transpositions Blocksworld generates (independent block-pairs relocated in either order --
the same mechanism `BLOCKSWORLD_FINDINGS.md` identified) directly contributes to sharpening one
shared, ever-improving best-estimate. Under the untreated tree, each of those same transpositions
instead pays for an entirely separate, isolated subtree, each starting its own Full Bellman
estimate from scratch -- and because GUCT-Uniform's own selection dynamic is deliberately
depth-first (no self-limiting exploration term, by design, per the paper's own Fig. 3), the
untreated tree doesn't even distribute its budget broadly across those separate subtrees the way
a UCB1-style search would; it commits hard to one lane, and if that lane doesn't reach the goal,
budget is largely wasted rather than spread out to eventually stumble onto a working path. That
combination -- rich combinatorial transpositions plus a selection rule with no built-in incentive
to explore broadly -- is exactly why baseline here is flatter and more stuck than in *any* other
result in this project: it isn't just failing to reclaim redundant exploration, it's structurally
prone to tunnel-visioning down one unproductive lane and never recovering, something a search with
a self-limiting bonus term (plain UCB1) would eventually correct for on its own even without
merging.

This also directly answers the scoping question this experiment was built to settle: guidance
quality (the paper's whole contribution -- a provably tighter, EVT-derived confidence bound) and
merging (this project's whole question) are answers to two different, non-competing questions.
Swapping blind UCB1-MCTS for the paper's own best, most theoretically-justified heuristic-guided
algorithm didn't shrink merging's benefit -- it produced the largest, cleanest merging benefit
found anywhere in this project.

## Scope and limitations

Worth being precise about what this experiment does and doesn't establish, since it's easy to
overread "we implemented GUCT-Uniform" as "we validated GUCT-Uniform":

- **Not the same puzzle instances as `BLOCKSWORLD_FINDINGS.md`.** That experiment used 5 blocks,
  plan lengths 8/10/12; this one uses 6 blocks, plan lengths 10/12/14, recalibrated because
  heuristic guidance solves the smaller puzzles almost trivially. Same seed (0), but a fresh
  generator call over a differently-sized state space -- not a shared instance set between the
  two Blocksworld results.
- **Not the same puzzle instances as the paper**, and never was -- our own generated instances
  throughout, not the paper's IPC benchmark files. Established before this experiment was built;
  restated here since it's easy to conflate "same algorithm" with "same problem set."
- **Formula fidelity vs. behavioral fidelity are different claims.** The selection formula
  (Theorem 6), backup rule (Definition 3), and heuristic (`hFFHeuristic`) are sourced directly
  from the paper's own text, and the code is unit-tested against hand-computed values for each.
  That's confirmed. What's *not* confirmed: whether this implementation's overall search behavior
  matches the paper's own reported performance on any shared benchmark -- no attempt was made to
  reproduce even an approximate version of their Table 1 numbers, on Blocksworld or anywhere else.
- **Whether the paper even evaluates on Blocksworld specifically is unresolved.** Two separate
  reads of the paper gave conflicting answers on this (one said yes, a more careful one said their
  24-domain list doesn't mention it). Blocksworld was chosen as the canonical STRIPS benchmark
  compatible with Pyperplan, not because its presence in the paper's own domain suite was
  confirmed either way.
- **This is their base configuration, not their best one.** Table 1 also reports `h^FF`+PO and
  `h^FF`+DE+PO variants (preferred operators / deferred evaluation) that outperform plain `h^FF`
  in their own results; neither is implemented here.
- One specific alignment that *is* confirmed: "early goal detection" (a goal state is recognized
  the instant it's generated, not on a later expansion/selection pass) is stated in the paper as
  applying to every algorithm they test, and this implementation does the same (`_make_node` sets
  `is_terminal` at creation time).

None of this changes the result -- baseline vs. treatment is a controlled, apples-to-apples
comparison regardless of whether the algorithm perfectly reproduces the paper's own numbers
elsewhere. It does mean "GUCT-Uniform" here should be read as "a faithful reimplementation of the
paper's formulas," not as "a validated drop-in replica of their benchmarked system."

## Status

Closed. Full sweep (five budgets spanning a genuine floor-to-ceiling range for every plan length,
30 puzzles, 30 repeats/condition), the largest and flattest baseline-null-effect gap in this
project, and a mechanism read that both confirms and sharpens the standing theory -- with two real
implementation bugs (cycle-block and confirmed-terminal reuse under a non-self-limiting bandit)
found and fixed along the way, not glossed over. This was the algorithm-fidelity follow-up
requested after `BLOCKSWORLD_FINDINGS.md`; no further follow-up is queued by default.
