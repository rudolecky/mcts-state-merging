# Classical MCTS Merging on Blocksworld (No LLM)

## Question

Reviewed arXiv:2405.18248 ("Extreme Value Monte Carlo Tree Search for Classical Planning") at
the user's request. Its own contribution -- a better bandit confidence bound (UCB1-Uniform) for
tree-shaped MCTS in classical STRIPS planning -- doesn't touch merging or transposition tables
anywhere; every variant it tests still expands exactly one child per action from a plain tree
open list, the same structural anti-duplicate guarantee this project's classical UCB1 and PUCT
modules already have. By this project's own standing theory, that bound-formula refinement
wouldn't change the merge-vs-tree question either way.

What the paper *did* surface is a domain family this project hadn't tested: classical PDDL/STRIPS
planning, using **Blocksworld** (the canonical benchmark) and **Pyperplan** (the exact planner the
paper itself uses) as the underlying primitives -- so this is a genuine test against that paper's
own literature context, not just a domain that resembles one. Blocksworld's four actions are
genuine inverse pairs (`pickup`/`putdown`, `stack`/`unstack`), and independent block-pairs commute
trivially (stacking A-on-B then C-on-D reaches the identical state as C-on-D then A-on-B) --
directly analogous to the 8-puzzle's and Sokoban's reversibility, but with a much larger,
combinatorially richer action space (50 grounded operators at 5 blocks) than any domain tested so
far. Prediction: should resemble the 8-puzzle/Sokoban pattern (large positive), not Connect
Four's/Morris's (null).

## Setup

- `src/mcts_phase0/datasets/blocksworld_engine.py`: grounds all `pickup`/`putdown`/`stack`/
  `unstack` operators for N blocks directly as `pyperplan.task.Operator` objects (no PDDL text
  parsed -- the grounded STRIPS encoding is standard and unambiguous enough to generate directly).
  Goal: all blocks in one numeric-order tower. `scrambled_state`: seeded random valid
  configurations (support restricted to already-placed blocks, guaranteeing a forest, never a
  cycle). `generate_puzzles`: seeded rejection sampling, each candidate verified for an *exact*
  optimal-plan-length match via `pyperplan.search.breadth_first_search` before being kept -- the
  same exact-difficulty-by-construction principle as every other domain here, reusing pyperplan's
  own oracle instead of a hand-rolled one.
- Two non-bugs diagnosed before the real run: all Blocksworld plan lengths from a scrambled state
  to the standard goal are structurally *even* (each relocation is a 2-action pickup/putdown or
  stack/unstack unit) -- `target_plan_length=3` is impossible by construction, not a generator
  bug. At `num_blocks=4` the true total state count is exactly 73 (independently verified by
  brute-force enumeration) -- too small a space for a real experiment, so all real runs use
  `num_blocks=5` (50 grounded operators).
- `src/mcts_phase0/classical_mcts_blocksworld.py`: single-agent, cycle-safe `select()`/
  `run_search` built in from the first version this time (the exact bug class hit and fixed twice
  already this project, for the 8-puzzle and Morris -- not rediscovered a third time). State key
  is the state's own frozenset directly in treatment mode -- already canonical and hashable,
  simpler than every prior domain's node-keying.
- Verified: `tests/test_blocksworld_engine.py` + `tests/test_classical_mcts_blocksworld.py` (17
  new tests, including an explicit cycle-avoidance regression test and an independent
  re-verification of `generate_puzzles`' exactness against `breadth_first_search`) + full suite
  (268 passed) green, zero regressions.

## Results

45 puzzles (15 each at `target_plan_length` 8/10/12, `num_blocks=5`), 50 repeats/condition,
budgets 200-3000, `rollout_depth=30`:

| budget | baseline | treatment | diff | 95% CI | wilcoxon p | nodes base→treat |
|---|---|---|---|---|---|---|
| 200 | 0.001 | 0.158 | +0.156 | [0.072, 0.251] | 0.0033 | 201.0→114.7 (43% fewer) |
| 500 | 0.027 | 0.412 | +0.384 | [0.264, 0.510] | <0.0001 | 501.0→230.5 (54% fewer) |
| 1000 | 0.115 | 0.602 | +0.487 | [0.365, 0.612] | <0.0001 | 1000.0→341.9 (66% fewer) |
| 1500 | 0.178 | 0.773 | +0.596 | [0.474, 0.708] | <0.0001 | 1490.0→402.9 (73% fewer) |
| 2000 | 0.211 | 0.980 | +0.770 | [0.653, 0.875] | <0.0001 | 1959.9→429.9 (78% fewer) |
| 3000 | 0.255 | 1.000 | +0.745 | [0.624, 0.858] | <0.0001 | 2836.0→431.7 (85% fewer) |

By target plan length (baseline / treatment):

| budget | pl=8 | pl=10 | pl=12 |
|---|---|---|---|
| 200 | 0.004 / 0.420 | 0.000 / 0.000 | 0.000 / 0.053 |
| 500 | 0.081 / 0.735 | 0.000 / 0.344 | 0.000 / 0.156 |
| 1000 | 0.345 / 0.912 | 0.000 / 0.511 | 0.000 / 0.383 |
| 1500 | 0.533 / 0.981 | 0.000 / 0.708 | 0.000 / 0.631 |
| 2000 | 0.631 / 1.000 | 0.001 / 0.988 | 0.000 / 0.953 |
| 3000 | 0.745 / 1.000 | 0.019 / 1.000 | 0.001 / 1.000 |

**pl=10 and pl=12: baseline essentially never solves across the entire swept range** (0.000 at
every budget through 1500, still ≤0.019 even at budget=3000 -- 1 or 2 puzzles out of 750 runs).
Treatment reaches 1.000 on every plan length by budget=3000, and is already at 0.63-0.99 by
budget=2000. This is the largest, cleanest effect of any domain tested in this project -- larger
than the 8-puzzle's and comparable to or exceeding Sokoban's, both in solve-rate gap and in
node-count reduction (43% at the lowest budget tested, still climbing to 85% at the highest,
never plateauing within the swept range).

One more artifact worth noting: baseline's mean node count falls slightly *below* budget+1 at the
higher budgets (1959.9 at budget=2000, 2836.0 at budget=3000, instead of 2001/3001). This is a
tree genuinely running out of room to grow: with `select()`'s cycle-safety restricting the
argmax to children not already on the current path, and every transposition consuming its own
separate, isomorphic subtree, the untreated tree can exhaust its own expandable frontier before
spending its full budget. Treatment never shows this -- merged nodes always have somewhere
genuinely new to explore.

> **Corrected 2026-09-24.** That explanation is wrong. In a path-keyed tree a child is never on the
> current path, so `select()` always reaches an untried move or a terminal. The shortfall comes from
> solved baseline runs: `run_search` has no solve-based exit, so a solved run spins on the goal until
> the 20× iteration cap without spending budget. The same pattern was measured on the cube (solved
> tree runs spent 96.6/100, 189/200 and 369/400). See `MERGING_MECHANISM_SUMMARY.md` §4.

## Mechanism

This matches the 8-puzzle's and Sokoban's pattern, not Connect Four's or Morris's, exactly as
predicted -- and does so more sharply than either. The reason is combinatorial density: at 5
blocks there isn't one mobile token (8-puzzle) or one box (Sokoban) generating transpositions --
there are up to 4-5 blocks simultaneously available to relocate at many points in a scrambled
state, and each independent pair of relocations commutes. That means the number of distinct move
*orders* reaching the same resulting state grows combinatorially with how many blocks are
"in play" at once, far faster than the 8-puzzle's single-blank-token reshuffling ever could. A
plain tree pays for every one of those orderings as a separate, isomorphic subtree; merging
collapses all of them into one node the first time any of them is reached, and every subsequent
arrival adds pure signal instead of paying rediscovery cost.

Combined with every other domain in this project, this is the sharpest illustration yet of the
standing mechanism: merging's benefit isn't about reversibility as a checkbox (Morris is
reversible and null), and it isn't about task genre (single-agent puzzle vs. adversarial game).
It's about how densely a search's *own* expansion mechanism, under its own selection policy,
revisits states it has already created nodes for. Blocksworld at 5 blocks generates that density
faster than any domain tested so far, and the result -- a domain where the untreated baseline
can spend its entire budget and still solve only a quarter of the puzzle set at the very top of
the swept range, while the merged treatment saturates to 100% -- is the theory's prediction
playing out at its clearest so far, and against the exact literature and planner the reviewed
paper itself uses.

## Status

Closed. Full sweep (six budgets spanning a genuine floor-to-ceiling range, 45 puzzles across
three plan lengths, 50 repeats/condition), a large, decisive, still-growing-in-node-reduction
positive effect, and a mechanism read that both confirms and sharpens the standing theory. This
was the domain motivated by the arXiv:2405.18248 literature review; no further follow-up domain
is queued by default.
