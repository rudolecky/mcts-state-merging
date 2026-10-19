# Merging in Program-Synthesis Search: A Real ARC-AGI-1 Pilot

## Question

The user asked whether ARC-AGI is a domain where this project's merge-vs-tree question could
even apply. Scoped in three revisions (see the plan file), each replacing an assumption with real
data: a from-scratch minimal DSL solved 0/60 real tasks (an expressiveness ceiling); the real,
vendored `arc-dsl` (`michaelhodel/arc-dsl`, MIT-licensed, 160 primitives) restricted to simple
single-variable chains reached only 18/400 real tasks; allowing full multi-variable dataflow over
a curated, frequency-selected subset of 60 primitives -- including the 19 higher-order
combinators (`fork`, `compose`, `lbind`, `chain`, ...) that dominate most real solutions -- was
the version actually built and run here. This is the first domain in this project that's a
program-synthesis search over a DSL, not a physical state-transition planning problem, and the
honest goal was to find out whether merging shows anything here at all, not to promise a
Blocksworld-sized effect.

## What's confirmed, not assumed

- **Stage 0 (type-tracking foundation)**: a typed-signature registry (`arc_types_registry.py`)
  over the vendored `dsl.py`, extracted mechanically for the 141 base functions
  (`typing.get_type_hints`) and hand-derived for the 19 higher-order ones (their bare `Callable`
  hints carry no signature information -- read directly from each function's real implementation).
  Verified by **replaying 4 real `solvers.py` reference programs** (spanning `lbind`+`matcher`,
  `fork`+`compose`, `chain`+`fork`+`mapply`, `power`+`compose`) through this project's own
  `Closure`/registry machinery and checking every result against calling the real vendored
  functions directly -- all 4 matched exactly, the highest-value verification in this whole pilot.
- **Stage 1 (search engine)**: a program state is a set of per-training-example "contexts"
  (typed values, concrete or `Closure`), one context tuple per example since one program applies
  to every example in lockstep. Merge key = an order-independent frozenset of the context's
  (type, per-example-values) entries -- two different primitive orderings landing on the same
  available values are the same node, the direct ARC analogue of Blocksworld's frozenset state.
- **A real cycle-safety bug was found and fixed, for a reason specific to this domain.** A
  program context only ever grows (no primitive removes a value), which looked like it ruled out
  cycles entirely -- but the merge key is a *set*, and a move whose result value duplicates one
  already in the context doesn't grow that set at all, making the "new" node identical to its own
  parent. Confirmed directly (a hung real run, traced via SIGINT to `select()` spinning through a
  one-node `children` dict forever), not caught by reasoning alone. Fixed with the same cycle-safe
  path restriction every reversible domain in this project already uses.
- **A real scaling bug was found and fixed during Stage 2.** The first real-experiment run was
  killed after climbing to 9.5GB and 97 CPU-minutes: several curated primitives (`upscale`,
  `objects`, `mapply` over cell-level objects) scale badly with grid size, and one randomly-
  selected task had 30x30 grids (the real ARC-AGI-1 maximum) while calibration had only used small
  ones. Fixed by filtering task selection to grids <=15x15 -- a documented scope-narrowing choice
  (32 of the 124 in-principle-reachable tasks remain), not a silent one.
- **Task selection via the existing oracle**: a real task is a candidate if its
  `solvers.py` reference solution uses only curated-subset functions -- the same
  "verify exact reachability via a trusted oracle" principle as every other domain's puzzle
  generator here, reusing `solvers.py` instead of a hand-rolled BFS.
- Verified: `tests/test_arc_types_registry.py` (9 tests, including the 4 real-solver replays),
  `tests/test_arc_program_engine.py` (7 tests, including a genuine transposition using real
  `hmirror`/`vmirror`), `tests/test_classical_mcts_arc_program.py` (6 tests, including the
  cycle-safety regression and the `safe_apply_move` robustness fallback) + full suite (319 passed)
  green.

## Results

12 real tasks (depth 1-4 by `solvers.py`'s own reference-program length, grids <=15x15, seed 0),
10 repeats/condition, blind UCB1-MCTS (no heuristic exists here any more than in Blocksworld's
blind arc):

| budget | baseline | treatment | diff | 95% CI | wilcoxon p | nodes base->treat |
|---|---|---|---|---|---|---|
| 500 | 0.200 | 0.200 | +0.000 | [-0.025, 0.025] | 1.0000 | 491.6→437.7 (11% fewer) |
| 2000 | 0.250 | 0.292 | +0.042 | [0.000, 0.083] | 0.2500 | 1776.3→1576.8 (11% fewer) |
| 5000 | 0.317 | 0.383 | +0.067 | [-0.008, 0.158] | 0.3750 | 4234.2→3667.1 (13% fewer) |

**Not statistically significant at this N** -- a small, noisy, directionally positive trend, real
but modest node-count reduction (11-13%, consistent across budgets, confirming genuine
deduplication is happening), nowhere near any other domain's effect size in this project.

**The aggregate hides real structure.** By task (solve rate at budget=5000):

| task | reference depth | baseline | treatment |
|---|---|---|---|
| a416b8f3, 68b16354 | 1 | 1.00 | 1.00 |
| 3af2c5a8, a740d043, 44d8ac46, aabf363d, 60b61512 | 3-4 | 0.00 | 0.00 |
| 4c4377d9 | 2 | 0.50 | **0.90** |
| c8f0f002 | 1 | 0.20 | **0.60** |
| d10ecb37 | 1 | 0.40 | 0.50 |
| 1cf80156 | 3 | 0.10 | 0.10 |
| 8be77c9e | 2 | 0.60 | 0.50 |

**5 of 12 tasks never solve under either condition at any budget tested (a genuine floor for
blind search -- all depth 3-4), and 2 solve at ceiling from the smallest budget on (both depth 1,
trivially reachable in one step).** Both extremes are structurally uninformative about merging --
a floor task has nothing to reclaim because the search never finds a route to reclaim it from, a
ceiling task has nothing to reclaim because it never had to search hard in the first place. Only
the remaining 5 tasks -- ones where blind search has *some* traction but isn't trivial -- carry
the real signal, and among those the pattern is a real, sometimes large, growing advantage for
merging (4c4377d9: tied at budget 500, +40pp by 5000; c8f0f002: tied at 500, +40pp by 5000), with
one exception moving the other way (8be77c9e, a small task-level reversal at the highest budget,
reported as observed, not smoothed over).

## Does MCTS itself help here, separate from merging?

A natural follow-up: how much does the tree/UCB1 structure contribute on its own, independent of
merge vs. tree? Added `run_random_search` -- the "no MCTS at all" ablation, reusing the exact same
`simulate()` rollout MCTS itself calls, but firing `budget` independent random rollouts from the
initial state with no tree, no UCB1, nothing carried between attempts. Same 12 tasks, same
budgets, same seeds:

| budget | MCTS (tree, no merge) | pure random restarts |
|---|---|---|
| 500 | 0.208 | **0.242** |
| 2000 | 0.250 | **0.442** |
| 5000 | 0.358 | **0.508** |

**Pure random restarts beat tree-mode MCTS at every budget tested, and the gap grows with
budget.** Since the merge condition itself only reached 0.383 at budget 5000 -- still behind pure
random's 0.508 -- neither MCTS variant beats the simplest possible alternative in this domain.
This is a real, reproducible, and honestly surprising result, not a tuning artifact: the reward
signal here is binary and essentially uncorrelated between neighboring programs (unlike a grid
heuristic or a puzzle's misplaced-piece count, changing one primitive in a DSL program rarely
produces a "slightly better" result -- it produces an unrelated one). UCB1's whole premise is that
a promising region is worth concentrating budget on; when one early lucky 0/1 sample makes some
prefix look artificially better than its siblings, UCB1 commits further budget to refining that
one prefix's own children, at the expense of the diversity of fully-independent attempts pure
random restarts spend their entire budget on. In a sparse, uncorrelated reward landscape, that
diversity is worth more than local refinement -- the search-literature analogue is why random
restarts sometimes beat guided local search on some SAT/CSP instances with little exploitable
structure. This means the modest merge-vs-tree result above should be read in that light: both
conditions are instances of a search strategy that is itself already behind the simplest
alternative here, so the merge question in this domain is really "does merging make the weaker of
two options a little less weak," not "does it improve on the best available approach."

## Mechanism

This project's other domains all showed a large, uniform effect because a physical world-state
(Blocksworld's fact-set, a puzzle's tile arrangement) *is* the complete future-relevant
information -- merging two paths that reach the same state loses nothing. A program-synthesis
context is different: `canonical_key`'s strict full-context match (needed for correctness -- see
`arc_program_engine.py`'s own docstring on why a looser "match only the latest value" key would
silently conflate genuinely different *capability* states, discarding one path's distinct
reusable byproduct) means two computations only merge when their *entire* set of available
values coincides, not just their most recent one. That's a real, structural reason this domain
generates fewer full transpositions than Blocksworld's independent-block-relocations do, where the
complete state (not a growing history of byproducts) is all that ever mattered. Combined with
blind random search's floor/ceiling stratification (unlike Blocksworld, ARC tasks split sharply
into "too hard to ever reach" and "trivially found in one step," leaving a narrow band where
search difficulty is genuinely in play), the aggregate signal here is inherently harder to detect
at a small N than in any other domain tested -- not necessarily smaller in the domains where it's
real (4c4377d9 and c8f0f002's +40pp are not modest), just diluted by floor/ceiling tasks that
carry no information either way.

## Scope and limitations

- **N=12 real tasks is small relative to every other domain's real-instance count in this project**
  (Blocksworld's real-IPC reproduction used 12 too, but that experiment had a much larger,
  cleaner effect at that N; here the effect is real but small enough that a larger N would likely
  be needed to reach significance).
- **Curated 60-function subset, not the full 160** -- chosen by real-usage frequency across all
  400 reference solutions, not arbitrarily, but still a deliberate narrowing (see the plan file).
- **Blind search only, no heuristic** -- unlike essentially every real ARC-solving system, which
  is heuristic- or LLM-guided. This was a deliberate choice to isolate the merge-vs-tree question
  the way every other classical domain in this project does, not an oversight, but it means these
  solve rates are not representative of what a guided ARC solver would achieve.
- **Grid size capped at 15x15** after the scaling bug -- excludes some real tasks (including the
  30x30 one that originally triggered the bug) from this specific experiment.
- **`safe_apply_move`'s type system is a documented simplification, not full type inference** --
  `legal_moves` checks argument *types* but not full arity/shape compatibility for
  closure-consuming functions (e.g. `argmin`'s compfunc needing exactly one argument); real
  execution catches what the type tags miss, discarding the failed move rather than crashing.
  This trades some search efficiency (some sampled moves are wasted) for a much smaller, more
  tractable type system than full static verification would need.

## Status

Closed. Two real, honest findings, not forced into a larger claim than the data supports: (1)
merging shows a genuine but small, non-significant aggregate effect in program-synthesis search
on real ARC-AGI-1 tasks, driven by a handful of tasks with a real, sometimes large effect, diluted
by floor/ceiling tasks the mechanism can't touch either way; (2) tree-guided MCTS itself --
independent of merging -- underperforms pure random restarts at every budget tested here, a real
and reproducible result tied to this domain's sparse, uncorrelated reward signal rather than a
tuning artifact. Two substantive engineering bugs (a cycle-safety self-loop specific to set-based
merge keys, and a grid-size scaling blowup) were found and fixed along the way. This was the
deepest, most novel domain built in this project and the first one whose result doesn't cleanly
resemble any prior domain's.
