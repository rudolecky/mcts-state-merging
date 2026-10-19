# Does Tree-Guided Search Even Help? Random Restarts vs. Tree-Mode MCTS, Seven Domains

## Question

The ARC-AGI pilot found something that should have been the headline, not a footnote: pure random
restarts (no tree, no UCB1, `budget` independent rollouts from scratch) beat tree-mode MCTS at
every budget tested. Tested for generalization across every classical domain in this project --
Blocksworld, the 8-puzzle, Sokoban (all established large merge-benefit domains), and Connect Four
and Three Men's Morris (this project's two established *null* merge-benefit domains) -- reusing
each domain's own `simulate()` rollout via a one-line ablation
(`run_random_search`, added to all five classical modules plus `classical_mcts_arc_program.py`'s
own ARC version): `budget` independent random rollouts from the start state, no tree, no UCB1,
nothing carried between attempts.

Initially this looked like it tracked cleanly with the existing merge-benefit split, with one open
question flagged in the first version of this doc: all six domains above are *blind* search (no
heuristic, no learned value) -- did the same gap hold when the tree actually has real guidance? A
direct test on ProsQA (`run_random_search` added to `search.py`, matched against vanilla PUCT with
the project's own real learned projection, same budget=15, same n=21 held-out set as
`DEEPSEARCH_COMPARISON_FINDINGS.md`) answers that question, and **the answer overturns what this
doc originally inferred**: real guidance does *not* prevent the gap. See "Real guidance doesn't
prevent it either" below.

**This is still the cleanest unifying result in the whole project**: whether the tree-vs-random
gap *persists* as budget grows, or *closes*, tracks exactly with whether merging helps at all --
but that split is now known to hold independently of whether the tree has real guidance, not as a
consequence of the tree being blind.

## Results

### Persistent gap (large merge-benefit domains)

| domain | budget range | tree | pure random | gap at highest budget |
|---|---|---|---|---|
| Blocksworld | 1000-3000 | 0.115->0.255 | 0.749->0.926 | **67pp**, still growing |
| 8-puzzle | 300-1200 | 0.000->0.127 | 0.538->0.730 | **60pp**, barely closing |
| Sokoban | 300-1200 | 0.015->0.243 | 0.587->0.820 | **58pp**, barely closing |
| ARC-AGI-1 (program synthesis) | 500-5000 | 0.208->0.358 | 0.242->0.508 | 15pp, not closing |

Every one of these gaps is statistically significant at every budget tested (p<0.0001 in every
classical case; p<0.05 for ARC), and none of them show real signs of closing within the budget
ranges this project has tested -- consistent with `BLOCKSWORLD_FINDINGS.md`'s and
`SOKOBAN_FINDINGS.md`'s own established finding that these are exactly the domains with the
densest redundant-transposition structure, and the domains where merging shows this project's
largest effects.

### Transient gap (established null merge-benefit domains)

| domain | budget | tree | pure random | diff |
|---|---|---|---|---|
| Morris | 5 | 0.000 | 0.936 | 0.936 |
| Morris | 10 | 0.001 | 0.996 | 0.994 |
| Morris | 20 | 0.179 | 1.000 | 0.821 |
| Morris | 40 | 0.853 | 1.000 | **0.147, converging** |
| Connect Four | 10 | 0.000 | 1.000 | 1.000 |
| Connect Four | 25 | 0.442 | 1.000 | 0.558 |
| Connect Four | 50 | 0.998 | 1.000 | **0.002, converged (p=0.32, not significant)** |

Both adversarial games show the *same shape* of gap at very low budget (in fact larger in relative
terms than the "persistent" domains at their own low end) -- but here the gap **closes rapidly** as
budget grows, and by a modest budget (40 for Morris, 50 for Connect Four) tree-mode has essentially
caught up to random, with Connect Four's final difference no longer statistically distinguishable
from zero.

### Real guidance doesn't prevent it either (ProsQA, real learned projection)

The previous version of this doc treated "real guidance appears to prevent the persistent gap" as
an open inference from `DEEPSEARCH_COMPARISON_FINDINGS.md`'s own baseline numbers (vanilla PUCT and
DeepSearch-style selection both scored 24-48% on ProsQA, well above the near-zero collapse seen in
blind classical baselines). Directly tested via `run_random_search` added to `search.py` (a fresh
K=1 analogue of `expand_node`'s own boundary logic, since this harness has no pre-existing
`simulate()`), matched against vanilla PUCT (`merge_enabled=False`, the project's real learned
projection, budget=15) on the exact same n=21 held-out ProsQA set used throughout this project:

| domain | budget | tree (real projection guidance) | pure random | diff |
|---|---|---|---|---|
| ProsQA (Qwen2.5-7B-Instruct) | 15 | 0.286 (6/21) | 0.905 (19/21) | **0.619, CI=[0.429,0.810], p=0.0003** |

**Random beats guided tree search here too, by a larger margin than any classical domain except
Blocksworld.** Every problem tree solved, random also solved (0 tree-only wins, 13 random-only
wins, 6 both, 2 neither) -- random is a strict superset of tree's solved set on this run, not just
a better average. This flatly contradicts the previous inference: a real, trained value signal
guiding PUCT's selection did not prevent the persistent-gap failure mode from engaging. Mean
wall-clock per problem was actually *lower* for random (73.1s vs. 159.7s) because
`run_random_search` exits on the first successful walk while `run_search` always spends its full
expansion budget regardless of whether it already solved the problem -- a real asymmetry in the two
loops' stopping conditions, not evidence about compute cost per unit of "budget" (which remains an
accepted cross-condition unit mismatch, same as every other domain here; see Scope below).

## Mechanism: this is an independent predictor of the merge-vs-tree answer

The persistent-vs-transient split lines up exactly with each domain's own established merge
result, via two genuinely different reasons a tree can trail random at low budget:

- **Persistent gap = dense redundant-transposition trap.** In Blocksworld, the 8-puzzle, Sokoban,
  and ARC, once one early branch looks even slightly better than its siblings, UCB1 commits
  further budget to refining it -- and because these domains have combinatorially many different
  action/token sequences reaching the same or an equivalent state, "refining it" mostly means
  re-deriving states the tree has already (redundantly) created elsewhere. This trap doesn't drain
  as budget grows, because the combinatorial space of redundant paths grows too. A memoryless
  random sampler never accumulates this commitment bias. This is exactly the mechanism merging
  fixes, and it's exactly why merging shows this project's largest effects in these same domains.
- **Transient gap = ordinary bandit cold-start noise.** In Connect Four and Morris, the established
  mechanism for the *null* merge result is that adversarial interleaving suppresses redundant
  re-exploration of the same joint state even with reversible moves (`MORRIS_FINDINGS.md`) or that
  `untried_moves` bookkeeping structurally prevents duplicate nodes in the first place
  (`CLASSICAL_MCTS_FINDINGS.md`). With no redundant-transposition trap available to fall into, the
  only thing that can make tree-mode lose to random here is generic early-budget noise in UCB1's
  Q-estimates -- which is exactly the kind of gap that closes once enough samples accumulate to
  stabilize the estimates, consistent with what's observed. Since this failure mode has nothing to
  do with duplicate-state redundancy, merging correctly has nothing to fix here -- matching the
  established null perfectly.

**This means "does pure random beat tree, and does the gap persist" is a cheap, independent
diagnostic for "will merging help here" -- discoverable without building any merge machinery at
all.** Formalized as a general checklist in `RANDOM_BASELINE_CHECKLIST.md`.

**Why doesn't real guidance prevent the persistent gap on ProsQA? Both candidate mechanisms are
real, and the budget-allocation one dominates.** Distinguished directly (`search_oracle.py`): swap
the noisy learned projection for a perfect, noise-free ground-truth value computed straight from a
ProsQA instance's own known fact graph (1.0 if the trace so far is well-formed and can still reach
the target, 0.0 the moment it's wrong -- no LLM, no training data, no possible noise), and re-run
the identical tree-vs-random comparison, same n=21 set, same budget=15:

| condition | solved | vs. pure random (0.952) |
|---|---|---|
| tree, learned projection | 0.286 (6/21) | diff=0.619, CI=[0.429,0.810], p=0.0003 |
| tree, perfect oracle value | 0.476 (10/21) | diff=0.476, CI=[0.286,0.668], p=0.0016 |
| pure random | 0.952 (20/21) | -- |

Both mechanisms are real:
1. **Guidance quality matters.** Swapping the noisy learned projection for a perfect oracle value
   raises tree's solve rate from 28.6% to 47.6% (paired diff=+0.190, p=0.0455; oracle beat the
   learned projection on 4 problems, never lost on any). A better signal genuinely helps PUCT
   exploit better.
2. **But budget allocation is the dominant limiter.** Even with a *perfect, noise-free* value
   function, tree still loses to pure random by 47.6 points (p=0.0016) -- a gap nearly as large as
   the worst classical domains, and not meaningfully smaller than the learned-projection gap's
   order of magnitude. Perfect guidance closed under half the total gap; the rest is structural:
   each of the 15 iterations expands exactly one leaf by one depth level, and with a 3-5 step
   required chain, a single early misstep still consumes most of the remaining budget
   re-exploring from a bad node, however correctly that node's badness is now recognized. Pure
   random's 15 fully-independent fresh tries at the *entire* chain each time isn't vulnerable to
   this, regardless of guidance quality.

**Guidance quality is a real but secondary lever; how a small budget gets allocated across a tree
is the primary cause of the persistent gap on ProsQA.** This refines rather than simply confirms
`RANDOM_BASELINE_CHECKLIST.md` item 2: don't expect *any* amount of value-signal improvement alone
to close a persistent gap if the tree still spends its budget one-leaf-per-iteration on a
multi-step chain -- the allocation structure needs fixing too (e.g. a per-branch budget floor,
or wider/shallower expansion), not just the value estimator.

## Scope and limitations

- Five of the seven domains tested use *blind* search (no heuristic, no learned value); ProsQA
  uses real learned guidance (a trained projection) and still shows a persistent gap of the same
  or larger magnitude -- so gap-persistence is not gated on the search being blind. The originally
  inferred "real guidance is an independent third lever preventing this failure mode" (from
  `DEEPSEARCH_COMPARISON_FINDINGS.md`'s baseline numbers alone, before this direct test existed) is
  now superseded by this section's direct result and should not be relied on.
- Modest N per domain (10-15 puzzles/positions, 20-50 repeats for classical; n=21 for ProsQA,
  matching this project's own established LLM trust-bar) -- enough for clear statistical
  significance at every reported point, not independently re-verified at a larger N.
- The "transient" domains were calibrated to their own much lower budget ranges (Morris/Connect
  Four saturate almost immediately at the budgets the "persistent" domains needed) -- a real,
  expected difference in scale given how much smaller these games' effective search depth is, not
  a methodological inconsistency.
- ProsQA's `budget` counts independent full walks for random vs. expansions for tree -- the same
  cross-condition unit mismatch already accepted for every other domain here (a "unit" of budget
  costs a different amount of real compute in each condition); wall-clock is not matched and should
  not be read as a compute-adjusted comparison (see the random-vs-tree timing note above).

## Status

Closed, now including both a direct (not inferred) test of the real-guidance case and a follow-up
oracle ablation resolving why guidance didn't fix it. Generalizes across every domain tested, blind
or guided: whether the tree-vs-random gap persists or closes still tracks with whether merging
helps. Persistence doesn't depend on the tree lacking real guidance, and isn't even fully resolved
by *perfect* guidance -- ProsQA's PUCT arm loses to random by 62pp with a real learned projection
and still by 48pp with a noise-free oracle value, confirming budget allocation (not guidance
quality) is the dominant cause of this specific persistent gap.

A further follow-up (`REAL_HEURISTIC_MERGE_FINDINGS.md`) closes the loop the other direction: gave
Blocksworld, the 8-puzzle, and Sokoban -- the domains where merging matters most -- a real
heuristic strong enough to make tree beat random, then re-tested whether merging still helps on top
of that. It does, substantially, for Sokoban (cleanly, significantly) and the 8-puzzle
(consistently, though underpowered at n=8); Blocksworld's own answer is genuinely inconclusive at
n=7-8, not a confirmed null. A bonus comparison found GUCT-Uniform's own real heuristic-guided
algorithm has such a severe persistent-gap pathology that pure random restart eventually overtakes
it outright (by budget=4000, p=0.036) -- the sharpest illustration yet that algorithmic
sophistication alone never guarantees beating random.

A final extension (`RUBIKS_CUBE_FINDINGS.md`) built a full-fidelity 2x2x2 Rubik's Cube domain
(exact 3,674,160-state rotation-group representation, verified against the known state count and
the published quarter-turn-metric god's number of 14) specifically for its extreme transposition
density. Oracle-guided tree beats random as decisively as anywhere in this project (reaching 1.000
at every difficulty tier given enough budget, vs. random capping near 0.05). But merging didn't
just fail to add value on top -- it was **exactly 0.000 at every single budget and difficulty
tested, including every point where tree-mode reached a perfect 1.000** (p=0.008 repeatedly, the
floor p-value at n=8). This is a *third* mechanism, distinct from the classical domains' "merging
helps" and Connect Four's "merging is structurally null": for a domain transposition-dense enough,
avoiding "redundant" node creation removes the very thing keeping tree-mode's search moving
forward, rather than removing something holding it back.

A direct follow-up resolved *why*, and fixed it. Two threshold-based mitigations (capping how many
parents or how many visits a merged node can absorb) both hit an identical all-or-nothing cliff --
any cap allowing genuine pooling reproduced the full 0.000 failure, ruling out "too much pooling"
as the mechanism and pointing at the backup rule itself. Porting `guct_uniform_blocksworld.py`'s
exact algorithm to the cube (Full Bellman min-backup + LCB1-Uniform selection, already proven
elsewhere in this project) eliminated the harm completely: merge and tree became statistically
indistinguishable in solve rate at every budget/difficulty tested (`merge_diff` exactly 0.000,
CI=[0.000,0.000], everywhere), while merge still delivered a consistent ~57% node-count reduction
-- the classic dedup efficiency benefit, now showing up cleanly. The lesson: this domain's
merge-hurts finding was never really about transposition density being "too high" for merging to
handle -- it was about MC-average backup diluting a value signal that Full Bellman min-backup
doesn't dilute at all.

The same swap on the 8-puzzle and Sokoban (`REAL_HEURISTIC_MERGE_FINDINGS.md`) -- domains where
merging already *helped* under UCB1+MC-average, the opposite starting point -- produced the
identical parity pattern there too: `diff` exactly 0.000 at all 16 budget points tested, merge
using 20-35% fewer nodes throughout. This was initially (and wrongly) written up as an algorithm-
level guarantee: `_lcb1_uniform_score` reads only a private per-edge value, never a shared node's
pooled statistic, so at any single node merging can't change which child gets picked -- true, but
it doesn't prove solve rate always ties. `guct_uniform_blocksworld.py`'s own already-established
result is the direct counterexample: tree-mode GUCT-Uniform on Blocksworld stays flat at
0.030-0.050 across a 40x budget range while merge climbs to 98.7% -- the identical algorithm,
massively different outcome. The corrected reading: merging is never *harmful* under GUCT-Uniform,
but whether it adds solve-rate *benefit* still depends on the same thing it always did -- whether
tree mode's own unmerged growth gets trapped and fails within budget on its own, exactly the
persistent-vs-transient gap distinction this doc already established for blind search. It predicts
GUCT-Uniform's tree-vs-merge answer too, not just UCB1's: persistent on Blocksworld (dense
transpositions, depth-first-committing selection), transient-to-nonexistent on the cube/8-puzzle/
Sokoban at the settings tested. A fifth domain, real ARC-AGI tasks, initially looked like it leaned
Blocksworld's direction at n=12 (2 informative tasks, both favoring merge), but retested at n=30
with wider move sampling the effect vanished: diff is essentially zero at budget=2000 and mildly
negative at budget=500 -- parity, like the cube/8-puzzle/Sokoban, not a persistent gap. The smaller
run's apparent lean was two data points agreeing by chance, not a real signal a larger sample
sharpened -- a useful reminder in its own right that a "suggestive, underpowered" result should be
retested before being trusted, not just flagged and left standing.

A last mechanism trace closed the remaining open question: why Blocksworld alone, among five
domains under GUCT-Uniform, shows a persistent gap. Instrumenting `new_nodes`/`merge_hits`
accounting identically across Blocksworld, the cube, and ARC-AGI (`REAL_HEURISTIC_MERGE_FINDINGS.md`)
shows it takes two factors together, not one: dense transpositions (redundant structure merging can
exploit) AND tree mode's own unmerged growth being structurally unable to solve within budget on its
own. The cube has the first without the second (tree reaches the same ceiling merge does, just
slower); ARC has the second mildly without the first (DSL composition is mostly order-sensitive, so
merge hits stay rare); Blocksworld alone has both, which is why it alone gets a dramatic rescue
rather than parity. No further follow-up queued by default.
