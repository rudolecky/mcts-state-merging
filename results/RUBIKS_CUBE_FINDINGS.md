# 2x2x2 Rubik's Cube: Merging Catastrophically Harms UCB1+MC-Average Search -- Then a Backup-Rule Redesign Fixes It Completely

## Question

Following the classical-domain heuristic sweep (`REAL_HEURISTIC_MERGE_FINDINGS.md`), asked what
other multistep benchmark to try. Recommended a 2x2x2 Rubik's Cube (Pocket Cube): the cube family
has extremely dense move redundancy (opposite-face moves commute, move+inverse cancels, many
distinct sequences reach the same state), exactly the property this project's whole thesis is
about, and it's a well-known AI-search benchmark. User chose full fidelity: proper whole-cube-
rotation canonicalization giving the true 3,674,160-reachable-state space, a full exact-BFS-
distance oracle, matching every other domain's `generate_puzzles(target_distance=...)` convention.

## Engine, verified before anything else was trusted

`rubiks_cube_engine.py` represents every corner's orientation as a full element of the 24-element
cube rotation group (not a hand-derived 0/1/2 twist-count), so every operation -- a face turn, a
whole-cube rotation, canonicalization -- reduces to the same group composition, checkable via group
axioms rather than manually-reasoned sign/parity case analysis. Verification gate, in order (all in
`tests/test_rubiks_cube_engine.py`):

1. Rotation group: exactly 24 distinct signed-permutation matrices with det=+1, closed under
   composition, every inverse cancels correctly.
2. Move mechanics: every one of the 12 generators has order exactly 4 (returns to solved after 4
   applications, not before); every move cancels with its designated inverse.
3. Canonical-form invariant (reference corner always at position 0 with identity orientation) held
   after all 2000 steps of a random walk, not just at the end.
4. **The ground-truth cross-check**: full BFS from solved reached **exactly 3,674,160 states** --
   the well-known, independently-derivable exact figure for this puzzle (`7! x 3^6`). Bonus
   independent confirmation: the resulting diameter (max distance from solved under this 12-move
   generator set) is **exactly 14**, matching the published quarter-turn-metric god's number for
   the 2x2x2 cube.

Both cross-checks matching independently-known values with zero slack (not "close," exactly right)
is strong evidence the design is correct, not a fluke.

## A diagnosed dead end: simple per-piece heuristics structurally can't discriminate here

Tried the natural realistic heuristic (count of corners not in solved position+orientation, the
cube analogue of "misplaced tiles") and a slightly richer one (sum of each corner's own optimal
distance-to-solved computed *ignoring every other corner*, a 24-state BFS per corner). Both
saturate almost immediately: mean heuristic value across a random sample of states levels off by
true distance 4 and stays flat all the way to distance 14 (misplaced-count: ~6.6 the whole way;
per-corner-sum: ~13.3 the whole way). This is a real, structural fact about this puzzle family, not
a calibration failure: a single quarter turn moves 4 of the 8 corners at once, and any one corner's
own alone-distance ceiling is only 3 -- so a handful of moves is enough to make every simple
per-piece heuristic indistinguishable from "very scrambled," regardless of true remaining distance.
This is exactly why real cube solvers use genuine multi-piece pattern databases (joint distance
tables over several pieces at once), which is real, separately-scoped engineering beyond this
plan -- dropped rather than reported as a false "heuristic doesn't help" result. All results below
use the **oracle** value_source (the exact BFS distance) as the sole guidance arm.

## Calibration

Same lesson as every other classical domain: c=1.4 (UCB1's textbook stochastic-reward constant)
badly under-exploits a deterministic value; c=0.02 used throughout. A second calibration surprise
specific to this domain: even with perfect (oracle) guidance, a budget that "should" suffice by
move-count alone (e.g. budget=40 for a 4-move scramble) solved 0/50 -- tracing a single run showed
the tree got to distance 1 from goal but exhausted its budget probing a promising node's 12
candidate moves without yet drawing the 1-in-12 correct one. Budget=100 fixed this completely
(50/50). This reflects genuine budget needed to *locate* the exact right move among many similarly-
scored candidates, not a bug (confirmed by re-testing at higher budgets, which resolved cleanly).

## Results (n=8 puzzles/tier, 30 repeats, c=0.02, oracle value_source)

**tree** = `merge_enabled=False`. **random** = `run_random_search`. **merge** =
`merge_enabled=True`. Stats: bootstrap 95% CI + Wilcoxon on the paired per-puzzle diff.

### td=6

| budget | tree | random | diff(rand-tree) | p | merge | diff(merge-tree) | p |
|---|---|---|---|---|---|---|---|
| 50 | 0.000 | 0.000 | 0.000 | 1.00 | 0.000 | 0.000 | 1.00 |
| 100 | 0.567 | 0.004 | **-0.562** | **0.016** | 0.000 | **-0.567** | **0.016** |
| 200 | 1.000 | 0.029 | **-0.971** | **0.008** | 0.000 | **-1.000** | **0.008** |
| 300 | 1.000 | 0.050 | **-0.950** | **0.008** | 0.000 | **-1.000** | **0.008** |

### td=8

| budget | tree | random | diff | p | merge | diff | p |
|---|---|---|---|---|---|---|---|
| 100 | 0.000 | 0.004 | +0.004 | 1.00 | 0.000 | 0.000 | 1.00 |
| 200 | 0.358 | 0.008 | **-0.350** | **0.008** | 0.000 | **-0.358** | **0.008** |
| 400 | 1.000 | 0.013 | **-0.988** | **0.008** | 0.000 | **-1.000** | **0.008** |
| 600 | 1.000 | 0.017 | **-0.983** | **0.008** | 0.000 | **-1.000** | **0.008** |

### td=10

| budget | tree | random | diff | p | merge | diff | p |
|---|---|---|---|---|---|---|---|
| 200 | 0.000 | 0.000 | 0.000 | 1.00 | 0.000 | 0.000 | 1.00 |
| 400 | 0.362 | 0.000 | -0.362 | 0.063 | 0.000 | -0.362 | 0.063 |
| 800 | 1.000 | 0.000 | **-1.000** | **0.008** | 0.000 | **-1.000** | **0.008** |
| 1200 | 1.000 | 0.000 | **-1.000** | **0.008** | 0.000 | **-1.000** | **0.008** |

**Stage 1 is as decisive as this project has ever seen anywhere**: oracle-guided tree reaches a
perfect 1.000 at every tier given enough budget, while random restart stays near zero throughout
(random peaks at 0.05, at td=6/budget=300 -- easiest tier, most budget).

**Stage 2 is the headline result, and it's the opposite of every other domain in this project**:
merge is **exactly 0.000 at every single budget and difficulty tested, including every point where
tree reaches a perfect 1.000** (p=0.008, the floor p-value at n=8, repeatedly). Merging doesn't
merely fail to help here -- it appears to prevent the search from ever reaching the goal at all,
within the same budget range where blind tree-mode solves every puzzle.

## Mechanism: merging collapses the tree's own escape velocity, it doesn't just avoid redundant work

Traced a single td=10/budget=400 run directly: tree-mode created 401 distinct nodes and got as
close as distance 4 from goal; merge-mode created only 159 nodes (confirming real, heavy merging
was occurring, not a no-op) but only reached distance 8 -- *worse* progress despite (or because of)
far more aggressive deduplication. Re-running merge at 30x more budget (3000 vs 100) never
recovered even one solve at td=6.

**Quantified directly** by instrumenting `run_search` to count, per expansion, whether it created a
genuinely new node or only added a parent pointer to an already-known state (`stats={"new_nodes":
..., "merge_hits": ...}`, backward-compatible optional parameter), then re-running the full
budget/difficulty sweep (n=5 puzzles/tier, 10 repeats, oracle guidance):

| td | budget | new_node_frac (tree) | new_node_frac (merge) | best_dist (tree) | best_dist (merge) |
|---|---|---|---|---|---|
| 6 | 50 | 1.000 | 0.446 | 3.08 | 4.60 |
| 6 | 100 | 1.000 | 0.428 | 0.50 | 4.58 |
| 6 | 200 | 1.000 | 0.414 | 0.00 | 4.58 |
| 6 | 300 | 1.000 | 0.406 | 0.00 | 4.58 |
| 8 | 100 | 1.000 | 0.428 | 4.64 | 6.64 |
| 8 | 200 | 1.000 | 0.414 | 1.02 | 6.58 |
| 8 | 400 | 1.000 | 0.407 | 0.00 | 6.46 |
| 8 | 600 | 1.000 | 0.403 | 0.00 | 6.36 |
| 10 | 200 | 1.000 | 0.414 | 6.30 | 7.72 |
| 10 | 400 | 1.000 | 0.407 | 2.16 | 7.20 |
| 10 | 800 | 1.000 | 0.404 | 0.00 | 6.78 |
| 10 | 1200 | 1.000 | 0.401 | 0.00 | 6.42 |

Two things confirmed with numbers, not just one anecdote:

1. **Tree mode's `new_node_frac` is exactly 1.000 at every single point** -- mechanically
   guaranteed by path-keying (it can never collide with an existing node), not an empirical
   tendency. **Merge mode's stabilizes at 0.40-0.45 everywhere tested** -- roughly *55-60% of its
   entire expansion budget goes to re-linking states it has already discovered*, regardless of
   difficulty or how much budget it's given.
2. **Merge mode's best-distance-found is almost budget-independent.** Tree mode's improves
   monotonically toward 0 as budget grows (it's actively converging on the goal). Merge mode's
   barely moves at all across a 6x budget range at td=8 (6.64 -> 6.58 -> 6.46 -> 6.36) or a 6x
   range at td=10 (7.72 -> 7.20 -> 6.78 -> 6.42) -- consistent with the search circulating within a
   fixed-size local neighborhood rather than making real headway, however much budget it gets.

This flips the mechanism that makes merging help in Blocksworld/8-puzzle/Sokoban. There, path-keyed
tree search is *blind* to revisiting an already-explored state -- it keeps "discovering" the same
territory under different move sequences, wasting budget that merging reclaims for genuine frontier
growth. The cube's transposition density is high enough that the opposite failure mode dominates
instead: **every single expansion in tree mode is guaranteed to create a genuinely new tree node**
(path-keying can never collide), so blind, redundant-looking growth still guarantees the search
keeps reaching further outward, node by node, regardless of how much of that growth is "semantically"
revisiting known states. Merging collapses this same growth into a much smaller set of heavily
-revisited hub nodes near the start (very many different short move sequences reconverge on a small
neighborhood almost immediately) -- and because expansion budget is consumed identically whether an
expansion creates a new node or reconfirms an existing merge target, an ever-larger share of the
budget gets spent re-linking the same small, already-well-explored cluster instead of pushing
outward at all. For a domain this densely connected, avoiding "redundant" node creation removes the
one thing keeping the search moving, rather than removing the thing holding it back.

**This is a genuinely different, third mechanism from anything else found in this project**:
Connect Four's AlphaZero result showed merging can be structurally *null* (no duplicates ever form,
so nothing to reclaim); ProsQA's oracle result showed merging's *absence* wasn't tested but budget
allocation dominated a persistent tree-vs-random gap regardless of guidance; here, merging is
actively *harmful*, and severely so, in a domain whose transposition density is (by design) higher
than anywhere else tested.

## Scope and limitations

- Only the oracle value_source arm was tested at scale; the realistic-heuristic arm was diagnosed
  as structurally non-discriminative (saturates by distance 4) and not run as a full experiment --
  a real limitation of this specific puzzle representation (2x2x2 corners-only, 12 quarter-turn
  generators), not necessarily true of a proper multi-piece pattern-database heuristic, which is
  out of scope here.
- n=8 puzzles/tier -- small by absolute standards, but the effect here is so large (0.000 vs up to
  1.000) that it reaches the floor p-value (0.008) at every meaningful budget regardless; this is
  not a small-N-noise artifact the way some of the classical-domain Stage 2 results were.
- The merge-hurts mechanism is now quantified (new_node_frac + best-distance-found, swept across
  all 3 tiers and 12 budget points, not just one representative run) -- resolved, not a remaining
  gap.
- `value_source="rollout"` (blind UCB1, no heuristic at all) was not tested for this domain --
  every result here already uses real (oracle) guidance; whether the merge-hurts effect also
  appears under blind UCB1 is untested.

## Attempted fix: capping how many parents a merged node can absorb

Given the mechanism above (a small hub of heavily-revisited nodes soaks up 55-60% of the entire
budget), the natural fix is to bound it: `merge_parent_cap` on `ClassicalMCTSConfig` -- once a
node has absorbed `merge_parent_cap` distinct parents, the next expansion that would merge into it
instead falls back to an ordinary path-keyed (guaranteed-fresh) node, same as `merge_enabled=False`
for that one expansion. Swept `merge_parent_cap in {1, 2, 3, None}` at td=8/budget=400 (n=5, 10
repeats):

| cap | solved | new_node_frac | best_dist_mean |
|---|---|---|---|
| 1 | 50/50 | 0.995 | 0.00 |
| 2 | 0/50 | 0.593 | 6.44 |
| 3 | 0/50 | 0.500 | 6.44 |
| None (uncapped) | 0/50 | 0.406 | 6.36 |

**There is no usable middle ground -- this is a cliff, not a dial.** `cap=1` "recovers" full
performance, but for a mechanical reason that makes the recovery hollow: a node can only reach
`parent_count >= 1` after its very first parent, so `merge_parent_cap=1` means a second parent can
*never* attach -- no two paths ever actually get pooled together. `new_node_frac=0.995` (vs. tree
mode's exact 1.000) confirms this: `cap=1` is, behaviorally, indistinguishable from
`merge_enabled=False`. It isn't a fix for merging, it's merging with the one feature that defines
merging (pooling multiple paths onto one node) switched off.

The more informative rows are `cap=2` and `cap=3`: they *do* measurably reduce wasted budget
(new_node_frac climbs from 0.406 to 0.50-0.59, a real, substantial improvement in that specific
metric) -- but solve rate and best-distance-found stay exactly as bad as fully uncapped merging.
**The instant a node is allowed even one real second parent, the full pathology reappears at
essentially undiminished severity**, even though the crude "fraction of budget wasted" metric says
things got meaningfully better. This rules out the simple story that the failure scales smoothly
with how much pooling happens; something else -- most plausibly UCB1 itself: a node with 2+ pooled
parents accumulates visits faster than any single-parent alternative, shrinking its own exploration
bonus and making it look more confidently "the best choice" purely from visit-count arithmetic,
independent of whether it's actually near the goal -- appears to lock in disproportionately once
pooling starts at all, not gradually as pooling increases.

**Follow-up: tried a visit-count throttle instead, to test that hypothesis directly.**
`merge_visit_cap` (same fallback mechanism, keyed on the target node's own `n_visits` rather than
`parent_count`) was swept over a much wider range at the same td=8/budget=400 point:

| visit_cap | solved | new_node_frac | best_dist_mean |
|---|---|---|---|
| 1 | 50/50 | 1.000 | 0.00 |
| 2 | 0/50 | 0.598 | 6.44 |
| 3 | 0/50 | 0.584 | 6.44 |
| 5 | 0/50 | 0.573 | 6.36 |
| 10 | 0/50 | 0.573 | 6.36 |
| 20 | 0/50 | 0.497 | 6.36 |
| 50 | 0/50 | 0.407 | 6.36 |
| None (uncapped) | 0/50 | 0.406 | 6.36 |

**The identical cliff, at the identical location.** `visit_cap=1` "recovers" full performance for
the same hollow reason as `merge_parent_cap=1` (a node reaches `n_visits>=1` after its very first
visit, so a cap of 1 blocks any second visit-through-merge from ever happening -- no real pooling
occurs). Every cap from 2 up to 50 -- a full 25x range, the most generous of which lets a node
absorb 12.5% of the *entire* budget before being throttled -- gives exactly 0/50, statistically
indistinguishable from fully uncapped, even as `new_node_frac` climbs substantially (0.41 to 0.60).

Two independent, principled throttles -- one keyed on distinct parents, one keyed on total
visits -- both show the same all-or-nothing cliff at the same location. This rules out "pooling
gradually gets worse" as the mechanism, and rules out visit-count specifically as *the* missing
variable (the original hypothesis for why `merge_parent_cap` might have failed). **The pathology
looks like it's triggered by the mere existence of any pooled node anywhere in the tree, not by how
much pooling accumulates** -- consistent with a corrupted signal propagating up through `backup()`
into every ancestor on whatever path first reaches a pooled node, rather than staying contained to
the pooled node itself. Neither threshold-based mitigation salvages merging on this domain; a fix,
if one exists, would need to change what gets pooled or how backup propagates through a merged
node, not just how often merging is allowed to fire -- a materially bigger redesign. **Attempted
and confirmed below** ("The real redesign: GUCT-Uniform (Full Bellman min-backup) fixes it").

## The real redesign: GUCT-Uniform (Full Bellman min-backup) fixes it

Both capping mitigations left the same conclusion: a fix needs to change *what* gets pooled and
*how* `backup()` propagates through a merged node, not just how often merging is allowed to fire.
`guct_uniform_blocksworld.py` already solved exactly this class of problem, for the same underlying
reason, on a different domain: it pairs a real heuristic with **Full Bellman backup** (each node's
value is the *minimum* ever seen anywhere in its subtree, not a running average) and **LCB1-Uniform
selection** (Theorem 6 of arXiv:2405.18248, no free exploration constant). Its own merge-vs-tree
result on Blocksworld was merging helping *even more* than under blind UCB1 -- the opposite
direction from this domain's UCB1+MC-average result. Ported the exact algorithm to the cube
(`guct_uniform_rubiks.py`), reward = the exact oracle distance (raw, not `1/(1+h)` -- staying in
the paper's own cost-minimization convention throughout), otherwise unchanged.

**It works.** Full sweep (n=8, 30 repeats, same 3 difficulty tiers):

| td | budget | tree | random | merge | merge_diff | p |
|---|---|---|---|---|---|---|
| 6 | 40 | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 |
| 6 | 60 | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 |
| 6 | 70 | 0.983 | 0.000 | 0.983 | 0.000 | 1.00 |
| 6 | 90 | 1.000 | 0.004 | 1.000 | 0.000 | 1.00 |
| 8 | 50 | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 |
| 8 | 70 | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 |
| 8 | 80 | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 |
| 8 | 100 | 1.000 | 0.004 | 1.000 | 0.000 | 1.00 |
| 10 | 90 | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 |
| 10 | 110 | 0.350 | 0.000 | 0.350 | 0.000 | 1.00 |
| 10 | 130 | 1.000 | 0.000 | 1.000 | 0.000 | 1.00 |
| 10 | 170 | 1.000 | 0.000 | 1.000 | 0.000 | 1.00 |

**`merge_diff` is exactly 0.000, CI=[0.000,0.000], at every single point.** This isn't "close to
tree" the way `merge_parent_cap=1`'s hollow recovery was (that one worked by preventing any real
pooling) -- direct inspection confirmed heavy, genuine merging is occurring (a representative run:
38 of 40 nodes had more than one parent, vs. tree mode's 91 nodes for the same budget) while
producing the *identical* solved/not-solved outcome on every single puzzle-repeat pair, not just
matching on average. Both bandit stages replicate GUCT-Uniform's own Blocksworld shape too: tree
mode alone beats pure random restart just as decisively here as everywhere else in this project.

**Why identical rather than better**: with an *exact* oracle, a node's own value is already fully
correct from its very first visit -- there's no noisy sample to average away, so Full Bellman's
min-aggregation converges to the true answer immediately whether one parent visits a state once or
five parents visit it collectively. Sharing statistics across parents (merging) and duplicating
them per-path (tree mode) end up mathematically equivalent for *what value a node ends up with*,
because there was never any signal-quality benefit left for pooling to capture. What merging *does*
still buy, cleanly, is efficiency: at matched solve rate, merge mode used **56-58% fewer nodes**
than tree mode at every tier checked (td=6/budget=90: 91.0 vs. 39.5; td=8/budget=100: 101.0 vs.
43.8; td=10/budget=170: 171.0 vs. 71.6) -- the classic transposition-dedup benefit, showing up
exactly where it should (search cost) rather than where an already-perfect oracle leaves no room
for it to matter (solve rate).

**This closes the redesign question**: merging's catastrophic harm under UCB1+MC-average was not
an inherent property of this domain's transposition density -- it was specific to that backup rule
diluting a signal that Full Bellman min-backup doesn't dilute at all. The fix that worked is
exactly the one both capping attempts predicted would be needed: change what gets pooled (min, not
mean) and how it propagates (Full Bellman, not MC-average), not how often pooling is allowed.

## Does the efficiency edge ever become a solve-rate edge? Pushed to td=12/14 (near the diameter=14) -- no, and the reason is provable, not empirical

Merge uses ~57% fewer nodes than tree at matched solve rate -- does that efficiency ever convert
into solving something tree can't, at a fixed budget? Tested at the hardest difficulties this
domain has (td=12, td=14 -- 14 is the exact diameter found during engine verification, the
hardest possible instance), narrowing the budget search to the sharpest transition threshold in
each case:

| td | budget | tree | merge |
|---|---|---|---|
| 12 | 130 | 0/50 | 0/50 |
| 12 | 140 | 46/50 | 46/50 |
| 14 | 155 | 0/50 | 0/50 |
| 14 | 165 | 48/50 | 48/50 |

**Still exactly identical, even at the single sharpest floor-to-ceiling transition point tested.**
`_lcb1_uniform_score(edge, T)` (the function `select()` uses to choose among a node's children)
takes only the **edge** object (`t`, `l_hat`, `u_hat`, populated purely from this specific parent's
own backup history) and `T` -- it never reads `h_gbfs` off the shared child node, so at any single
node, merging can never change *which* child gets picked among that node's own candidates. That
much is provably true of the algorithm's code, on any domain. **What it does not prove, and what an
earlier version of this doc overstated, is that merging's efficiency gain can never become a
solve-rate gain anywhere** -- see "Correction" below: `guct_uniform_blocksworld.py`'s own
already-established result shows exactly that happening, dramatically, on a different domain. On
the cube specifically, at the difficulties and budgets tested here, tree mode's own redundant
growth (every expansion guaranteed to create a new node, however many of those nodes are
semantically duplicate) still reaches the same ceiling merge does -- an empirical fact about this
domain at these settings, not a law the algorithm's code guarantees for every domain. The ~57%
node-count reduction is real value (cheaper memory/wall-clock per search), just not a value that
this bandit's own selection logic is wired to spend on trying harder.

## Status

Closed, including a resolved redesign. The full arc: merging catastrophically harms this domain
under UCB1+MC-average (drives solve rate to exactly 0.000 everywhere tree reaches 1.000); two
independent threshold-based mitigations (capping parents-per-node, capping visits-per-node) both
hit an identical all-or-nothing cliff, ruling out "too much pooling" as the mechanism and pointing
at the backup rule itself; porting GUCT-Uniform's Full-Bellman-min-backup + LCB1-Uniform selection
(already proven on Blocksworld) eliminates the harm completely -- merge and tree become statistically
indistinguishable in solve rate at every budget/difficulty tested here, including at td=12/td=14
(the exact diameter, the hardest instances this domain has), while merge still delivers a
consistent ~57% node-count reduction, the classic efficiency benefit, with zero cost.

**Correction, caught before over-claiming further**: an earlier version of this section said this
solve-rate parity was "structurally guaranteed on any domain," reasoning from the fact that
`_lcb1_uniform_score` never reads a shared node's pooled value. That reasoning proves something
real but narrower than what was claimed: merging can never make GUCT-Uniform's search *worse* than
tree mode would have found on its own (a merged node's Full-Bellman-min value is always at least
as good as any single duplicate's, and each parent's own selection math is unaffected in form by
merging). It does **not** prove tree mode itself always succeeds -- and `guct_uniform_blocksworld.py`'s
own already-established result (`BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`, n=30, budgets 100-4000,
p<0.0001 throughout) is a real counterexample: tree-mode GUCT-Uniform on Blocksworld stays flat at
0.030-0.050 across the *entire* 40x budget range, while merge climbs to 98.7% by budget=4000 -- a
massive, real solve-rate gap this project's own prior work already measured.

**The corrected, complete picture**: merging is *never harmful* under GUCT-Uniform (confirmed on
all four domains now, cube included), but it *is* still capable of providing a large solve-rate
benefit -- exactly when tree mode's own unmerged, redundant growth gets trapped and can't reach the
goal within budget on its own. That's a domain-specific question, not an algorithm-level constant:
Blocksworld's combinatorial transposition density is dense enough, and GUCT-Uniform's deliberately
depth-first selection (no exploration term) commits hard enough to one lane, that tree mode's
budget gets consumed re-deriving already-known shallow territory and never escapes -- the same
"persistent gap" signature this project's own `RANDOM_VS_MCTS_FINDINGS.md` diagnostic already
predicts for that domain. The cube, 8-puzzle, and Sokoban, at the specific difficulties and budgets
tested here, don't show that trap: tree mode reaches the same ceiling merge does, just using more
nodes to get there -- so merging's real, measured efficiency gain (20-57% fewer nodes across all
four domains) shows up as a genuine solve-rate advantage only where tree mode was already failing
without it, and as pure search-cost savings everywhere else. Whether a given domain lands in one
camp or the other is exactly the same question `RANDOM_BASELINE_CHECKLIST.md` already tells you how
to check cheaply, before building any of this.

**Follow-up closed**: `REAL_HEURISTIC_MERGE_FINDINGS.md`'s "Why is Blocksworld the outlier?" section
traces this cube's own budget accounting (`classical_mcts_rubiks.py`'s `new_nodes`/`merge_hits`
stats, ~57% wasted on re-links here) against Blocksworld's and ARC-AGI's identical instrumentation,
and resolves it precisely: a persistent gap needs dense transpositions AND a structurally-stuck tree
mode *together*, not either alone. The cube has the former (even denser than Blocksworld's own
35-41% waste) but not the latter (tree reaches the same ceiling merge does, just slower) -- which is
exactly why the cube lands in the "efficiency-only" camp instead of Blocksworld's "solve-rate
rescue" camp. No further follow-up queued.
