# Article-Derived Merging Improvements: One Defect, One Strong Positive, One Clean Negative

## Question

An ARC-AGI/MCTS landscape review was read for transferable ideas about *merging* specifically.
Six of its claims mapped onto something concrete and testable in this repo. This documents what
each one turned into when actually run. Items are lettered as in the review.

The two headline claims of the review itself -- that frontier models now saturate ARC-AGI-2
without external search, and that refinement/evolutionary loops beat tree search under compute
constraints -- are **not** treated as evidence for or against merging. They are claims about
where solve-rate progress on one benchmark comes from; this project studies a search mechanism.
What did transfer were its diagnoses of *why* tree search underdelivers: a weak verifier, an
unbounded action space full of near-duplicate siblings, and a mismatch between the space searched
and the space the task varies over.

---

## 0. Two merge-key inconsistencies on ARC, and the reproducibility bug found looking for them

### 0a. `is_goal` was not a function of the merge key -- real, and unreachable

`arc_program_engine.canonical_key` is order-independent by design (a frozenset over the program
state's entries, so two primitive orderings reaching the same available values are one node), while
`is_goal` read `ctx[-1]`. Two states sharing a merge key could therefore disagree on goal-ness, and
`_make_node` caches `is_terminal` from whichever ordering created the node first.

**Fixed** (`goal_index` scans every index), and then **checked, which showed it could never have
fired.** A state holding the target at a non-final index can only be produced by expanding its own
prefix -- and that prefix holds the target at *its* last index, so even the old test marks the prefix
terminal and gives it no untried moves. It is never expanded, so the exposing state is never
created. Two independent confirmations: the merge arm run under old and fixed `is_goal` on the same
tasks and seeds in one process gave byte-identical results, and an instrumented comparison of both
tests at every node creation across all 32 curated tasks found 0 divergences in ~60k calls.

Kept anyway, because it costs nothing (`last_value_is_goal` handles the rollout hot path in O(1)),
because the invariant should not rest on an expansion-order argument holding forever, and because
it is the right test on its own terms. But it is an invariant repair, not a recovered result.

### 0b. The same inconsistency in GUCT-Uniform's heuristic -- reachable, and still inert

`guct_uniform_arc._arc_heuristic` read `ctx[-1]` too, which `canonical_key` does not determine. Here
the exposing states *are* reachable, and it matters more in principle: GUCT-Uniform is entirely
h-driven, with no rollouts to wash the difference out, so the merge arm scored each node with
whichever ordering arrived first while the tree arm scored each ordering separately.

Fixed by evaluating h at the best context index (via `goal_progress`), which makes it a function of
the merge key and restores `h == 0.0` iff `is_goal` in both directions. Run at the published
settings (30 tasks, seed 0, 10 repeats, moves-per-node 40), against the old heuristic with
everything else identical:

| budget | heuristic | tree | merge | diff | p |
|---|---|---|---|---|---|
| 500 | `ctx[-1]` | 0.113 | 0.113 | -0.000 | 0.786 |
| 500 | best index | 0.120 | 0.120 | -0.000 | 0.874 |
| 2000 | `ctx[-1]` | 0.127 | 0.143 | +0.017 | 0.437 |
| 2000 | best index | 0.143 | 0.160 | +0.017 | 0.357 |

A better heuristic lifts both arms slightly and **leaves the merge-vs-tree difference exactly where
it was** (-0.000 and +0.017 in both). So this asymmetry was also real in the code and also without
measurable consequence for the comparison the project cares about.

### 0c. The actual bug: ARC searches were not reproducible across processes

Chasing an apparent shift in GUCT-ARC's published numbers turned up something that was not mine and
matters more than either inconsistency above. `legal_moves` ended with `return list(moves)`, where
`moves` is a **set of tuples containing strings**. Set iteration follows hash order, Python
randomizes string hashes per process, so the move list order -- and therefore every
`rng.choice(untried_moves)` and every `rng.shuffle(moves)` inside `simulate` -- depended on which
process the search ran in.

Confirmed three ways: the same state and seed under two `PYTHONHASHSEED` values returned two
different orderings; re-running a completed experiment config from its own saved args and seeds
disagreed on 2 of 12 per-task solve rates (one task 0.4 vs 0.0); and two runs of the same ARC config
minutes apart reported tree solve rates of 0.192 and 0.200.

**Every ARC number in this repo predating this fix is process-dependent**, including
`ARC_AGI_FINDINGS.md`'s and `guct_uniform_arc_experiment`'s. That is the real explanation for the
published-vs-rerun drift I had initially, and wrongly, attributed to the `is_goal` change. Fixed
with `sorted(moves, key=repr)` (`key=repr` because arg specs mix `('ctx', int)` with
`('const', str)`, which are not mutually orderable), pinned by a test that compares orderings across
subprocesses under different hash seeds. Two independently launched processes now report identical
tree-arm numbers to every printed digit, which was impossible before.

Every ARC and GUCT-ARC figure below was produced after this fix.

---

## CORRECTION (added later): the Blocksworld comparison below was not compute-matched

A rejected expansion never becomes an expansion, so under this project's "budget counts
expansions" convention it is free — but it costs a real `apply_move`. I verified this stayed
inside budget on the cube (80 + 92 = 172 < 200, below) and then generalised from that single check.
It does not generalise.

Re-audited with rejections charged to the budget (`charge_rejections=True`, added to both configs),
5 blocks, plan lengths 8/10/12, 15 puzzles, 20 repeats:

| budget | tree | merge | reject (charged) | reject − merge | 95% CI | p |
|---|---|---|---|---|---|---|
| 200 | 0.000 | 0.127 | 0.133 | +0.007 | [−0.033, +0.043] | 0.5807 |
| 1000 | 0.127 | 0.503 | 0.580 | +0.077 | [−0.020, +0.203] | 0.3105 |

**So "the filter beats merging on Blocksworld" does not survive.** Charged, it is a tie. The
Blocksworld section below should be read as: the filter *matches* merging there, at ~1.7× the state
generations if you do not charge them.

**The cube result below does survive**, and is now the load-bearing classical claim. Re-verified
with charging, oracle-guided, c=0.02, 8 puzzles, 15 repeats:

| tier/budget | tree | merge | reject (charged) | generations merge/reject |
|---|---|---|---|---|
| td6/b100 | 0.533 | 0.000 | **1.000** | 44 / 76 |
| td8/b200 | 0.400 | 0.000 | **1.000** | 84 / 108 |
| td10/b200 | 0.000 | 0.000 | **0.983** | 84 / 166 |
| td10/b400 | 0.367 | 0.000 | **1.000** | 163 / 166 |

The corrected overall claim is narrower than what this document originally argued: **the filter is
a repair for pooling's catastrophic failure mode, not a general improvement on merging.** Where
pooling works, the filter ties it. Where pooling is broken, the filter fixes it — and so does
swapping the backup rule, which makes the two fixes redundant rather than complementary. Full
accounting in `GUCT_UNIFORM_REJECT_CROSSING_FINDINGS.md`.

The LLM-harness results are unaffected: there a rejection causes no extra generation, because
`expand_node` draws all K candidates and computes their hidden states before the reject decision.

---

## C. Reject-duplicates: the strongest result here

The review's "unbounded action space, siblings are near-duplicates" point exposes something this
project had not separated. Ordinary merging does **two** things at once:

1. makes expansions land on distinct states, and
2. pools statistics from every path reaching one.

Every result in this repo -- the Blocksworld win, the cube catastrophe, both failed capping
mitigations, the GUCT-Uniform backup-rule fix -- is about the combination. A third arm was added
that keeps (1) and drops (2): compute the merge key, but **discard** an expansion landing on an
already-seen state and try another untried move instead, keeping path-based node keys so no
statistic is ever aggregated and the backup rule is untouched.

Budget accounting: a rejected move never becomes an expansion, so it is not charged to the
expansion budget (consistent with this project's convention). It does cost a real `apply_move`, so
this arm does more work per unit of budget -- which is why the cube numbers below are also checked
against the stricter accounting where rejections *are* charged.

### Rubik's Cube -- the domain where merging is catastrophic

Oracle-guided (exact BFS distance), c=0.02, 8 puzzles/tier, 15 repeats. `tree` and `merge` are the
published conditions; `reject` is new.

| tier | budget | tree | merge | reject | reject vs tree | p |
|---|---|---|---|---|---|---|
| td=6 | 50 | 0.000 | 0.000 | **1.000** | +1.000 [1.000, 1.000] | 0.0078 |
| td=6 | 100 | 0.533 | 0.000 | **1.000** | +0.467 [0.267, 0.692] | 0.0156 |
| td=6 | 200 | 1.000 | 0.000 | 1.000 | +0.000 | 1.0000 |
| td=6 | 300 | 1.000 | 0.000 | 1.000 | +0.000 | 1.0000 |
| td=8 | 100 | 0.000 | 0.000 | **1.000** | +1.000 [1.000, 1.000] | 0.0078 |
| td=8 | 200 | 0.400 | 0.000 | **1.000** | +0.600 [0.458, 0.767] | 0.0078 |
| td=8 | 400 | 1.000 | 0.000 | 1.000 | +0.000 | 1.0000 |
| td=8 | 600 | 1.000 | 0.000 | 1.000 | +0.000 | 1.0000 |
| td=10 | 200 | 0.000 | 0.000 | **1.000** | +1.000 [1.000, 1.000] | 0.0078 |
| td=10 | 400 | 0.367 | 0.000 | **1.000** | +0.633 [0.308, 0.900] | 0.0312 |
| td=10 | 800 | 1.000 | 0.000 | 1.000 | +0.000 | 1.0000 |
| td=10 | 1200 | 1.000 | 0.000 | 1.000 | +0.000 | 1.0000 |

The reject arm solves **100% at every budget and difficulty**, including budgets where plain tree
search solves nothing (td=10 at budget 200: 0.000 -> 1.000) and where merging is at its documented
zero. It does not merely avoid the catastrophe; it strictly dominates the tree baseline at low
budget and matches it once the tree saturates.

**Independently verified, not read off the solve-rate counter.** For three td=10 instances
(confirmed exactly 10 moves from solved against the BFS table), at budget 200:

| instance | arm | solved | nodes | rejected | terminal at BFS depth |
|---|---|---|---|---|---|
| cube_d10_0 | tree | no | 201 | – | – |
| cube_d10_0 | merge | no | 83 | – | – |
| cube_d10_0 | reject | **yes** | 80 | 92 | **10** (optimal) |
| cube_d10_1 | reject | **yes** | 73 | 77 | **10** |
| cube_d10_2 | reject | **yes** | 75 | 76 | **10** |

Every terminal node holds the genuinely `rc.SOLVED` cube, reached at exactly the true optimal
distance. And the stricter accounting holds: 80+92 = 172, 73+77 = 150, 75+76 = 151 -- all under the
200 budget, so the win survives charging rejections as expansions.

**Mechanism.** With an exact oracle and near-greedy c=0.02, the search wants to descend the
distance gradient. Plain tree search re-derives the same cube states through different paths and
burns budget on them (201 nodes, no solve). Merging deduplicates them but pools their statistics
into hubs that then absorb the budget (83 nodes from 200 expansions -- the documented hub
pathology). The filter deduplicates without pooling, and greedy descent on an exact distance with
no revisits is just a shortest-path walk. That is why the depth is exactly optimal.

This is the cleanest available confirmation of what `RUBIKS_CUBE_FINDINGS.md` inferred: the cube
failure is about **pooling**, not about deduplication. Both capping mitigations tried to weaken
pooling by refusing merges and failed as a cliff; removing pooling outright while keeping
deduplication works completely.

### Blocksworld -- the domain where merging is the largest win

45 puzzles (5 blocks, plan lengths 8/10/12), 20 repeats, blind UCB1. The `merge` column is
`BLOCKSWORLD_FINDINGS.md`'s published run (50 repeats); the tree baseline reproduces it closely
enough (0.000/0.117/0.209/0.252 here vs. 0.001/0.115/0.211/0.255 published) that the columns are
comparable.

| budget | tree | merge (published) | reject | reject vs tree | p |
|---|---|---|---|---|---|
| 200 | 0.000 | 0.158 | **0.370** | +0.370 [0.244, 0.498] | <0.0001 |
| 1000 | 0.117 | 0.602 | **1.000** | +0.883 [0.807, 0.950] | <0.0001 |
| 2000 | 0.209 | 0.980 | **1.000** | +0.791 [0.673, 0.897] | <0.0001 |
| 3000 | 0.252 | 1.000 | **1.000** | +0.748 [0.628, 0.861] | <0.0001 |

**The admission filter beats merging on merging's best domain**, by a wide margin exactly where
the budget is tight: 0.370 vs 0.158 at budget 200, 1.000 vs 0.602 at budget 1000. So the
Blocksworld win is not something pooling provides and rejection merely approximates -- rejection
delivers more of it, sooner.

Node counts tell the mechanism apart: merge reaches these rates with 43–85% fewer nodes than
budget (pooling), while reject stays near budget at low budget (201 -> 180) and plateaus at ~395
from budget 1000 on. That plateau is post-solve iteration spinning, not exhaustion -- checked
directly: on three plan-length-10 instances at budget 1000 the arm created 204–815 nodes against an
866-state reachable space with the root never exhausted, so it is guided search, not enumeration.

The honest caveat: 5-block Blocksworld has only 866 reachable states, so a budget of 1000+ is on
the order of the whole space, and a non-revisiting search there is approaching exhaustive coverage.
The Rubik's result is the one where that explanation is unavailable -- 3.67M states, solved from
~75 nodes.

### ARC-AGI -- null, like everything else on this domain

12 tasks, 10 repeats, seed 0, same task set as `ARC_AGI_FINDINGS.md`, deterministic move ordering,
and `stop_on_solve=True` so the node and wall-clock columns are comparable across arms (it cannot
affect a solve rate -- verified over 120 runs, identical solved status with and without).

**binary reward, typed key:**

| budget | tree | merge | reject | merge diff | reject diff |
|---|---|---|---|---|---|
| 500 | 0.192 | 0.208 | 0.200 | +0.017 (p=0.50) | +0.008 (p=1.00) |
| 2000 | 0.267 | 0.267 | 0.242 | +0.000 (p=1.00) | -0.025 (p=0.50) |
| 5000 | 0.342 | 0.358 | 0.325 | +0.017 (p=0.75) | -0.017 (p=1.00) |

**graded reward, typed key** (item A):

| budget | tree | merge | reject | merge diff | reject diff |
|---|---|---|---|---|---|
| 500 | 0.208 | 0.208 | 0.208 | +0.000 (p=1.00) | +0.000 (p=1.00) |
| 2000 | 0.225 | 0.275 | 0.275 | +0.050 (p=0.25) | +0.050 (p=0.13) |
| 5000 | 0.300 | 0.342 | 0.350 | +0.042 (p=0.25) | +0.050 (p=0.25) |

**graded reward, values key** (items A+B together):

| budget | tree | merge | merge diff | merge_hits |
|---|---|---|---|---|
| 500 | 0.208 | 0.217 | +0.008 (p=1.00) | 77.0 |
| 2000 | 0.225 | 0.250 | +0.025 (p=0.75) | 301.1 |
| 5000 | 0.300 | 0.325 | +0.025 (p=0.63) | 782.4 |

Nothing reaches significance anywhere, in any configuration, for either treatment arm. What can be
said about the two hypotheses:

* **Item A (reward density) is not supported, though it is no longer contradicted either.** The
  article's verifier argument predicts that ARC's binary exact-match reward is why merging shows
  this project's smallest effect, since pooling zeros carries no information. Under graded reward the
  merge diffs do turn uniformly positive (+0.000/+0.050/+0.042 vs binary's +0.017/+0.000/+0.017),
  but every one of them is inside the noise, and the *baseline* is worse at the budgets that matter
  (tree at 5000: 0.342 binary vs 0.300 graded). Cell-level grid overlap looks like a misleading
  gradient in program space -- a program can be one cell from the target and structurally nowhere
  near a solution. Confound worth stating: `c=1.4` was tuned for 0/1 values and graded rewards
  compress the Q range, so some of this is exploration-constant mismatch rather than the signal.
* **Item B (observational-equivalence key) buys merges, not accuracy.** The coarser key roughly
  doubles the merge rate at matched budget (782 vs 449 merge hits at budget 5000) and cuts nodes
  further (2930 vs 3182), with solve rate statistically unchanged. The extra equivalences are real;
  they are not on paths that mattered.

**The fourth cell (binary reward, values key) could not be completed.** The run ballooned to 14.3 GB
RSS and was killed. This is the DSL value-scaling failure already documented in
`ARC_AGI_FINDINGS.md` (`upscale`/`objects`/`mapply` over cell-level objects), which the
`max_grid_dim<=15` task filter bounds at the *inputs* but not for values a program builds
mid-search. Per-task timing at a fixed seed showed no pathology (all 12 tasks under 0.5s at budget
500), so it is a seed-dependent tail, not a property of that configuration. Reported as an
incomplete cell, folded into no conclusion.

---

## G. Wall-clock: on ARC, the node saving is not a compute saving

Every arm reports seconds per repeat next to its node count, with `stop_on_solve` making the two
comparable. At budget 5000:

| config | arm | nodes | sec/repeat |
|---|---|---|---|
| binary/typed | tree | 3599 | 3.53 |
| binary/typed | merge | 3156 (-12%) | 3.12 (-12%) |
| binary/typed | reject | 3731 (+4%) | 3.19 (-10%) |
| graded/typed | tree | 3785 | 3.34 |
| graded/typed | merge | 3182 (-16%) | 3.31 (-1%) |
| graded/typed | reject | 3587 (-5%) | 4.56 (+37%) |

Merging's 12–16% node reduction converts to somewhere between 12% and 1% of wall-clock depending on
configuration, because the merge key is a frozenset over per-example grid values and computing it
costs about what the saved expansion did. The reject arm's cost is the clearer signal: it pays a
full `apply_move` for every state it computes and throws away, up to +37%.

The article's cost-accounting point lands: on this domain the node-count column and the compute
column are nearly unrelated, and only the second is a budget anyone pays. On the cube it goes the
other way -- reject solves within 200 budget where tree fails at 201 nodes, and even charging
rejections as expansions (80+92 = 172) it stays inside the budget.

---

## E. DAG in-degree as a free prior: real signal, nothing usable beyond visit count

Merging turns the tree into a DAG, and the only structure a DAG has that a tree does not is that
some nodes have several parents. The review's ARC-AGI-3 section argues for intrinsic/structural
exploration signals where extrinsic reward is absent, so the question is whether in-degree predicts
solution-path membership. `latent-mcts-merging-plan.md` §13 asks the same thing.

Merge-mode UCB1 on Blocksworld (45 puzzles, plan lengths 8/10/12) and Sokoban (45 puzzles, target
distances 4/6/8), 10 searches each at budget 800, `stop_on_solve=True`, 500 within-graph label
permutations, 40 strata. "On a solution path" = reachable from the root **and** able to reach a
terminal, over the real edge set (so merge-induced cycles are handled by reachability, not assumed
away). Root and terminal nodes are excluded, both on the on-path side, because their feature values
are fixed by construction.

Depth points the opposite way from the other two features (on-path nodes are *shallower*), so the
comparable figure is discrimination = max(AUC, 1-AUC), not the raw AUC:

| domain | feature | mean on-path | mean off-path | AUC | discrimination | perm p |
|---|---|---|---|---|---|---|
| blocksworld | in-degree | 2.19 | 1.06 | 0.837 | 0.837 | 0.0020 |
| blocksworld | depth | 5.40 | 6.89 | 0.296 | 0.704 | 0.0020 |
| blocksworld | visits | 30.86 | 1.36 | 0.984 | 0.984 | 0.0020 |
| sokoban | in-degree | 3.03 | 1.47 | 0.873 | 0.873 | 0.0020 |
| sokoban | depth | 4.27 | 5.88 | 0.272 | 0.728 | 0.0020 |
| sokoban | visits | 94.67 | 4.86 | 0.929 | 0.929 | 0.0020 |

In-degree looks strong unconditionally (0.84–0.87). **Conditioned on visit count it stops being
usable:**

| domain | in-degree \| visits | in-degree \| depth |
|---|---|---|
| blocksworld | AUC **0.405** (discrim 0.595, 5 strata, 8.8M pairs) | AUC 0.798 (12 strata, 57M pairs) |
| sokoban | AUC **0.511** (discrim 0.511, 38 strata, 5.7M pairs) | AUC 0.840 (11 strata, 48M pairs) |

Sokoban is flat chance. Blocksworld is not quite chance but is *inverted*: among nodes the search
visited equally often, on-path nodes carry slightly **lower** in-degree. Either way there is no
positive prior here -- the unconditional signal is a re-reading of where the search already spent
budget, and UCB1 already has that number. In-degree does survive conditioning on depth (0.80/0.84),
so it is not merely "distance from the root", but that was never the useful comparison.

**This kills the item as proposed.**

Four methodological points, each of which changed a number:

* The searches must run with `stop_on_solve=True` (added to the Blocksworld and Sokoban
  `run_search`, default off, cannot affect any published solve rate since budget is spent only on
  expansions). Without it, once a terminal node exists the loop keeps re-selecting it and backing
  up 1.0 along the solution path without consuming expansion budget, inflating on-path visits from
  ~16 to ~589. That said, it removes an artifact rather than the confound: with it, visits still
  discriminates at 0.93–0.98, because on-path nodes genuinely sit on the UCB1-preferred descent.
* Comparing raw AUCs across features is wrong when one points the other way. Read raw, depth's 0.27
  looks like a dead control; it is discriminating at 0.73, close to in-degree's margin over it.
* The conditional statistic needs strata that are actually homogeneous. Visit counts put most of
  their mass on a few small integers and then trail into a long tail of unique large values, so a
  quantile-binning fallback collapses onto the dense values and leaves confounder spread inside
  each bin -- measured 2 usable strata of 10 requested, 10% of pairs. Binning by the most *frequent*
  values instead, and excluding the pooled remainder bin from scoring, fixes it independently of
  tail shape; a `degenerate` flag now fires below 3 strata or 1000 pairs.
* The permutation null has to shuffle labels *within* each search graph. A global shuffle destroys
  the very clustering it was introduced to respect, which reinstates the independence assumption
  and pins the p-value at the 1/(n_perm+1) floor.

## D. Self-play target quality (switch only, no run)

Merging was found structurally null for PUCT **solve rate**
(`ALPHAZERO_CONNECT_FOUR_FINDINGS.md`): PUCT never creates duplicate nodes, so there is nothing to
deduplicate. But the review's strongest empirical claim is that tree search pays off as an outer
loop generating training data, not as an inference method -- and self-play's policy target *is* the
root visit distribution, which under merging pools every path that reached a state. Whether that
lower-variance target trains a stronger network at identical self-play budget is a different
question from whether it solves more puzzles at search time, and it has never been asked here.

`run_selfplay_train.py` now takes `--selfplay-merge/--no-selfplay-merge` (default on, matching
every run so far), so the two arms are runnable at matched self-play budget. Running them to
convergence is a real training spend and was left unstarted.

## F. Bandit tau

Explicitly gated behind C in the review's own ordering. Not implemented.

---

## What this adds up to

The one idea that paid was the one that came from taking the article's *diagnosis* seriously rather
than its recommendations: separating merging into deduplication and statistic-pooling, and keeping
only the first. That arm

* fixes the Rubik's Cube catastrophe completely (0.000 -> 1.000 at every budget and difficulty),
  and beats plain tree search at low budget rather than merely matching it,
* beats merging on Blocksworld, merging's strongest domain, at every budget (0.370 vs 0.158 at
  budget 200; 1.000 vs 0.602 at 1000),
* is null on ARC, like every other manipulation of this domain,
* and costs wall-clock (+30% on ARC at budget 5000) for a node count barely below the tree's.

That sharpens this project's existing thesis. `RUBIKS_CUBE_FINDINGS.md` concluded that whether
merging helps is a property of the backup rule; the GUCT-Uniform port then showed a different
backup rule removes the harm. This says the harm was never in deduplication at all -- it is
entirely in what the backup rule does with pooled statistics, and simply not pooling is a third
option that neither capping mitigation approximated.

Two negatives worth keeping:

* Reward density does not explain ARC's small merge effect (item A). Graded reward turns the merge
  diffs uniformly positive but all within noise, and it makes the baseline worse at high budget.
* DAG in-degree is not a usable prior: 0.84-0.87 discrimination unconditionally, chance-to-inverted
  once conditioned on visit count (item E).

Two merge-key inconsistencies fixed with no measurable payoff, both real in the code and both inert
in effect: `is_goal` on ARC (provably unreachable), and GUCT-Uniform's ARC heuristic (reachable, but
it moves both arms equally and leaves the merge-vs-tree difference identical).

And one bug that was not mine and matters more than either: **ARC searches were not reproducible
across processes.** `legal_moves` returned `list(<set of tuples containing strings>)`, so move order
followed randomized string hashes. Re-running a saved config from its own args and seeds disagreed
on 2 of 12 per-task solve rates. Every ARC figure in this repo predating the fix -- including
`ARC_AGI_FINDINGS.md`'s -- is process-dependent, and it, not the goal-test change, explains the
published-vs-rerun drift I first misattributed.

### Follow-ups this opens

* The reject arm has never been run under GUCT-Uniform's Full-Bellman-minimum backup. That is the
  natural next comparison: the backup-rule swap and the no-pooling arm are two different fixes for
  what now looks like the same underlying problem, and they have not been crossed.
* Its cost profile is wrong for a tight-budget setting (a rejected move pays a full `apply_move`,
  up to +37% wall-clock on ARC). Cheap-to-test states first, or a Bloom-filter pre-check, would
  attack that directly.
* Item D's switch is in place and unrun.
* Blocksworld's reject numbers were collected before `stop_on_solve` was threaded into that runner,
  so its node column carries the same post-solve-spinning confound described under item G. Solve
  rates are unaffected (a solved run is solved either way); the node column should be re-collected
  before it is quoted as an efficiency figure.
* `ARC_AGI_FINDINGS.md` and `guct_uniform_arc_experiment` predate the reproducibility fix and should
  be re-run before their numbers are cited again.
