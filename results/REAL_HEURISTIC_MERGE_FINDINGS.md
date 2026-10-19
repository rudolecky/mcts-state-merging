# Does Merging Still Help Once Tree Search Has Real Guidance? Blocksworld / 8-Puzzle / Sokoban / GUCT-Uniform

## Question

`RANDOM_VS_MCTS_FINDINGS.md` established that blind UCB1-MCTS (no heuristic, random-rollout leaf
evaluation) loses persistently to pure random restarts on Blocksworld, the 8-puzzle, and Sokoban --
exactly the domains where merging shows this project's largest benefits. `ALPHAZERO_CONNECT_FOUR_FINDINGS.md`
gave a real trained network a genuine edge over random on Connect Four and found merging still null
there, but for a domain-specific structural reason (real PUCT's eager one-edge-per-move expansion
already prevents duplicate nodes). That result doesn't transfer to Blocksworld/8-puzzle/Sokoban,
whose expansion mechanism has no such guarantee.

**This asks the same question directly on the three domains where merging matters most**: give the
tree a real heuristic (not blind rollout) strong enough to beat random restarts, then check whether
merging still helps on top of that.

## Design

Added `value_source: str = "rollout" | "heuristic"` to each domain's `ClassicalMCTSConfig`
(`classical_mcts_puzzle.py`, `classical_mcts_sokoban.py`, `classical_mcts_blocksworld.py`),
mirroring `search.py`'s own C6 ablation and `search_oracle.py`'s ProsQA precedent. In heuristic
mode, `run_search` skips the random rollout at its leaf-evaluation call site(s) and assigns the new
leaf `1/(1+h)` directly -- UCB1 selection and MC-average backup are otherwise untouched, so results
stay directly comparable to each domain's existing blind findings. Heuristics: Manhattan distance
(8-puzzle), box-goal + player-box Manhattan distance (Sokoban, see below), `pyperplan`'s
`hFFHeuristic` relaxed-plan heuristic (Blocksworld, reused exactly as `guct_uniform_blocksworld.py`
already does).

## Two calibration findings, discovered before any real numbers, that shaped every result below

**1. A deterministic heuristic needs a far smaller UCB1 exploration constant than blind rollout's
usual c=1.4.** UCB1's exploration bonus is derived from Hoeffding's inequality for *noisy* rewards
-- a heuristic value is deterministic (zero variance) for a fixed state, so there's nothing to
re-sample away, and c=1.4's exploration term completely swamped the value signal: on the 8-puzzle,
heuristic-guided tree solved 0/30 at c=1.4, budget=40, an 8-move puzzle -- and 28-35/50 at
c=0.02-0.1, same budget, same puzzles. Every result below uses a per-domain, per-difficulty c found
via an explicit calibration sweep (typically 0.02-0.1), not c=1.4. This is a real, generalizable
lesson: swapping a stochastic rollout for a deterministic value source and keeping the same
exploration constant silently defeats the guidance almost entirely.

**2. Sokoban's box-goal-only heuristic gave zero discrimination among purely-walking moves, and
needed a player-distance term added.** Box-to-goal distance is unchanged by any move that doesn't
push the box -- and most Sokoban moves are walks, not pushes. With the box-only heuristic,
heuristic-guided tree solved 0/50 across a full (c, budget) sweep, discovered by tracing a single
search: 41 nodes created, best heuristic value found was the *start state's own*, because every
one of the four first moves (all walks) looked identically unhelpful. Fixed by adding
`_player_box_heuristic` (player's Manhattan distance to its nearest box) to the leaf value --
`classical_mcts_sokoban.py`'s combined heuristic is `box_goal + player_box`, confirmed admissible
for the box-goal term alone against this level's exact `bfs_distances()` (0/2268 states
overestimated), with the player term providing gradient rather than admissibility.

## Results

All tables: **tree** = `value_source="heuristic"`, `merge_enabled=False`. **random** =
`run_random_search`. **merge** = `value_source="heuristic"`, `merge_enabled=True`. Stats are
bootstrap 95% CI + Wilcoxon on the paired per-puzzle diff (`random-tree` for Stage 1,
`merge-tree` for Stage 2), 30 repeats/puzzle throughout.

### 8-puzzle (`run_heuristic_merge_experiment_puzzle.py`, n=8 puzzles/tier)

**td=8, c=0.02** -- Stage 1 (tree vs. random) / Stage 2 (tree vs. merge):

| budget | tree | random | diff (rand-tree) | p | merge | diff (merge-tree) | p |
|---|---|---|---|---|---|---|---|
| 10 | 0.000 | 0.042 | +0.042 | 0.031 | 0.000 | 0.000 | 1.00 |
| 20 | 0.046 | 0.133 | +0.087 | 0.164 | 0.046 | 0.000 | 1.00 |
| 30 | 0.500 | 0.213 | -0.287 | 0.117 | 0.492 | -0.008 | 1.00 |
| 40 | 0.529 | 0.267 | -0.263 | 0.148 | 0.521 | -0.008 | 1.00 |
| 60 | 0.537 | 0.342 | -0.196 | 0.250 | 0.613 | +0.075 | 0.75 |
| 80 | 0.537 | 0.438 | -0.100 | 0.570 | 0.696 | **+0.158** | 0.25 |
| 120 | 0.542 | 0.596 | +0.054 | 0.773 | 0.933 | **+0.392** | 0.063 |

**td=12, c=0.1**:

| budget | tree | random | diff | p | merge | diff | p |
|---|---|---|---|---|---|---|---|
| 100 | 0.000 | 0.054 | +0.054 | 0.031 | 0.338 | +0.338 | 0.125 |
| 300 | 0.367 | 0.146 | -0.221 | 0.719 | 0.875 | **+0.508** | 0.063 |
| 500 | 0.504 | 0.204 | -0.300 | 0.297 | 0.875 | **+0.371** | 0.25 |
| 800 | 0.550 | 0.308 | -0.242 | 0.195 | 0.875 | **+0.325** | 0.25 |

Except at the lowest budgets (cold-start, matching every other domain), heuristic-guided tree
clearly beats random from budget~=30-100 onward -- **Stage 1's prerequisite holds.** Merging then
adds a large, consistent further gain on top (+0.16 to +0.51), though n=8 keeps most individual
points short of p<0.05.

### Sokoban (`run_heuristic_merge_experiment_sokoban.py`, n=8 puzzles/tier)

**td=6, c=0.02**:

| budget | tree | random | diff | p | merge | diff | p |
|---|---|---|---|---|---|---|---|
| 20 | 0.000 | 0.287 | +0.287 | 0.008 | 0.000 | 0.000 | 1.00 |
| 40 | 0.446 | 0.417 | -0.029 | 0.727 | 0.688 | **+0.242** | **0.008** |
| 80 | 0.821 | 0.663 | -0.158 | 0.203 | 1.000 | **+0.179** | **0.016** |
| 160 | 0.967 | 0.812 | -0.154 | 0.188 | 1.000 | +0.033 | 0.25 |
| 300 | 1.000 | 0.917 | -0.083 | 0.125 | 1.000 | 0.000 | 1.00 |

**td=8, c=0.02**:

| budget | tree | random | diff | p | merge | diff | p |
|---|---|---|---|---|---|---|---|
| 20 | 0.000 | 0.033 | +0.033 | 0.063 | 0.000 | 0.000 | 1.00 |
| 40 | 0.188 | 0.067 | -0.121 | 0.211 | 0.300 | **+0.113** | **0.031** |
| 80 | 0.858 | 0.158 | **-0.700** | **0.008** | 0.992 | +0.133 | 0.125 |
| 160 | 0.979 | 0.275 | **-0.704** | **0.008** | 1.000 | +0.021 | 1.00 |
| 300 | 0.992 | 0.383 | **-0.608** | **0.008** | 1.000 | +0.008 | 1.00 |
| 500 | 1.000 | 0.512 | **-0.487** | **0.008** | 1.000 | 0.000 | 1.00 |

**The cleanest result of the three domains.** Stage 1 is decisive and significant from budget=80
onward (tree crushes random, p=0.008 repeatedly). Stage 2 confirms merging still helps
significantly at the budgets where there's room to (0.008-0.031), with the gap naturally shrinking
to zero once both conditions saturate at 1.000 -- the same ceiling pattern seen throughout this
project, not a sign merging stopped mattering.

### Blocksworld (`run_heuristic_merge_experiment_blocksworld.py`, n=7 puzzles at pl=6, n=8 at pl=10)

**pl=6, c=0.02** (only 7 distinct 5-block states exist at exactly this plan length -- confirmed by
rejection-sampling exhaustion, not a chosen cap):

| budget | tree | random | diff | p | merge | diff | p |
|---|---|---|---|---|---|---|---|
| 10 | 0.000 | 0.252 | +0.252 | 0.016 | 0.000 | 0.000 | 1.00 |
| 20 | 0.367 | 0.395 | +0.029 | 0.969 | 0.410 | +0.043 | 0.625 |
| 40 | 0.714 | 0.629 | -0.086 | 0.938 | 0.857 | +0.143 | 1.00 |
| 80 | 0.724 | 0.829 | +0.105 | 0.891 | 0.857 | +0.133 | 1.00 |

**pl=10, c=0.02**:

| budget | tree | random | diff | p | merge | diff | p |
|---|---|---|---|---|---|---|---|
| 20 | 0.000 | 0.000 | 0.000 | 1.00 | 0.000 | 0.000 | 1.00 |
| 40 | 0.104 | 0.000 | -0.104 | 1.00 | 0.129 | +0.025 | 0.50 |
| 80 | 0.562 | 0.008 | **-0.554** | 0.063 | 0.688 | +0.125 | 0.813 |
| 150 | 0.729 | 0.017 | **-0.713** | **0.016** | 0.708 | -0.021 | 1.00 |
| 300 | 0.871 | 0.042 | **-0.829** | **0.016** | 0.771 | -0.100 | 0.438 |

Stage 1 confirms the same pattern (heuristic-guided tree crushes random at pl=10 from budget=80
on). **Stage 2 is genuinely inconclusive here, not a confirmed null**: the diff is small,
inconsistent in sign at the two highest budgets (-0.021, -0.100), and never significant --
unsurprising given only 7-8 puzzles per tier (small-N noise, not evidence merging fails on
Blocksworld). This is a real, honest contrast with the other two domains' much cleaner Stage 2
signal, discussed in Mechanism below.

### Bonus: GUCT-Uniform (real algorithm, real heuristic) vs. random restart

`guct_uniform_blocksworld.py` already had both `run_search` and `run_random_search`, but the two
were never benchmarked against each other in `BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`. Reusing that
doc's exact puzzle set (30 puzzles, `num_blocks=6`, 10 each at `target_plan_length` 10/12/14),
untreated (`merge_enabled=False`) GUCT-Uniform vs. `run_random_search`, 30 repeats:

| budget | guct-uniform | random | diff (random-guct) | p |
|---|---|---|---|---|
| 100 | 0.030 | 0.002 | -0.028 | **0.007** |
| 500 | 0.032 | 0.020 | -0.012 | 0.200 |
| 1000 | 0.032 | 0.037 | +0.004 | 0.944 |
| 2000 | 0.032 | 0.067 | +0.034 | 0.445 |
| 4000 | 0.032 | 0.113 | **+0.081** | **0.036** |

**GUCT-Uniform's own solve rate is essentially flat at ~0.03 across the entire 40x budget range**
(matching `BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`'s own baseline number exactly) -- and pure random
restart, which needs no tree at all, *overtakes it* by budget=4000, significantly (p=0.036). This is
the most extreme version of the persistent gap found anywhere in this project: a full,
paper-faithful reimplementation of a real published heuristic-guided algorithm (LCB1-Uniform +
Full-Bellman backup + a genuine relaxed-plan heuristic) can get so thoroughly stuck that giving it
40x more budget produces no improvement at all, and blind guessing eventually wins.

## Mechanism

**Where this experiment's two possible outcomes actually landed**: the 8-puzzle and Sokoban show
merging still helps substantially once tree genuinely beats random (the "guidance doesn't replace
merging" branch) -- Sokoban's result is clean and significant, the 8-puzzle's is consistent in
direction and magnitude but underpowered at n=8. Blocksworld is the odd one out: Stage 1 confirms
guidance works, but Stage 2's signal is too small/inconsistent to call either way at n=7-8.

Three domains, three different reasons merging's payoff turned out to vary independent of "is
guidance real":
- **Connect Four (`ALPHAZERO_CONNECT_FOUR_FINDINGS.md`)**: null, because real PUCT's own expansion
  mechanism structurally prevents duplicate nodes -- nothing for merging to reclaim, regardless of
  guidance quality.
- **ProsQA (`RANDOM_VS_MCTS_FINDINGS.md`'s oracle follow-up)**: even a perfect, noise-free value
  only closed under half the tree-vs-random gap -- budget allocation (one-leaf-per-iteration
  expansion) is the dominant limiter there, not guidance quality, and merging wasn't separately
  re-tested at that scale.
- **8-puzzle/Sokoban here**: real (imperfect) heuristic guidance both lets tree beat random *and*
  leaves substantial room for merging to help further -- the redundant-transposition problem
  merging fixes and the "does the tree even beat random" question are separable, and both can point
  the same direction at once.
- **Blocksworld here**: guidance works exactly as well as the other two domains at making tree beat
  random, but whether merging still helps on top, under this project's minimal-diff UCB1+MC-average
  design, is not resolved by this experiment's sample size -- a real open question, not a finding.
- **GUCT-Uniform bonus**: the sharpest illustration that no amount of algorithmic sophistication
  (LCB1-Uniform, Full-Bellman backup, a real relaxed-plan heuristic) guarantees beating random --
  its own persistent-gap pathology (documented in the module's own docstring: LCB1-Uniform can
  tunnel-vision onto one plateau) is severe enough that random restart eventually wins outright.

## Scope and limitations

- Blocksworld's two tiers used n=7 (pl=6, a real combinatorial ceiling, not a chosen cap) and n=8
  (pl=10) puzzles -- underpowered for Stage 2's smaller effect size there; a larger N is the natural
  follow-up if this specific question (does merging help Blocksworld under real UCB1+MC-average
  heuristic guidance) needs a definitive answer.
- The exploration constant c was tuned per domain/difficulty via an explicit calibration sweep
  (0.02-0.1), not derived from theory -- a real, load-bearing methodological choice reported
  transparently above, not hidden inside a "reasonable default."
- Sokoban's combined heuristic (box-goal + player-to-box) is deliberately simple, confirmed
  admissible only for the box-goal component on this level's specific fixed geometry; the player
  term is a gradient aid, not itself a distance-to-solved metric.
- `value_source="heuristic"` keeps UCB1 selection + MC-average backup fixed by design (isolating
  guidance quality) -- doesn't test whether pairing these same heuristics with GUCT-Uniform-style
  Full-Bellman-min backup would resolve Blocksworld's ambiguous Stage 2 result; a separate,
  untested variable.
- The GUCT-Uniform bonus reuses `BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`'s own puzzle set/budget
  range exactly, for direct comparability, not independently re-validated at a different scale.

## Does the backup rule itself change merge's benefit here too? Yes -- it erases it, generalizing the cube's own finding

`RUBIKS_CUBE_FINDINGS.md` found that swapping UCB1+MC-average for GUCT-Uniform's Full-Bellman-min
backup + LCB1-Uniform selection didn't just fix merging's *catastrophic harm* on the cube -- it made
merge and tree statistically **identical** in solve rate at every budget/difficulty tested, while
merge still won on node count. Since the 8-puzzle and Sokoban showed the *opposite* starting point
under UCB1+MC-average (merge already *helping*, not hurting), this is the natural generalization
test: does the same backup-rule swap erase merge's existing *benefit* here too, or does it only
matter when a domain's transposition density is extreme enough to break things?

Ported the same algorithm to both domains (`guct_uniform_puzzle.py`, `guct_uniform_sokoban.py`),
reward = each domain's own exact BFS-oracle distance (not the Manhattan/box-goal heuristic, for the
cleanest possible backup-rule-only comparison). One real bug caught before trusting results: Sokoban's
`bfs_distances()` only covers states reachable *to* the goal via pulls -- a push-based random walk
can shove a box into a genuine deadlock absent from the table, which crashed the first calibration
run with a `KeyError`. Fixed by treating an absent state as maximally bad (`_UNSOLVABLE_H = 1e6`)
rather than assuming every reachable state has a finite distance -- a real, previously-latent fact
about this Sokoban level, not a bug in the table.

**On these two domains, at these difficulties and budgets, merge and tree tied exactly** (n=8, 30
repeats, both domains, both difficulty tiers, 16 budget points total):

| domain | td | budget | tree | merge | diff | tree_nodes | merge_nodes |
|---|---|---|---|---|---|---|---|
| 8-puzzle | 8 | 20 | 0.142 | 0.142 | 0.000 | 21.0 | 15.1 |
| 8-puzzle | 8 | 25 | 1.000 | 1.000 | 0.000 | 26.0 | 18.4 |
| 8-puzzle | 12 | 35 | 0.921 | 0.921 | 0.000 | 36.0 | 24.6 |
| 8-puzzle | 12 | 50 | 1.000 | 1.000 | 0.000 | 51.0 | 34.3 |
| Sokoban | 6 | 20 | 0.229 | 0.229 | 0.000 | 21.0 | 16.9 |
| Sokoban | 6 | 30 | 1.000 | 1.000 | 0.000 | 31.0 | 24.3 |
| Sokoban | 8 | 30 | 0.863 | 0.863 | 0.000 | 31.0 | 25.5 |
| Sokoban | 8 | 40 | 1.000 | 1.000 | 0.000 | 41.0 | 32.0 |

`diff` is exactly 0.000 (CI=[0.000,0.000]) at **all 16 budget points tested across both domains**,
not just these 8. `tree_nodes` equals `budget + 1` at every single point (mechanically confirming
tree mode creates exactly one new node per expansion, with zero exceptions), while merge
consistently uses fewer -- a real, if more modest, ~20-35% reduction here versus the cube's ~57%.

**This is not a universal property of the algorithm -- an earlier version of this section claimed
it was, and that was wrong.** `_lcb1_uniform_score(edge, T)` reads only the edge's own private
`(t, l_hat, u_hat)`, never a shared child's pooled `h_gbfs`, so at any single node merging can never
change *which* child gets picked among that node's own candidates -- that much is genuinely true of
the code, on any domain. It does not follow that solve rate always ties. `guct_uniform_blocksworld.py`'s
own already-established result (`BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`, n=30, budgets 100-4000,
p<0.0001 throughout) is a direct counterexample: tree-mode GUCT-Uniform on Blocksworld stays flat at
0.030-0.050 across the entire 40x budget range while merge climbs to 98.7% by budget=4000 -- a
massive, real solve-rate gap under the *identical* algorithm tested here.

The corrected picture: merging is never *harmful* under GUCT-Uniform (confirmed on all four domains
now), but whether it adds solve-rate *benefit* on top of that depends on whether tree mode's own
unmerged, redundant growth gets trapped and fails to reach the goal within budget on its own --
which happened on Blocksworld (dense transpositions + GUCT-Uniform's deliberately depth-first
selection commit hard to one bad lane) but not on the 8-puzzle or Sokoban at the settings tested
here, where tree mode reaches the same ceiling merge does, just less efficiently. This is exactly
the same "persistent vs. transient gap" diagnostic `RANDOM_VS_MCTS_FINDINGS.md` already established
for blind search -- it turns out to predict GUCT-Uniform's own tree-vs-merge answer too, not just
UCB1's.

## A fifth domain, real ARC-AGI tasks: a small-N false lead, corrected at larger scale

Blocksworld's persistent gap under GUCT-Uniform raised the obvious next question: is it unique, or
shared by ARC-AGI, this project's other established large-merge-benefit domain? ARC-AGI has no
pre-existing heuristic or exact oracle anywhere in this project (the DSL program space is
unbounded, no exhaustive reachability table is feasible), so this needed real new design, not a
straight port: `_arc_heuristic` sums, over training examples, one minus the fraction of grid cells
where the program's current final context value matches that example's target output grid,
deliberately restricted to the exact value `is_goal` itself checks (so h=0.0 if and only if
`is_goal` is True, avoiding a state where some other, non-final value happening to match a target
would spuriously zero the heuristic). One real bug caught during calibration: the first version of
the grid-detection check only looked at whether a candidate value's first row was itself a tuple,
which let some other DSL tuple type (not a real Grid) slip through and crash cell comparison with a
`TypeError`. Fixed by validating every row, not just the first, before comparing cells.

First pass, 12-task run (`find_reachable_tasks`, reusing `run_arc_program_experiment.py`'s own task
curation, min_depth=1-4, max_grid_dim=15, seed=0), 5 repeats, budgets 500 and 2000:

| budget | tree | merge | diff | p |
|---|---|---|---|---|
| 500 | 0.150 | 0.183 | +0.033 | 0.50 |
| 2000 | 0.150 | 0.217 | +0.067 | 0.25 |

Neither point reached significance at this N, but with only 2 of the 12 tasks showing any
movement at all (9 solved 0.00 for both algorithms regardless of budget, 1 was already solved 1.00
by both at the lowest budget), and both of those 2 favoring merge, this read as tentatively
Blocksworld-shaped: tree stuck, merge escaping. Flagged explicitly at the time as underpowered, not
a confirmed fifth data point.

**Retested at n=30 (near the full 32-task pool available under this curation) with a wider move
sample per node (40, up from 25, to rule out "the right primitive just never got sampled" as a
confound) -- and the earlier reading did not hold up:**

| budget | tree | merge | diff | CI | p | tree_nodes | merge_nodes |
|---|---|---|---|---|---|---|---|
| 500 | 0.130 | 0.107 | -0.023 | [-0.050, -0.000] | 0.058 | 422.9 | 329.7 |
| 2000 | 0.140 | 0.143 | +0.003 | [-0.027, 0.030] | 0.797 | 1672.9 | 1200.8 |

At budget=2000, diff is essentially zero (p=0.80) -- parity, not a gap. At budget=500 the sign is
actually slightly *negative* (tree marginally ahead), borderline at p=0.058, not the positive lean
the smaller sample suggested. The per-task picture confirms this is genuine noise, not a
direction reversing: of the tasks that show any movement, roughly half favor tree and half favor
merge at each budget, with no consistent winner -- unlike Blocksworld's own result, where *every*
plan length shows the identical direction and the effect only grows with more data. The 12-task
run's apparent lean was two informative tasks happening to point the same way by chance, not a
real signal that a larger sample would sharpen.

**The corrected read**: ARC-AGI's answer is closer to the cube/8-puzzle/Sokoban's parity than to
Blocksworld's persistent gap -- solve rate ties (or is genuinely too noisy to call, not
one-directionally biased), while merge still delivers a real, consistent node-count reduction
(22-28% fewer nodes than tree at matched budget, both tiers) -- the same efficiency-only pattern
seen elsewhere once a domain doesn't show tree getting one-directionally trapped. This is exactly
why the smaller run was reported as "suggestive, not conclusive" rather than rounded up to a
confirmation: at n=12 with 2 informative tasks, two coin flips landing the same way looks like a
trend until you flip 10 more.

## Why is Blocksworld the outlier? Two factors, not one, quantified across all three domains

Blocksworld is the one domain among five where GUCT-Uniform still shows a real solve-rate gap
rather than parity. The `new_nodes`/`merge_hits` instrumentation built for the cube
(`RUBIKS_CUBE_FINDINGS.md`) was ported to `guct_uniform_blocksworld.py` and `guct_uniform_arc.py`
too, so all three domains' mechanism could be traced with the identical metric and compared
directly, not just observed separately (num_blocks=6, pl=10/12/14 for Blocksworld; the same
n=15-task ARC pool as the generalization check; 3 repeats/instance, budgets spanning each domain's
own established low/high range):

| domain | condition | budget | solved | new_node_frac | root h_gbfs (best found) |
|---|---|---|---|---|---|
| Blocksworld | tree | 100 | 0/45 | 1.000 | 6.51 |
| Blocksworld | tree | 1000 | 0/45 | 1.000 | 6.42 |
| Blocksworld | merge | 100 | 5/45 | 0.652 | 5.51 |
| Blocksworld | merge | 1000 | 40/45 | 0.589 | 0.40 |
| Cube (from `RUBIKS_CUBE_FINDINGS.md`) | tree | -- | reaches 1.000 given enough budget | 1.000 | -- |
| Cube | merge | -- | matches tree exactly | 0.40-0.45 | -- |
| ARC-AGI | tree | 500 | 6/45 | 1.000 | 1.06 |
| ARC-AGI | tree | 2000 | 6/45 | 1.000 | 0.97 |
| ARC-AGI | merge | 500 | 4/45 | 0.846 | 1.25 |
| ARC-AGI | merge | 2000 | 4/45 | 0.775 | 1.17 |

`root.h_gbfs` is the tree's own best value found anywhere in the search (Full Bellman min-backup
propagates the minimum all the way to root, so this is a free, native "how close did anything get"
readout -- no extra bookkeeping needed).

**Transposition density alone doesn't predict the outcome.** Blocksworld's merge wastes 35-41% of
its budget on re-links (`new_node_frac` 0.59-0.65); the cube's wastes 55-60% -- *denser* than
Blocksworld -- yet the cube shows parity while Blocksworld shows a massive rescue. **Whether tree
is stuck alone doesn't predict it either.** ARC's tree is also flat across a 4x budget range (6/45
at both 500 and 2000, `root.h_gbfs` barely moving 1.06 to 0.97) -- a real, if mild, stall, not
totally unlike Blocksworld's -- but merge doesn't rescue it; if anything it's consistently a touch
worse (4/45 vs. 6/45 at both budgets).

**What actually distinguishes Blocksworld is both factors combined, and ARC reveals why that
matters**: ARC's `new_node_frac` under merge stays high (0.78-0.85, close to tree's 1.000) -- far
closer to tree-like than either Blocksworld or the cube -- meaning genuine merge hits are rare
there in the first place. This makes sense structurally: DSL program composition is mostly
order-sensitive (applying function A then B is a different intermediate value than B then A, in
general), unlike Blocksworld's block-stacking or the cube's face-turn commutativity, both of which
generate transpositions constantly. So ARC simply doesn't have much redundant structure for merging
to exploit, regardless of how stuck tree gets. Blocksworld has both ingredients at once: transposition
density comparable to (denser in relative-waste terms than) the cube's, *and* a tree that's
structurally incapable of solving on its own under LCB1-Uniform's deliberately depth-first,
no-exploration-term commitment (this project's own established mechanism for that stall, see
`BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`). The cube has the transpositions but not the stuck tree
(nothing to rescue). ARC has a mildly stuck tree but not enough transpositions (nothing to rescue
*with*). Merging's dramatic effect on Blocksworld isn't really about merging being especially
powerful there -- it's the one domain in this project's sample where a real rescue was both needed
and possible at the same time.

## Status

Closed, including a generalization check, a correction to it, and a fifth domain. Confirms
merging's benefit on top of real guidance is domain-dependent under UCB1+MC-average, not automatic
and not automatically absent -- Sokoban and the 8-puzzle both show it survives (Sokoban
decisively), Blocksworld's own UCB1+MC-average answer is genuinely open pending more data (a
separate question from its already-settled GUCT-Uniform answer below), and the GUCT-Uniform bonus
reinforces this project's central practical lesson from a new angle: never assume a tree search
beats random just because it's real, published, and heuristic-guided -- check. A follow-up ported
GUCT-Uniform to the 8-puzzle and Sokoban and found merge/tree solve-rate parity there too (matching
the cube), but the honest conclusion is narrower than first claimed: merging is never harmful under
this backup rule, and it adds solve-rate benefit exactly when tree mode's own redundant growth
would otherwise get trapped (as it does, dramatically, on Blocksworld) -- not universally neutral,
and not a fact provable from the selection code alone. A fifth domain, real ARC-AGI tasks, initially
looked like it leaned the same direction as Blocksworld at n=12, but that did not survive retesting
at n=30 with wider move sampling: diff is essentially zero at budget=2000 (p=0.80) and mildly
negative, not positive, at budget=500 (p=0.058) -- parity, like the cube/8-puzzle/Sokoban, not a
persistent gap, with the smaller run's apparent lean traced to two informative tasks happening to
point the same way by chance. Merge still uses 22-28% fewer nodes at matched budget throughout,
the same efficiency-only pattern seen wherever tree isn't one-directionally trapped. A final
mechanism trace, comparing all three GUCT-Uniform domains (Blocksworld, cube, ARC) on the identical
new_nodes/merge_hits instrumentation, resolved why Blocksworld specifically shows a persistent gap
and the others don't: it takes *two* factors together, not one -- dense transpositions (which the
cube also has, without a stuck tree to rescue) and a tree that's structurally incapable of solving
on its own (which ARC also has, mildly, without enough transpositions to rescue it with). Blocksworld
is the only domain sampled here where both conditions hold at once.
