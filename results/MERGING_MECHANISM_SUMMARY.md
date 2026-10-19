# Merging Is Two Mechanisms: Consolidated Result

Single place for the claim this line of work established, what supports it, and what was retracted
along the way. Supersedes the summary sections of the individual findings docs where they disagree.

## The claim

Merging bundles two operations:

1. **Deduplication** — do not expand a state already in the tree.
2. **Pooling** — average the returns of every path reaching a state into one shared node.

Every result in this repo before this work measured the bundle. Separating them:

**Pooling is where the risk lives, and it is the half that can fail catastrophically. Deduplication
carries the benefit. Where pooling works, how you cash the deduplication in does not matter —
filtering and pooling tie. Where pooling is broken, filtering is the repair — up to a point: with a
bad enough equivalence relation the filter's own repair fails, and it fails in the opposite
direction, falling below doing nothing rather than merely losing its advantage.**

**Both catastrophes — pooling's and the filter's floor — require proposals in uniformly random
order.** With a proposal policy of moderate quality, pooling recovers from 0.000 to 0.917–1.000 on
the cube and a maximally coarse relation leaves the filter at 0.975–1.000 (§2.5). LLM search never
proposes at random, so both failure modes are properties of uninformed classical MCTS rather than of
the mechanisms themselves.

Four parts of that were narrowed by this project's own tests, and none of the narrowing came from
outside: an accounting audit (§4), a second model (§2), a controlled approximate-relation sweep and
a proposal-quality sweep (§2.5). The version first written up here was broader than the evidence
supported.

## 1. Classical domains, exact equivalence

Arms: `tree` (path-keyed), `merge` (pool into the existing node), `reject` (discard the duplicate,
keep the tree a tree). **All comparisons below charge rejections to the budget** — a rejected move
costs a real `apply_move`, and not charging it hands the filter ~2× the state generations.

| domain | backup rule | pooling's state | `reject` vs `merge`, charged |
|---|---|---|---|
| Rubik's Cube | UCB1 + MC-average | **catastrophic** (0.000 at every budget) | **+0.98 to +1.00, p=0.0078** |
| Blocksworld | UCB1 + MC-average | works (large win over tree) | +0.007, +0.077 (p=0.58, 0.31) |
| Blocksworld | GUCT-Uniform | works | +0.000, +0.008, −0.042 (all n.s.) |
| Cube / 8-puzzle / Sokoban | GUCT-Uniform | at parity with tree | null at every budget |
| ARC-AGI | UCB1 + MC-average | null | null *(uncharged — the ARC module has no charge option; its reject arm used ~12% more generations than tree, so the null is conservative)* |

The surviving classical headline, every point verified inside budget:

| tier/budget | tree | merge | `reject` (charged) | `apply_move` merge/reject |
|---|---|---|---|---|
| td6/b100 | 0.533 | 0.000 | **1.000** | 100 / 74.8 |
| td8/b200 | 0.400 | 0.000 | **1.000** | 200 / 106.8 |
| td10/b200 | 0.000 | 0.000 | **0.983** | 200 / 164.4 |
| td10/b400 | 0.367 | 0.000 | **1.000** | 400 / 164.9 |

> **Corrected 2026-09-21.** This column previously read "generations merge/reject" with merge at
> 44/84/84/163 and reject at 76/108/166/166. Neither side was an `apply_move` count: merge's was a
> *node* count, and a pooled expansion creates no node, so it undercut merge's real work by
> 2.27–2.45×; reject's was `nodes + rejections`, which overcounts by exactly 1 (the root). All four
> merge rows are now directly measured at the full budget, and reject at 74.8/106.8/164.4/164.9.
>
> **What this does and does not license.** It does *not* mean the filter is the cheaper arm —
> `run_search` has no solve-based exit (a terminal leaf is backed up and `continue`d, so a solved
> run spins on the goal without expanding until the 20× iteration cap), so a filter run that solves
> early simply stops spending while pooling, which never solves, expands to the last unit of budget.
> Split by outcome the ranking inverts: the filter spends more than pooling on the runs it *fails*,
> and under GUCT-Uniform at td6/b40 the charged filter costs 40.6 against merge's 40.0. What it does
> license is the claim this table needs: **the filter never exceeds pooling's budget in the runs it
> wins**, so `charge_rejections` is satisfied and the result is not bought with extra compute.
>
> No solve rate changes. The same mislabelling appears in `ARTICLE_DERIVED_MERGING_FINDINGS.md`,
> `GUCT_UNIFORM_REJECT_CROSSING_FINDINGS.md` and `PAPER_DRAFT.md`; its magnitude on the Blocksworld
> rows is unmeasured. Found while running `APPROXIMATE_RELATION_FINDINGS.md`, which has the
> measurement table.

On the cube under MC-average backup the filter beats *both* tree and merge. That is the one place
deduplication-without-pooling buys something no other arm gets.

**The two fixes are redundant.** Swapping the backup rule (GUCT-Uniform) and declining to pool each
independently repair the same failure; neither adds anything on top of the other. Detail and
retraction: `GUCT_UNIFORM_REJECT_CROSSING_FINDINGS.md`.

## 2. LLM reasoning search, learned approximate equivalence

Step-level PUCT over reasoning states, equivalence by a frozen Ridge projection thresholded at τ.
Four arms, seeded per `(seed, problem, arm)` and paired by problem (each arm draws its own LM samples — the per-arm seed includes the arm's name; corrected 2026-10-02).

**Rejection is genuinely free here**, which is the crucial asymmetry: `expand_node` draws all K
candidates and computes every hidden state *before* the reject decision, so declining one causes no
extra generation and no extra forward pass. The arms are matched in LLM sample batches, the
dominant cost.

| arm | ProsQA (n=21) | Countdown (n=19) | pooled (n=40) | multi-parent fraction |
|---|---|---|---|---|
| `baseline` | 5/21 | 5/19 | 10/40 (25.0%) | — |
| `treatment` (pool) | 18/21 | 11/19 | 29/40 (72.5%) | 0.085 |
| **`reject`** | **19/21** | **16/19** | **35/40 (87.5%)** | **0.000** |
| `reject_all` | 18/21 | 11/19 | 29/40 (72.5%) | 0.000 |

| comparison | ProsQA | Countdown | pooled |
|---|---|---|---|
| `reject` − `baseline` | +0.667, p=0.0001 | +0.579, p=0.0074 | **+0.625, p<0.0001** |
| `treatment` − `baseline` | +0.619, p=0.0002 | +0.316, p=0.18 (n.s.) | +0.475, p=0.0003 |
| `reject` − `treatment` | +0.048 (1 vs 2) | +0.263 (0 vs 5) | +0.150, CI [+0.025, +0.275] |
| `reject_all` − `reject` | −0.048 | −0.263 | −0.150, CI [−0.275, −0.025] |

**A second model withdraws the `reject` − `treatment` line above.** Qwen2.5-1.5B on ProsQA (n=42,
`SECOND_MODEL_FINDINGS.md`): the filter still beats the baseline significantly (+0.214, p=0.0225),
so H3 replicates — but `reject` − `treatment` is −0.024 (5 vs 4 discordant), a tie. Across all
three settings the filter clearly beats pooling in **one** (7B Countdown, +0.263) and ties in the
other two. Pooled across models: +0.061, CI [−0.037, +0.159], p=0.33.

So the honest H3′ statement is symmetric: **pooling adds nothing over filtering, and filtering adds
nothing over pooling.** "Filtering beats pooling" is withdrawn to a single-setting result.

It also killed a prediction of mine: the 1.5B's projection is materially worse (false-merge rate
0.24 vs 0.113), and I expected that to hurt *pooling* most, since a false match corrupts a pooled
estimate while costing the filter one candidate. Pooling did marginally better. Training-set
false-merge rate does not predict which mechanism wins.

Phase 1's gate is met on both 7B datasets and the core effect replicates on a second model. What
does *not* survive is any ordering between the two mechanisms. The one setting where the filter
clearly won (7B Countdown, where pooling did not even clear the baseline significantly) is now a
single data point against two ties, not the general pattern.

**The deduplicating arms leaked budget** (`THIRD_MODEL_BUDGET_LEAK_FINDINGS.md`). A search whose
every reachable leaf is a terminal answer spins on those answers until the iteration cap with budget
unspent; deduplication produces that state, plain search rarely does. A first attempt at
compute-matching truncated each baseline to the expansions its paired arm actually spent — exact as
a replay, but an **upper bound** on the arms, since it cannot see whether they would have used the
leaked budget well. Superseded by the leak-fixed re-runs below:

| setting | pool − baseline, compute-matched | filter − baseline, compute-matched |
|---|---|---|
| 7B ProsQA | +0.714, p=0.0001 | +0.762, p<0.0001 |
| 7B Countdown | **+0.579, p=0.0010** (raw: +0.316, n.s.) | +0.737, p=0.0005 |
| 1.5B ProsQA | +0.357, p=0.0007 | +0.286, p=0.0042 |
| 1.5B Countdown-3 (third model) | +0.050, n.s. | +0.075, n.s. |

Truncation turned out to be a poor substitute for fixing the leak. Re-running both 1.5B settings
with the leak actually fixed (`widen_on_starvation`; `LEAK_FIX_RERUN_FINDINGS.md`):

| 1.5B, leak fixed | pool − baseline | filter − baseline | random rejection − baseline | filter − pool |
|---|---|---|---|---|
| ProsQA (n=42) | **+0.333, p=0.0013** | +0.214, p=0.023 | +0.119, n.s. | −0.119, n.s. |
| Countdown-3 (n=40) | −0.125, n.s. | −0.100, n.s. | **−0.225, p=0.004** | +0.025, n.s. |

- **The leak penalised pooling.** On ProsQA the fix changed exactly four outcomes, all pooling
  gains (baseline and filter identical on all 42 problems). On Countdown-3 it changed nothing.
- **The relation beats random rejection at the filter's own rate:** 11 discordant problems to 2
  across the two settings (p=0.022). The filter is deduplicating, not just pruning — but on
  Countdown-3 it rejects 60% of candidates and still falls below baseline, the cube's filter floor
  appearing in an LLM setting (n.s.).
- **No ordering between pool and filter.** Both 7B settings were then re-run too: every arm
  reproduces its published outcome problem for problem, so the one filter win (7B Countdown,
  +0.263) stands. Across all four settings the filter beats random rejection at its own rate
  **20 discordant problems to 3 (p=0.0005)**, while random rejection alone is worth −0.225 to
  +0.429 over baseline depending on setting — 64–73% of the filter's gain on the two 7B settings.

**Answer-exemption is load-bearing, not a detail.** Filtering verifiable terminals too costs −0.150
pooled and 5 of 19 problems on Countdown, where `reject_all` lands *below* pooling. An answer can be
checked against ground truth for one verifier call on generation already paid for; discarding it
because its projected value sits within τ of another answer throws away a checkable solution on the
least reliable signal in the system.

**τ sweep (ProsQA, 10× range).** The filter is flat at 19/21 for τ ∈ {0.003, 0.01, 0.03} while
pooling slips 18→17, so the gap widens with τ. Read honestly this shows *insensitivity*, not
robustness: match counts barely move (338/340/331) because ProsQA's projected values are crisply
separated. A τ wide enough to force real false matches was never reached.

## 2.5 Does the repair need an exact relation? Yes and no

Every catastrophic-pooling result above used an exact relation (the cube). Every approximate
relation above sat in a domain where pooling worked (§2). Follow-on runs filled that cell.

**On the cube, with a controlled approximate relation (ground truth available):** pooling stays at
0.000 at every relation quality tested. The filter degrades gracefully from a measured false-merge
rate of 0.5 up through ~0.8, still beating tree search — the repair does not require exactness. But
past τ*≈1.3–1.9 (rising with budget) it **crosses below plain tree search**, and by τ=4 the deficit
is significant on every tier tested. So "a wrong rejection costs one discarded branch and does not
accumulate" is false past a threshold: when the rejected branch is the one on the solution path, the
cost is the solution. A follow-up τ×noise grid found τ dominates over noise magnitude, and that
false-merge rate is *not* a sufficient statistic for the damage — two configurations at the same
measured rate (0.81) solved 1.000 and 0.750. Full write-up: `APPROXIMATE_RELATION_FINDINGS.md`.

**On Countdown, sweeping the LLM harness's real projection 60× past its published τ:** the collapse
above did **not** reproduce. The filter's solve rate never fell below baseline, floors at double it.
An earlier version of this paragraph said the relation "could not be made bad enough" there. Wrong:
search-time values clip to [0,1], so τ ≥ 1 is the coarsest relation the bucketing allows, and the
saved τ=2.0 graphs hold exactly one step node per depth in every problem — a single chain — which
still solves 4/10 against the baseline's 2/10. A scale-matched cube run confirmed the difference is
not compute budget (tree solves 0.167 at budget 28 and the collapse still appears), and a
proposal-quality sweep located it (next paragraph). A point estimate at n=10 also suggested
`reject − treatment` flipped sign past τ≈0.36 toward pooling; pooled to n=19 (extending seeds 71/72, matching the
published headline's pooling) the flip **does not survive** — filter and pool tie exactly at τ=0.36,
and the filter significantly beats baseline at τ=0.17 (p=0.0215) where the n=10 slice could not show
significance. Full write-up: `COUNTDOWN_WIDE_TAU_FINDINGS.md`.

**Proposal quality explains both catastrophes** (`PROPOSAL_QUALITY_FINDINGS.md`). At a saturated
relation the filter keeps whichever child is proposed first at each depth: a random move on the
cube, an LLM sample on Countdown. Giving the cube a proposal policy of controllable quality (child
oracle distance + Gaussian noise σ, same policy for every arm) at td8/b200:

| proposals | pool, exact | filter, saturated τ=4 |
|---|---|---|
| random | 0.000 | 0.000 |
| σ = 2 | 0.283 | 0.150 |
| σ = 1 | 0.917 | 0.975 |
| σ ≤ 0.5 | 1.000 | 1.000 |

So the floor and pooling's cube catastrophe both need uninformed proposals. (Tree search in this
module unexpectedly *degrades* with better proposals, 0.400 → 0.000, going shallower and wasting
more budget on duplicates and undo moves; unexplained and specific to this MCTS variant, so
filter − tree is not meaningful in those rows.) One implication cuts the other way: a single chain
with no real deduplication beat baseline 4/10 vs 2/10 on Countdown, so part of the filter's gain
there may be forced depth rather than deduplication. A matched random-rejection arm would separate
them and has not been run.

**Net:** the filter's repair is real under an approximate relation, bounded rather than unconditional,
and the bound only binds when proposals are poor. Countdown never hit it because its proposals
come from a strong LLM, not because its budget was small or its relation too fine.

## 3. Negative results worth keeping

| item | claim tested | outcome |
|---|---|---|
| Reward density | ARC's small merge effect is due to binary exact-match reward | **Not supported.** Graded reward turns merge diffs positive but all inside noise, and makes the *baseline* worse at high budget (0.342 → 0.300). |
| Observational-equivalence keys | A coarser merge key finds more useful equivalences | **Buys merges, not accuracy.** ~2× the merge rate, solve rate flat to slightly worse. |
| DAG in-degree | Multi-parent structure predicts solution paths, so it is a free prior | **Refuted.** 0.84–0.87 discrimination alone; chance-to-inverted (0.405, 0.511) conditioned on visit count. Visit count fully explains it and UCB1 already has it. |
| Self-play target quality (unrun) | Pooling gives lower-variance training targets | **Predicted null by the above** — pooling is the refuted half, and PUCT creates no duplicates to deduplicate. Dropped rather than deferred. |

## 4. Methodology failures, and what each cost

These were more expensive than any single experiment, and all four were invisible in the output.

**Budget accounting that stopped measuring what it was chosen to measure.** "Budget counts
expansions" was picked when every arm spent one state generation per expansion. The filter can
*decline* work, so rejections were free and it ran at ~2× the compute of its comparators. I verified
this once, on one domain, found it fine, and generalised. **Cost: two committed claims retracted.**

**Hash-order nondeterminism.** `legal_moves` returned `list(<set of tuples containing strings>)`, so
every ARC search depended on which process it ran in. Re-running a saved config from its own args
and seeds disagreed on 2 of 12 per-task rates. **Cost: every ARC number in the repo was
process-dependent; two published figures moved on republish** (`ARC_DETERMINISTIC_REPUBLISH.md`),
including a borderline merge penalty at p=0.058 that vanished entirely.

**Statistics that flattered the hypothesis.** Raw AUCs compared across features pointing opposite
ways; quantile strata collapsing on a discrete confounder and leaving it inside the bins; a global
permutation null destroying the within-graph clustering it was introduced to respect. **Cost: item
E's numbers all moved**, though its conclusion held.

**Tests that could not fail.** Three of mine reimplemented the logic they tested, compared a dict to
itself, or used a 9-key stand-in for a 30-key parser. **Cost: they would have passed through the
regressions they were written to catch.**

### Node counts are not compute (audit, 2026-09-24)

An audit counted real state generations (`apply_move` or equivalent) per arm in all 13 classical
modules. **Per charged expansion, tree and merge do identical work everywhere** — one generation per
expansion — so the charging flaw above never touched a tree-vs-merge solve rate. What merge saves is
**nodes**: memory and, where a node carries one, a heuristic/oracle evaluation. It does not save
expansions, generations, rollouts or LLM calls. Merge's node count undercounts its real generations
by 2.0–2.4× on the cube, 1.3–1.7× on Blocksworld, 1.1–2.2× on Sokoban, 1.4–1.5× on the 8-puzzle.

Where the arms' *totals* differ, it is never per-expansion cost:
- **No solve-based exit** in the classical UCB1 modules: whichever arm solves more stops expanding
  and spins to the 20× iteration cap (8-puzzle td12: merge −40%; Sokoban td6: −28%). The GUCT
  modules close a terminal edge after one visit and don't spin. Wall-clock in the classical
  modules is dominated by that spinning and is not a compute measure.
- **Uncharged cycle-block rollouts, merge only**, in reversible domains in rollout mode:
  Blocksworld +7% / +16% rollout generations, Sokoban +6%.

Claims in older documents that read node counts as compute, or "merging reclaims wasted budget",
should be read as node/evaluation savings at identical search cost:
`FINAL_SUMMARY.md` (GUCT "~57% node-count reduction … efficiency", "just less efficiently"),
`RUBIKS_CUBE_FINDINGS.md` ("56–58% fewer nodes … search cost", "pure search-cost savings", the caps
"reclaiming wasted budget"), `REAL_HEURISTIC_MERGE_FINDINGS.md`, `RANDOM_VS_MCTS_FINDINGS.md`,
`SOKOBAN_FINDINGS.md`, `SLIDING_PUZZLE_FINDINGS.md`, `BLOCKSWORLD_FINDINGS.md`,
`CLASSICAL_MCTS_SUMMARY.md`, `MORRIS_FINDINGS.md`, `MORRIS_KSAMPLE_FINDINGS.md`. In the LLM harness,
`H3_MERGE_SEARCH_FINDINGS.md`'s "~2× faster … fewer redundant forward passes" and
`PLAN_GATE_REVIEW.md`'s restatement come from executing **fewer expansions** (a solved search
stops) — every candidate still gets its forward pass before the merge check. The same caveat
applies to the node and wall-clock comparisons in `PROSQA_ADMISSION_FILTER_FINDINGS.md`,
`COUNTDOWN_ADMISSION_FILTER_FINDINGS.md`, `CONNECT_FOUR_FINDINGS.md` and
`DEEPSEARCH_COMPARISON_FINDINGS.md`. Where node reduction was used as a *mechanism* measure
(`CROSS_DOMAIN_MECHANISM_CHECK.md`, `CLASSICAL_MCTS_SUMMARY.md`), it is partly caused by solving
and so partly circular. No solve rate in any of these documents changes.

Two code issues found by the same audit:
- **`guct_uniform_arc.py` terminated early.** `run_search` broke when the root's last untried
  moves all failed to execute, even though the root still had live children (23/120 tree and 16/120
  merge runs stopped after ~15–24 nodes at budget 500); below the root it pruned live subtrees.
  **Fixed and re-run** (`ARC_DETERMINISTIC_REPUBLISH.md` addendum): both arms rise slightly, and
  merge − tree stays null (−0.003, p=0.58 at budget 500; −0.033, p=0.26 at 2000 on 27/30 tasks).
- **`classical_mcts_rubiks.py` let a tree arm pool** when `key_tau` was set without merge or reject.
  Fixed (refused in `__post_init__`); no published row set it.

The common shape: a convention or diagnostic chosen for one situation, still trusted after the
situation changed. On this project the effects being chased are the same size as the noise these
introduce.

## 5. What I would do next

1. ~~**A τ wide enough to force real false matches**, on Countdown~~. **Done**
   (`COUNTDOWN_WIDE_TAU_FINDINGS.md`). Six τ points, published to 60×. The filter's solve rate
   never fell below baseline — unlike the cube's approximate-relation result in §2.5, where it
   does. An apparent `reject − treatment` sign flip past τ≈0.36 at n=10 did not survive extension
   to n=19 (an exact tie). The widest two τ points are byte-identical because values clip to [0,1]:
   τ ≥ 1 is the coarsest relation possible, the filter became a single chain, and it still beat
   baseline — explained by proposal quality (§2.5).
2. ~~**A second model.**~~ **Done** (`SECOND_MODEL_FINDINGS.md`). H3 replicates on Qwen2.5-1.5B
   (+0.214, p=0.0225); "filtering beats pooling" does not. A third model would tell you whether
   Countdown-under-7B was the outlier or the two ties were.
3. ~~**Re-audit the remaining classical merge-vs-tree findings for the accounting flaw.**~~
   **Done** (§4, "Node counts are not compute"). Per charged expansion, tree and merge do identical
   work in all 13 classical modules, so the charging flaw never touched tree-vs-merge solve rates.
   But many older docs read merge's *node* count as compute, which it is not.
