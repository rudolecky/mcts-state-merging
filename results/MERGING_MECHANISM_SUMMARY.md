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
filtering and pooling tie. Where pooling is broken, filtering is the repair.**

Both halves of that were narrowed by this project's own tests: the first by an accounting audit
(§4), the second by a second model (§2).

That is narrower than the version first written up here, and the narrowing came from an accounting
error I made and then caught (§4).

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
| ARC-AGI | UCB1 + MC-average | null | null |

The surviving classical headline, every point verified inside budget:

| tier/budget | tree | merge | `reject` (charged) | generations merge/reject |
|---|---|---|---|---|
| td6/b100 | 0.533 | 0.000 | **1.000** | 44 / 76 |
| td8/b200 | 0.400 | 0.000 | **1.000** | 84 / 108 |
| td10/b200 | 0.000 | 0.000 | **0.983** | 84 / 166 |
| td10/b400 | 0.367 | 0.000 | **1.000** | 163 / 166 |

On the cube under MC-average backup the filter beats *both* tree and merge. That is the one place
deduplication-without-pooling buys something no other arm gets.

**The two fixes are redundant.** Swapping the backup rule (GUCT-Uniform) and declining to pool each
independently repair the same failure; neither adds anything on top of the other. Detail and
retraction: `GUCT_UNIFORM_REJECT_CROSSING_FINDINGS.md`.

## 2. LLM reasoning search, learned approximate equivalence

Step-level PUCT over reasoning states, equivalence by a frozen Ridge projection thresholded at τ.
Four arms, paired on identical LM draws via per-`(seed, problem, arm)` seeding.

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

Phase 1's gate is met on both datasets. Pooling fails to add anything, and on Countdown does not
even clear the baseline significantly while the filter does. Caveat: the `reject` − `treatment`
p-values (0.063–0.070) are the floor McNemar's exact test can return for those one-sided
discordances, so n is the constraint.

**Answer-exemption is load-bearing, not a detail.** Filtering verifiable terminals too costs −0.150
pooled and 5 of 19 problems on Countdown, where `reject_all` lands *below* pooling. An answer can be
checked against ground truth for one verifier call on generation already paid for; discarding it
because its projected value sits within τ of another answer throws away a checkable solution on the
least reliable signal in the system.

**τ sweep (ProsQA, 10× range).** The filter is flat at 19/21 for τ ∈ {0.003, 0.01, 0.03} while
pooling slips 18→17, so the gap widens with τ. Read honestly this shows *insensitivity*, not
robustness: match counts barely move (338/340/331) because ProsQA's projected values are crisply
separated. A τ wide enough to force real false matches was never reached.

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

The common shape: a convention or diagnostic chosen for one situation, still trusted after the
situation changed. On this project the effects being chased are the same size as the noise these
introduce.

## 5. What I would do next

1. **A τ wide enough to force real false matches**, on Countdown — where `reject_all` losing 5 of 19
   problems is direct evidence false matches bite, unlike ProsQA where the sweep only showed
   insensitivity.
2. ~~**A second model.**~~ **Done** (`SECOND_MODEL_FINDINGS.md`). H3 replicates on Qwen2.5-1.5B
   (+0.214, p=0.0225); "filtering beats pooling" does not. A third model would tell you whether
   Countdown-under-7B was the outlier or the two ties were.
3. **Re-audit the remaining classical merge-vs-tree findings for the accounting flaw** if any of
   them are ever quoted comparatively. The flaw only affects arms that decline work, so the
   pre-existing tree/merge results are unaffected — but that should be checked, not assumed, given
   §4.
