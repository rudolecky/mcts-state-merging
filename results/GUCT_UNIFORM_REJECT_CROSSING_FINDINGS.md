# RETRACTED AND CORRECTED: The Two Fixes Are Redundant After All

> **This document originally claimed the opposite.** An earlier version reported that the
> admission filter beats statistic pooling under GUCT-Uniform's backup rule (+0.090 at p=0.0005 on
> Blocksworld) and concluded the two fixes for merging's harm were not redundant. That was wrong.
> The comparison was not compute-matched: a rejected expansion still costs a real `apply_move`, and
> the filter was performing ~2.3× the state generations of the arm it was being compared against.
> Charging rejections to the budget erases the entire effect. The correction, and how it was
> caught, is below.

## The claim under test

Two independent fixes for the damage merging can do, never previously run together:

1. **Swap the backup rule.** GUCT-Uniform (Full-Bellman-minimum, LCB1-Uniform) brought merge and
   tree to parity on the cube, 8-puzzle and Sokoban, where UCB1 + Monte-Carlo-average backup made
   merging catastrophic.
2. **Stop pooling.** Deduplicate expansions without sharing statistics.

If those repair the same failure, applying the second on top of the first should do nothing.

## What the accounting flaw was

`budget` counts expansions. A rejected move never becomes an expansion, so under the default
accounting it is free — but it costs a real `apply_move`: the state is generated, hashed, compared
and discarded. Measured on the cube at td=6, budget 40:

| arm | nodes | rejections | real `apply_move` | vs budget 40 |
|---|---|---|---|---|
| merge | 19.6 | 0 | 40.0 | at budget |
| reject (uncharged) | 41.0 | 53.3 | **93.2** | **2.3× over** |

> **Corrected 2026-09-21.** The "state generations" column originally read 19.6 for merge and 94.3
> for reject. Neither was an `apply_move` count: 19.6 is merge's *node* count (a pooled expansion
> creates no node) and 94.3 is reject's `nodes + rejections`. Both re-measured by counting
> `apply_move` directly. The ratio is 2.3×, which is what this document's summary already said, so
> **the retraction itself is unaffected** — it was fixed by `charge_rejections`, which is the right
> mechanism regardless of how the diagnostic was labelled.

I had checked this once, on the classical UCB1 cube, where the arm did stay inside budget
(80 + 92 = 172 < 200) — and then generalised from that single check instead of re-verifying per
domain. It does not generalise.

`charge_rejections` now exists on all six affected configs (four GUCT modules, classical
Blocksworld and cube). It is the accounting any comparative claim should be read off.

## Corrected results, with rejections charged

**Blocksworld under GUCT-Uniform** (6 blocks, plan lengths 10/12/14, 12 puzzles, 10 repeats):

| budget | merge | reject (charged) | diff | 95% CI | p | nodes merge / reject |
|---|---|---|---|---|---|---|
| 100 | 0.217 | 0.217 | +0.000 | [−0.033, +0.033] | 1.0000 | 67 / 101 |
| 300 | 0.550 | 0.558 | +0.008 | [−0.133, +0.150] | 0.8994 | 188 / 301 |
| 1000 | 0.883 | 0.842 | −0.042 | [−0.125, +0.050] | 0.6758 | 589 / 1001 |

> **Corrected 2026-09-24.** This column was labelled "generations merge/reject". Merge's figure is
> its node count (equal to its hFF evaluation count); reject's is `nodes + rejections`. An audit that
> counted `apply_move` directly in GUCT Blocksworld found merge spends exactly its budget and charged
> reject spends budget +≤1, so the two arms are **compute-matched** here, not reject-heavy. Measured
> at 5-block configurations, not these exact 6-block ones; the mechanism (one generation per
> expansion, terminal edges closed after one visit, so no post-solve spinning) is the same.

The +0.090 / +0.087 that this document previously reported are gone.

**The cube, 8-puzzle and Sokoban under GUCT-Uniform.** These were run as a *falsification test* of
the original (wrong) claim, which predicted a null here since GUCT-Uniform had already brought
merge and tree to parity. Uncharged, the filter appeared to win enormously — +1.000 at six of
twelve cube budget points, +0.900 on Sokoban — which looked like a refutation. Charged:

> **Accounting note (2026-09-21).** In the "generations merge/reject" column below, the merge side
> is a *node* count and the reject side is `nodes + rejections`; neither is an `apply_move` count.
> A pooled expansion creates no node, so merge's compute is understated — by 2.27–2.45× on the cube
> rows, magnitude unmeasured on the Blocksworld rows, where the reject column being exactly
> budget+1 suggests the two arms are compute-matched rather than the filter being costlier. See
> `MERGING_MECHANISM_SUMMARY.md`.

| tier/budget | tree | merge | reject (charged) | diff | p | generations merge/reject |
|---|---|---|---|---|---|---|
| td6/b40 | 0.000 | 0.000 | 0.000 | +0.000 | 1.0000 | 20 / 42 |
| td6/b60 | 0.000 | 0.000 | 0.000 | +0.000 | 1.0000 | 27 / 61 |
| td8/b50 | 0.000 | 0.000 | 0.000 | +0.000 | 1.0000 | 23 / 51 |
| td8/b80 | 0.000 | 0.000 | 0.000 | +0.000 | 1.0000 | 36 / 83 |
| td10/b90 | 0.000 | 0.000 | 0.000 | +0.000 | 1.0000 | 40 / 92 |
| td10/b110 | 0.325 | 0.325 | 0.367 | +0.042 | 0.1250 | 48 / 111 |

Null, as the prediction said. **The prediction was right and the experiment that appeared to refute
it was the thing at fault.**

> **Retracted 2026-09-21.** This paragraph previously added that "merge does not even spend its
> budget here (20 generations against 40) ... it is the more compute-efficient arm". False. 20 is
> merge's node count; counting `apply_move` directly, merge spends **exactly its budget** in all six
> rows of the table above (40.0 / 60.0 / 50.0 / 80.0 / 90.0 / 110.0), and the charged filter spends
> marginally *more* (40.6 / 60.0 / 50.4 / 81.8 / 90.9 / 110.3). The arms are compute-matched here
> and the filter is, if anything, the slightly more expensive one. The null result is unchanged.

## So the fixes are redundant

| setting | pooling's state | filter vs pooling, charged |
|---|---|---|
| UCB1 + MC-average, cube | **catastrophic** (0.000 everywhere) | **filter wins, +0.98 to +1.00, p=0.0078** |
| UCB1 + MC-average, Blocksworld | works (large win over tree) | tie (+0.007, +0.077; p=0.58, 0.31) |
| GUCT-Uniform, Blocksworld | works | tie (+0.000, +0.008, −0.042) |
| GUCT-Uniform, cube / 8-puzzle / Sokoban | at parity with tree | tie |

The filter is **a repair for pooling's catastrophic failure mode, not a general improvement on
merging.** Where pooling works, the filter matches it. Where pooling is broken — the cube under
MC-average backup — the filter fixes it, and so does the GUCT-Uniform backup rule. Each fix
independently repairs the same failure, and neither adds anything on top of the other.

The one surviving classical headline, verified inside budget at every point:

| tier/budget | tree | merge | reject (charged) | `apply_move` merge/reject |
|---|---|---|---|---|
| td6/b100 | 0.533 | 0.000 | **1.000** | 100 / 74.8 |
| td8/b200 | 0.400 | 0.000 | **1.000** | 200 / 106.8 |
| td10/b200 | 0.000 | 0.000 | **0.983** | 200 / 164.4 |
| td10/b400 | 0.367 | 0.000 | **1.000** | 400 / 164.9 |

> **Corrected 2026-09-21**, same mislabelling as above; see `MERGING_MECHANISM_SUMMARY.md`. The
> filter staying under budget in the runs it wins is what `charge_rejections` has to guarantee, and
> it does — but this is not evidence the filter is the more efficient arm (there is no solve-based
> exit, so a solved run stops expanding while pooling spends to the last unit).

Under UCB1 + MC-average on the cube the filter beats *both* tree and merge at charged budget. That
is a real result and it is the one to keep.

## Why the LLM results are not affected

In `search.py` a rejection is genuinely free. `expand_node` calls `generate_traces` once for all K
candidates and computes every candidate's hidden state *before* the merge/reject decision, so
filtering some of them causes no additional generation and no additional forward pass. The arms
there are matched per expansion in LLM sample batches, which is the dominant cost.

So `PROSQA_ADMISSION_FILTER_FINDINGS.md` and `COUNTDOWN_ADMISSION_FILTER_FINDINGS.md` stand as
written. That asymmetry is worth stating plainly: **in the classical domains a rejection costs an
extra state generation; in the LLM harness it costs nothing extra.** It is the reason the filter is
attractive for the latent/LLM work specifically, and it is not a general property of the technique.

## The lesson, which cost me two wrong claims

I verified the budget accounting once, on one domain, found it fine, and then wrote comparative
claims for five more domains without re-checking. Both wrong claims had the same shape: an arm that
gets extra compute for free under a convention written for a different arm.

"Budget counts expansions" was a convention chosen when every arm spent one state generation per
expansion. The filter breaks that assumption, so the convention silently stopped measuring what it
was chosen to measure. Any future arm that can *decline* work needs its own accounting audited
before its numbers are compared to anything — per domain, not once.
