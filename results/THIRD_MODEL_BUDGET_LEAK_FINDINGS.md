# A Budget Leak in the LLM Harness

> **Superseded in part, 2026-10-02** (`LEAK_FIX_RERUN_FINDINGS.md`). The compute-matched
> re-analysis below truncates the *baseline*; it cannot see whether a deduplicating arm would have
> used its leaked budget well. Re-running 1.5B Countdown-3 with the leak actually fixed changed **no**
> outcome — the deduplicating arms' starved runs got their full budget and solved nothing more — so
> for that setting the truncation result (+0.050 / +0.075) and "the third model's apparent harm
> was the leak" are retracted: the arms are −0.125 / −0.100 against a full-budget baseline. The
> truncation numbers for the published settings are upper bounds until they are re-run the same way.

## What was run

The open question from `SECOND_MODEL_FINDINGS.md`: is 7B Countdown — the only setting where the
filter beat pooling — the outlier? A third model on Countdown answers it.

Candidates tried first:
- **Qwen2.5-3B, Countdown-4:** 3 of 60 training problems in-band (57 at pass rate 0). Too weak.
- **Qwen2.5-Math-7B, Countdown-4:** 0 of 30 pass on every problem — almost certainly an output
  format the Countdown verifier cannot parse, not inability. Not pursued.

Run instead: **Qwen2.5-1.5B on Countdown-3**, the repo's existing 1.5B Countdown setup (n=40,
held-out seed 99, budget 10, K=4, τ=0.045, layer 3/4, 48 tokens/step), all three arms fresh under the
current harness with per-arm seeding. Differences from the 7B headline: 3 numbers not 4, budget 10
not 15, no in-band filter. 25 minutes on MPS.

## Result

| arm | solved | expansions used (of 10) | LLM samples | runs stopped short *and unsolved* |
|---|---|---|---|---|
| baseline | 15/40 | 9.9 | 40 | 1 |
| pool (`treatment`) | 10/40 | 5.9 | 24 | **26** |
| filter (`reject`) | 11/40 | 6.0 | 24 | **23** |

| comparison | Δ | 95% CI | discordant | p |
|---|---|---|---|---|
| filter − baseline | −0.100 | [−0.250, +0.050] | 7 vs 3 | 0.34 |
| pool − baseline | −0.125 | [−0.275, +0.025] | 8 vs 3 | 0.23 |
| filter − pool | +0.025 | [−0.100, +0.150] | 3 vs 4 | 1.00 |

Read at face value, deduplication *hurts* here. **It should not be read at face value**: the two
deduplicating arms spent 60% of the baseline's LLM samples.

## Why: a second budget leak in the LLM harness

In the deduplicating arms 26 (pool) and 23 (filter) of 40 runs ended with budget unspent and the
problem unsolved, some after 2–4 of 10 expansions. None are the leak fixed earlier (nodes with no
admissible children: 0.0–0.2 per run). The `run_search` loop exits only on budget, on the root
having no admissible children, or on `10 × budget` iterations — and a search whose every reachable
leaf is a terminal answer spins on those answers, backing up their values, until the iteration cap.

Deduplication produces that state and plain search rarely does. When an expansion's step candidates
are all merged into existing nodes or rejected, but answer candidates are admitted (answers are
exempt from the filter), the node's only new children are answers. Once every step node in reach is
expanded that way, selection can only reach terminals. The baseline, which keeps every candidate,
almost always has an unexpanded step leaf somewhere. Answer nodes per run: baseline 18.6, filter
9.4, pool 2.8; the pooled arm has the fewest nodes of all (8.0 per problem), so it runs out first.

## It is in the published runs too

Re-reading the 7B Countdown shards from this week's sweeps:

| setting | arm | solved | mean expansions (of 15) | unsolved runs that stopped short |
|---|---|---|---|---|
| τ=0.0334 (published), seed 70, n=10 | baseline | 2 | 15.0 | 0 |
| | pool | 6 | 8.3 | 3 (used 4, 7, 8) |
| | filter | 8 | 10.6 | 2 (used 13, 13) |
| τ=0.17, seeds 71+72, n=9 | baseline | 3 | 15.0 | 0 |
| | pool | 5 | 5.3 | 4 (used 4, 4, 5, 6) |
| | filter | 6 | 5.9 | 3 (used 5, 7, 8) |

Two consequences for results already published:

1. **The deduplicating arms' wins over baseline are conservative.** They were measured while
   leaking budget the baseline never leaked. H3 is not weakened by this; it is understated.
2. **Filter − pool is confounded.** Pooling leaks more than filtering (it keeps fewer nodes), so
   part of any filter-over-pool margin may be pooling running out of search rather than pooling
   being worse. The one setting where the filter clearly beat pooling — 7B Countdown, +0.263 — is
   exactly where this matters most, and it no longer counts as evidence of an ordering until re-run
   without the leak.

## What the third model does and doesn't say

- **Filter vs pool: a tie again** (+0.025, 3 vs 4 discordant). That makes four settings — one filter
  win, three ties — and the one win is the leak-confounded one. The honest reading is that 7B
  Countdown was the outlier, pending a leak-free re-run.
- **H3 (deduplication beats baseline): not replicated here, but not contradicted either.** At
  matched compute both arms are slightly ahead of the baseline (n.s.).

## Compute-matched re-analysis (baseline truncation — an upper bound, see banner)

PUCT selection has no budget-dependent schedule and the arms are seeded per `(seed, problem, arm)`,
so a baseline truncated at *e* expansions is exactly a baseline run at budget *e*. Node ids are
assigned in creation order, which makes the truncation reconstructible from the saved graphs; the
reconstruction reproduces the recorded outcome of **all 59 baselines checked** at their own
expansion count. Truncating each baseline to the expansions its paired deduplicating arm actually
spent:

| setting | arm | solved | expansions arm / baseline | vs full baseline | vs compute-matched baseline |
|---|---|---|---|---|---|
| 7B ProsQA (n=21) | pool | 18 | 8.6 / 14.0 | +0.619, p=0.0002 | **+0.714**, p=0.0001 |
| | filter | 19 | 8.7 / 14.0 | +0.667, p=0.0001 | **+0.762**, p<0.0001 |
| 7B Countdown (n=19) | pool | 11 | 7.6 / 15.0 | +0.316, p=0.18 | **+0.579, p=0.0010** |
| | filter | 16 | 9.4 / 15.0 | +0.579, p=0.0074 | **+0.737**, p=0.0005 |
| 1.5B ProsQA (n=42) | pool | 21 | 9.2 / 13.7 | +0.238, p=0.031 | **+0.357**, p=0.0007 |
| | filter | 20 | 10.7 / 13.7 | +0.214, p=0.023 | **+0.286**, p=0.0042 |
| 1.5B Countdown-3 (n=40) | pool | 10 | 5.9 / 9.9 | −0.125, p=0.23 | +0.050, p=0.75 |
| | filter | 11 | 6.0 / 9.9 | −0.100, p=0.34 | +0.075, p=0.51 |

**Under truncation, H3 is stronger than published on every setting** — an upper bound (banner). The published claim that pooling "does not
clear the baseline significantly" on 7B Countdown is a leak artifact: at matched compute pooling
beats it at p=0.001. The third model's apparent harm disappears and turns slightly positive
(n.s.) — on the easier 3-number task the baseline truncated to 6 expansions still solves 8/40, so
there is less room for deduplication to add.

**Filter − pool cannot be compute-matched this way.** Truncating a deduplicating arm needs its
expansion order, and a merged graph's children are not created in expansion order. So the
filter-vs-pool margins stay raw — and raw, pooling is the more starved arm.

## Fixing it is a design choice, not a patch

The earlier fix deliberately stopped re-expanding a node whose candidates were all duplicates,
because re-drawing tends to regenerate the same duplicates. The leak is the cost of that choice.
Options, each of which changes every LLM number and needs a re-run:
- **Widen on starvation:** when no expandable step leaf is reachable, re-expand the best step node
  with fresh samples (temperature 0.8 gives new draws), charging it as an expansion.
- **Stop on a verified answer, and otherwise force expansion:** end the search when a correct answer
  exists; never let iterations be spent on terminals while budget remains.
- **Report compute-matched comparisons:** keep the harness and compare arms at equal LLM samples
  (truncate the baseline to the deduplicating arm's sample count per problem).

The third is now done (above) for every deduplicating-arm-vs-baseline comparison. A harness fix
is still needed for filter-vs-pool and for any run where the leak cost the arm a solve it would
otherwise have found.
