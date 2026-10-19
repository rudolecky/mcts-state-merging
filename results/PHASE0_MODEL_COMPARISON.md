# Phase 0 — Model-Class Comparison (options 4 & 3 follow-up)

Three configurations, same pipeline throughout:

| Config | Model | Problems | K traces | Rollouts | Best config (any layer/metric) | Gate |
|---|---|---|---|---|---|---|
| Baseline | Qwen2.5-1.5B-Instruct | 22 (8 cd + 14 pq) | 4 | N=8 | prosqa/mid/cosine ρ=0.176 (tol=0: **0.313**) | fail (false-merge 0.167) |
| R1-Distill/N=8 | DeepSeek-R1-Distill-Qwen-1.5B | 5 (prosqa only) | 3 | N=8, rollout budget 1024 tok | all configs **negative** ρ (-0.09 to -0.39) | fail |
| R1-Distill/N=16 | DeepSeek-R1-Distill-Qwen-1.5B | 5 (prosqa only) | 3 | N=16, rollout budget 2048 tok | prosqa/final/cosine/z **ρ=0.306, false-merge=0.0** | **pass** (1/12 configs) |

Countdown was dropped for R1-Distill: 0/8 calibration instances passed even at reduced
difficulty (3 numbers, range 1-6) and a 2048-token budget — failures split between genuine
reasoning errors and outright truncation/degeneration (garbled multilingual token leakage
under long exploration). This is itself informative: a 1.5B reasoning-distilled model can
lose coherence on a small combinatorial search task within a budget that comfortably solves
the same task's ProsQA-style counterpart.

## What changed between the two R1-Distill runs

Not just "more rollouts." The N=8 run used the same 1024-token cap for rollout continuations
that the original Qwen baseline used — but calibration (which sizes pass-rate off full,
2048-token traces) showed healthy in-band pass rates. Real collection then rollout-scored at
a 1024-token cap for continuations starting *mid-thought*, which silently truncated most
rollouts before they could finish (V-hat mean collapsed to 0.09, values compressed into 4
discrete levels — a floor effect from budget starvation, not genuine task difficulty). Raising
the rollout budget to match the trace budget (2048) *and* doubling N is what actually produced
the passing config — the budget fix is arguably the bigger factor of the two.

## Honest read of the one passing config

ρ=0.306 clears the bar, false-merge=0.0 clears the bar — but both numbers rest on n=62 pairs
from 5 problems, with only **7 pairs in the bottom decile** (false-merge is 0/7; one flipped
pair pushes it to ~14-29%). The scatter plot shows the apparent trend is driven by a left-tail
cluster of ~5-6 low-distance points that happen to have low |ΔV-hat|; the bulk of the data
(distance > 0.9) shows no further visible separation. This is 1 pass out of 12
layer/metric/z-scoring combinations tested on the same 62 pairs — at p=0.016, it does not
survive even a naive multiple-comparisons correction (Bonferroni needs p<0.004 across 12
tests). Directionally encouraging (cosine similarity rises monotonically from mid→3/4→final
layer: 0.10→0.23→0.29 raw, 0.07→0.14→0.31 z-scored — consistent with deeper layers carrying
more outcome-relevant signal), but not a confirmation.

## Follow-up: properly-powered rerun (pushed further per explicit request)

The 5-problem "pass" was flagged as fragile and worth re-checking at scale. Ran a second
calibration batch (seed=1, 26 more candidates, 6 survived) and collected it with the identical
K=3/2-snapshots/N=16/2048-budget design, then merged with the original 5 problems: **11
problems, 66 records, 147 same-step pairs.**

| dataset | best config | ρ | n_pairs |
|---|---|---|---|
| R1-Distill/N=16, 5 problems | final/cosine/z | **0.306** (passed) | 62 |
| R1-Distill/N=16, 11 problems (combined) | 3/4/cosine/raw | **0.145** | 147 |

**Gate fails, cleanly, at the properly-powered scale.** The scatter (n=147) shows points
spanning the full |ΔV-hat| range at every distance from ~0.8 to 1.2 z-scored cosine, no visible
separation — confirms the 5-problem result was exactly the small-n fluke it was flagged as:
tripling the data didn't refine a real effect, it erased it. The layer-depth trend that looked
directionally encouraging before (mid→3/4→final rising) is also gone; 3/4 is now the (weak)
maximum, not final.

## Bottom line

Across four configurations now tested -- Qwen-Instruct baseline, R1-Distill/N=8 (budget-bugged),
R1-Distill/N=16 at n=62, and R1-Distill/N=16 properly powered at n=147 -- **none clear the gate
in a way that survives scrutiny.** The one apparent pass evaporated under more data rather than
strengthening. This is a real, checked negative result for H1 across both model classes tested
(generic instruct and RL-distilled reasoning) on these two testbeds, not an artifact of an
under-powered pilot or a fixable measurement bug -- both of those were specifically checked and
ruled out.
