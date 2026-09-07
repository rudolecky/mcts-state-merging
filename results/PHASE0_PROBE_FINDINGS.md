# Phase 0 Follow-up A2 — Linear Probe Findings

**Method:** RidgeCV (z-scored hidden vector → v_hat), leave-one-problem-out CV, decomposed
into between-problem (does the probe just recognize which problem this is?) and
within-problem (does it track state-level quality within a fixed problem — the thing that
actually matters for merge-relevant value estimation) correlation, each with its own
permutation-based null (200 permutations).

## Result: real signal for Qwen-Instruct, none for R1-Distill

| Dataset | Layer | Within-problem ρ | p-value | Between-problem ρ |
|---|---|---|---|---|
| Qwen / countdown | mid | 0.282 | 0.055 | 0.512 |
| Qwen / countdown | 3/4 | **0.443** | **0.000** | 0.927 |
| Qwen / countdown | final | **0.455** | **0.000** | 0.756 |
| Qwen / prosqa | mid | **0.550** | **0.000** | 0.736 |
| Qwen / prosqa | 3/4 | **0.538** | **0.000** | 0.495 |
| Qwen / prosqa | final | **0.489** | **0.000** | 0.284 |
| R1-Distill / prosqa | mid | 0.034 | 0.800 | -0.991 |
| R1-Distill / prosqa | 3/4 | 0.043 | 0.780 | -0.964 |
| R1-Distill / prosqa | final | 0.135 | 0.400 | -0.900 |

Qwen2.5-1.5B-Instruct: significant, substantial within-problem signal in 5/6
layer×dataset configs (ρ 0.44–0.55, p<0.001). R1-Distill: no within-problem signal at all
(ρ 0.03–0.14, p 0.4–0.8) — a clean, doubly-confirmed negative for that model (matches its
flat distance-based result too).

**R1-Distill's between-problem ρ (-0.90 to -0.99) is not a real finding** — near-perfect
anti-correlation across only 11 problems is the signature of overfitting on so few
problem-level data points, not a discovered pattern. Reported for transparency, not as
evidence of anything.

**Robustness check — is the Qwen signal just rediscovering step depth?** Within-problem
correlation between step_idx and v_hat alone: countdown ρ=0.017 (n.s.) — no depth
confound at all; prosqa ρ=-0.363 (p<0.001) — depth alone predicts the *opposite sign* from
what the probe found. The probe isn't reading off position; it's tracking something else.

## What this changes

Every geometry.py (raw/z-scored distance) test in this investigation was flat. This was
read as "no signal." **That conclusion doesn't survive this test.** For Qwen-Instruct,
value *is* linearly decodable from individual hidden states in a way that generalizes
across held-out problems — the informative direction exists, it's just not the dominant
direction under raw or per-dimension-standardized distance (consistent with the
anisotropy/rogue-dimension concern the original research audit flagged for Phase 0, and
with Phase 2's whole premise: a *learned* metric, not raw distance, is the right tool).

This reframes the Phase 0 conclusion from a flat "H1 fails, stop" to something more
specific: **raw-distance H1 fails; decodability does not — for Qwen-class models.**
R1-Distill-1.5B remains a clean negative on both tests, which is itself informative
(matches the plan's own concern C5/C2-adjacent worry that small reasoning-distilled models
may not encode this the way larger ones do) but shouldn't be generalized to "reasoning
models" broadly without testing scale, per the follow-up plan's C2 gate.

---

## A3 addendum — is the geometry result actually ruled out, or just underpowered?

Bootstrap 95% CIs (resampled at the *problem* level, 1000 resamples) on the same-step
cosine/z-scored correlation, for every dataset:

| Dataset | Layer | ρ | 95% CI |
|---|---|---|---|
| Qwen countdown | final | 0.132 | [-0.059, 0.253] |
| Qwen prosqa | mid | 0.163 | [-0.071, 0.462] |
| R1-Distill prosqa | 3/4 | 0.068 | [-0.212, 0.320] |

Every CI spans zero — none of the geometry results are statistically distinguishable from
no-effect at current sample sizes. But the *point estimates* (0.13–0.16 for Qwen) sit in a
range that's plausible-but-unresolved, not ruled out: a standard power calculation says
detecting ρ=0.15 at 80% power needs **347 pairs**; ρ=0.20 needs **194**; ρ=0.30 (the
original gate) needs only **85**. Qwen/prosqa currently has 90 pairs (exact same-step) to
228 (±1 tolerance) — in the right range to resolve ρ=0.30 if it existed, but not ρ=0.15–0.20,
which is where the point estimates actually sit.

**Combined with A2:** the probe finding real within-problem signal (ρ 0.44–0.55) while raw
distance sits at an unresolved 0.13–0.16 is consistent with one specific story — the
value-relevant direction exists but occupies a small fraction of the variance raw distance
is sensitive to, so distance-based ρ is real but small, and would need ~2-4x the current
pair count to confirm rather than being genuinely zero. This argues for testing a *learned*
projection (train the probe's direction, then re-run the same distance-based merge-relevant
regression in that reduced space) before spending compute on a much bigger raw-geometry
pilot — the probe already tells us where the signal lives.
