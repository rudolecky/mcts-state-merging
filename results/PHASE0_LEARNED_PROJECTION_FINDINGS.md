# Phase 0 Follow-up — Learned-Projection Findings

**Method:** for each held-out problem, take the RidgeCV probe's out-of-fold predicted
value (trained on every *other* problem — never circular) as a 1D learned projection.
Re-run the exact same H1 pair regression from geometry.py (same-step pairing, Spearman ρ,
false-merge rate, bottom-decile cutoff), but measure distance as `|oof_pred(s1) -
oof_pred(s2)|` instead of raw/z-scored cosine or L2 on the full hidden vector.

Validated first on synthetic data with a "rogue dimension" (a few high-variance,
value-irrelevant dimensions dominating raw distance — the real anisotropy mechanism, not
generic noise; plain i.i.d. noise dims turned out *not* to dilute rank correlation much,
due to concentration of measure — an instructive dead end before landing on the right
synthetic test). Confirmed: raw distance ρ<0.3, projected distance ρ>0.4 in that controlled
case.

## Result on real data

| Dataset | Layer | Same-step ρ | p | false-merge | vs. raw-distance ρ |
|---|---|---|---|---|---|
| Qwen countdown | final | **0.322** | <0.0001 | **0.0** | 0.236 (tol=0) |
| Qwen countdown | 3/4 | 0.254 | 0.0001 | 0.0 | 0.176 |
| Qwen prosqa | final | 0.217 | 0.0010 | 0.229 | 0.313 (tol=0, but that was the fragile all-pairs-diluted number) |
| R1-Distill prosqa | final | -0.290 | 0.0009 | 0.077 | see caveat below |

**Qwen/countdown/final clears the original gate** (ρ>0.3 and false-merge<0.15) for the
first time in this entire investigation with a real, large pair count (n=218, not the
5-problem fluke from earlier). Bootstrapped 95% CI (resampling at the problem level,
1000 resamples): **[0.010, 0.598]** — barely excludes zero. That reflects the real
constraint here: only 8 *problems* drive this, however many pairs they produce. Directionally
this is solid and consistent with the probe's own much more robust within-problem finding for
countdown (ρ 0.44–0.46, p<0.001, decomposed and permutation-tested) — but the projected-distance
gate-pass itself is right at the edge of what 8 problems can confirm, not a slam dunk.

### Update — tightened with 14 more problems (22 total, 604 pairs)

Ran a fresh calibration batch (seed=2, 80 candidates, 14 in-band — lower survival than the
first batch's 29%, but still enough) and merged with the original 8. Result, same Ridge
projection, same-step pairing:

| Layer | ρ | false-merge | n_pairs |
|---|---|---|---|
| mid | 0.376 | 0.013 | 604 |
| 3/4 | **0.457** | 0.013 | 604 |
| final | 0.432 | 0.013 | 604 |

**All three layers now clear the gate**, not just final. False-merge dropped to 0.013
(from 0.0 at n=8, still clean, now on 20x the pairs). Bootstrap 95% CI on 3/4 (the best
layer): **[0.330, 0.578]** — comfortably excludes zero, and its *lower* bound clears the
original 0.3 gate threshold outright. This is qualitatively different from the earlier
8-problem result: that one was "passes the point estimate, CI barely doesn't touch zero";
this one is "passes with room to spare." Raw geometry improved too (best config, 3/4/L2/
z-scored, reached ρ=0.2915 — just short of the 0.3 bar) but still doesn't clear the gate on
its own; the learned projection remains necessary to see the full effect.

**This is the first fully robust positive result in the entire investigation.**

**Qwen/prosqa improves over raw distance in ρ but not false-merge** (0.23–0.31, worse than
several raw-distance configs).

**Update — the "coarse scalar projection" explanation above was tested and rejected.**
Swapped in a PLS probe (2 and 3 supervised components instead of Ridge's single direction)
and re-ran the exact same pair test: false-merge rate barely moved (0.229 → 0.229 → 0.229
for k=1,2,3, despite ρ shifting some layer-to-layer). Root-caused it instead: 35/228 pairs
are **byte-identical raw hidden states** (independently-sampled traces of the same problem
that happened to share an identical prefix) — no projection, however rich, can separate
these, since identical inputs map to identical outputs under any faithful function. Of
those 35 exact-duplicate pairs, 8 (22.9%) disagree by >0.3 in V-hat — a number that alone
reproduces almost the entire observed false-merge rate. But excluding all 35 duplicate
pairs and recomputing on the remaining 193 gives false-merge=0.250 — *not* better. So exact
duplicates aren't the whole story either: the broader driver is **V-hat estimation noise**.
At N=8 rollouts, the binomial standard error at p=0.5 is ≈0.18 — enough on its own to make
even genuinely *close* (not just identical) states show |ΔV-hat|>0.3 by pure sampling
variance. No representation fix addresses this; it needs more rollouts per snapshot, the
same N=8→16 lever already used for R1-Distill. The projection idea remains validated for ρ
(genuine signal exists, confirmed independently via the probe); false-merge specifically
was mis-diagnosed as a representation problem when it's a measurement-noise problem.

**R1-Distill/prosqa's significant negative result should be distrusted, not reported as a
finding.** It contradicts the earlier, more carefully controlled decomposition (which found
within-problem ρ≈0, p=0.4–0.8, after properly isolating within- from between-problem effects).
The discrepancy is diagnostic: R1-Distill's between-problem probe signal is a huge (and
almost certainly spurious, 11-problem) confound, and because within-problem oof_pred variance
is genuinely near-zero for this model, `analyze_projected_distance`'s pooled same-problem-pairs
test ends up measuring residual noise in an otherwise near-constant score rather than real
signal — it isn't controlled for that degeneracy the way the explicit decomposition is. Trust
the decomposition's null result for R1-Distill, not this number.

## Bottom line

The learned-projection idea works, partially, for Qwen-class models: countdown numerically
clears the original gate (fragile-but-real, tightly matching the probe's independent
confirmation), prosqa improves ρ but at a false-merge cost that a richer (non-scalar)
projection would likely fix. For R1-Distill, there's still nothing — consistent across every
method tried (raw distance, decomposed probe, and — once the pooled test's blind spot is
accounted for — projected distance too).
