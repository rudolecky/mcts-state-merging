# The Empty Cell: With an Approximate Relation the Filter Degrades Gracefully — Until It Goes Below Doing Nothing

## The gap this fills

Every setting where pooling failed catastrophically used an **exact** equivalence relation (the
cube under UCB1 + MC-average backup). Every setting with an **approximate** relation (a learned
Ridge projection plus τ) sat in a domain where pooling worked. So the question that decides whether
any of this transfers to the latent programme — *does the filter's repair depend on the relation
being exact?* — had never been asked.

## Design

Cube, UCB1 + MC-average, oracle guidance, c=0.02 — the configuration where pooling scores 0.000 at
every budget and difficulty. The only change: swap exact state identity for an approximate relation
shaped like `search.py`'s, so the cube becomes a controlled stand-in for the learned-relation
setting with ground truth available.

> Two states match when they sit at the **same depth** and their **1-D value differs by less than
> τ**, where the value is the oracle distance plus deterministic per-state noise. Same bucketing,
> strict `<`, nearest-wins, exhausted-candidate skip, and terminal carve-out as `find_merge_target`.

Three properties make it a fair stand-in, and all three were absent from a first version of this
experiment that had to be thrown out (see the last section):

* **The value is continuous.** Cube distance changes by exactly ±1 per move, so states in one depth
  bucket share a parity and all gaps are even — τ becomes a three-position dial whose first
  non-trivial band is *transitive*, i.e. exact identity on a coarsened space. Noise makes the
  relation continuous and non-transitive, which is what a learned projection actually is: a noisy
  estimate of value, not a coarsening of it. Noise is a SHA-256 of the state, not `hash()`, which
  is randomized per process.
* **The goal is spared**, matching `search.py`'s `reject_answer_nodes=False`.
* **The approximate relation is the only relation** — no exact merging underneath it.

Ground truth makes the false-merge rate a **measurement**, not an estimate. `key_tau=None` verified
a strict no-op against the committed version: **3240 paired runs across 108 configurations**, 0
mismatches on full graph signatures — including `value_source="oracle"`, the setting every arm
here uses, which an earlier version of this check omitted.

## Results

8 puzzles per tier, 15 repeats, arms paired on identical seeds. Noise fixed at ±1.0; τ swept.
False-merge is measured on the filter arm and given as a range across the three tiers.

| τ | false-merge | td6/100 tree · pool · **filter** | td8/200 tree · pool · **filter** | td10/400 tree · pool · **filter** |
|---|---|---|---|---|
| exact | — | 0.533 · 0.000 · **1.000** | 0.400 · 0.000 · **1.000** | 0.367 · 0.000 · **1.000** |
| 0.25 | 0.50–0.57 | 0.533 · 0.000 · **0.908** | 0.400 · 0.000 · **0.958** | 0.367 · 0.000 · **0.983** |
| 0.5 | 0.66–0.72 | 0.533 · 0.000 · **0.875** | 0.400 · 0.000 · **0.942** | 0.367 · 0.000 · **0.883** |
| 1.0 | 0.78–0.80 | 0.533 · 0.000 · **0.642** | 0.400 · 0.000 · **0.850** | 0.367 · 0.000 · **0.875** |
| 2.0 | 0.855–0.864 | 0.533 · 0.000 · 0.142 | 0.400 · 0.000 · 0.242 | 0.367 · 0.000 · 0.275 |
| 4.0 | 0.909 | 0.533 · 0.000 · 0.000 | 0.400 · 0.000 · 0.000 | 0.367 · 0.000 · 0.000 |

`filter − pool` is significant (p=0.0078 through τ=1, p=0.0312 at τ=2) and null at τ=4, where both
are 0.

### The repair does not need an exact relation

At a measured false-merge rate of **0.5** — every other match joining genuinely different states —
the filter still solves 0.91–0.98. At **0.78** it solves 0.64–0.88, still above the tree baseline on
all three tiers. The degradation is smooth and monotone in the false-merge rate along this
sweep, with no discontinuity — though the τ×noise grid below shows that holds only at fixed noise,
and that the false-merge rate is not a sufficient statistic for the damage.

This is the answer the latent programme needed: the mechanism is not an artifact of exact
equivalence.

### But there is a point where the filter is worse than having no relation at all

A follow-up sweep tested `filter − tree` directly on a 10-point grid (below). The crossing is
real, and it **moves with budget**: τ* ≈ 1.3 (td6), 1.75 (td8), 1.9 (td10). By τ=4 the filter
scores 0.000 against a tree baseline of 0.367–0.533, and the deficit is significant on all three
tiers. **Pooling is at the floor and cannot fall; the filter can, and does.**

Every earlier framing in this project said a wrong rejection "costs one discarded candidate" and
therefore does not accumulate. That is true only while branches are interchangeable. When the
rejected branch is the one on the solution path, the cost is the solution — and at high τ that
happens often enough to take the search below doing nothing.

**This floor needs random proposals.** Everything in this document expands untried moves in
uniformly random order. At a saturated relation the filter keeps whichever child is proposed first
at each depth, so the survivor is a random move. With a proposal policy of moderate quality (child
oracle distance + Gaussian noise σ=1) the saturated filter solves 0.975 instead of 0.000, and pooling
recovers too (`PROPOSAL_QUALITY_FINDINGS.md`).

### Pooling never works here, at any relation quality

0.000 in all 18 cells. The asymmetry this project rests on survives in its sharpened form: under
random proposals pooling has **no** regime where it works, and the filter has a wide one that
shrinks gracefully as the relation degrades. (With informed proposals pooling works here too —
0.917 at σ=1 — see `PROPOSAL_QUALITY_FINDINGS.md`.)

## Closing the three gaps this write-up first disclosed

The first version of this section listed `filter − tree`, the location of the crossing, and the
τ×noise plane as things not measured. All three are now measured.

### `filter − tree`, the comparison the headline actually rests on

Paired per-puzzle, same 8 puzzles × 15 repeats, bootstrap CI and Wilcoxon. `*` marks a CI
excluding zero.

| τ | false-merge | td6/100 | td8/200 | td10/400 |
|---|---|---|---|---|
| exact | — | +0.467 * | +0.600 * | +0.633 * |
| 0.25 | 0.50–0.57 | +0.375 * | +0.558 * | +0.617 * |
| 0.5 | 0.66–0.72 | +0.342 * | +0.542 * | +0.517 * |
| 1.0 | 0.78–0.80 | +0.108 | +0.450 * | +0.508 * |
| 1.2 | ~0.81 | +0.050 | +0.350 * | +0.467 * |
| 1.4 | ~0.83 | −0.083 | +0.283 * | +0.375 * |
| 1.6 | ~0.84 | −0.208 | +0.217 | +0.183 |
| 1.8 | ~0.85 | −0.300 | −0.033 | +0.108 |
| 2.0 | ~0.86 | −0.392 * | −0.158 | −0.092 |
| 4.0 | 0.909 | **−0.533 \*** | **−0.400 \*** | **−0.367 \*** |

**"The filter can go below doing nothing" is now tested rather than asserted.** At τ=4 the deficit
against plain tree search is −0.367 to −0.533 with every CI excluding zero. At τ=2 only td6 is
distinguishable from zero; the earlier statement that the filter is below tree on all three tiers
at τ=2 was true of the point estimates but not of the intervals, and is corrected here.

### The crossing is located, and it moves with budget

τ* (where `filter − tree` changes sign) is **≈1.3 on td6, ≈1.75 on td8, ≈1.9 on td10** — it rises
monotonically with tier and budget. More budget buys tolerance for a worse relation, which is the
sign the mechanism story predicts: the filter's damage is destroying branches the search then has
no budget to re-find, so more budget partly compensates. The earlier bracket of 1 < τ* < 2 was
correct but tier-blind.

### τ and noise are not interchangeable, and the false-merge rate is not a sufficient statistic

td8/200, filter solve rate / measured false-merge rate. Tree baseline 0.400, pooling 0.000
throughout.

| τ \ noise | 0.25 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|
| 0.5 | 1.000 / 0.810 | 1.000 / 0.737 | 0.942 / 0.664 | 0.517 / 0.667 |
| 1.0 | 1.000 / 0.810 | 1.000 / 0.810 | 0.850 / 0.781 | 0.317 / 0.780 |
| 2.0 | 0.242 / 0.860 | 0.242 / 0.860 | 0.242 / 0.860 | 0.050 / 0.855 |

Two things, and the second **partly retracts a claim made earlier in this document**.

**τ dominates.** At τ=2 the solve rate is 0.242 at noise 0.25, 0.5 and 1.0 alike — four times the
noise changes nothing. Noise only bites once it is comparable to τ (the τ=0.5 and τ=1.0 rows are
flat until noise reaches τ, then fall). What kills the search is how many genuinely different
states one τ-bucket admits, not how noisy the value estimate is.

**False-merge rate does not determine the damage.** (τ=0.5, noise=0.25) has a measured false-merge
rate of **0.810** and solves **1.000**; (τ=1.2, noise=1.0) has essentially the same rate, **0.811**,
and solves **0.750**. Same fraction of wrong matches, different outcomes. So the earlier claim here
that "false-merge rate predicts the degradation, monotonically" holds only **along a fixed-noise τ
sweep** and is not a general property. This is the third time in this project that false-merge rate
has failed as a diagnostic — after the second-model result and the v1 artifact — and the first time
with a clean controlled demonstration of *why*: it counts wrong matches without weighting how much
of the state space each one swallows.

## Budget accounting, measured rather than assumed

Given that an accounting error already cost this project two retracted claims, both arms' real
`apply_move` counts were measured directly rather than inferred from node counts. Mean per run:

| τ | pool `apply_move` | filter `apply_move` | filter runs over budget |
|---|---|---|---|
| exact | 100 / 200 / 400 | 74.8 / 106.8 / 164.9 | 0% / 0% / 0% |
| 0.25 | 100 / 200 / 400 | 78.2 / 113.3 / 173.5 | 11.7% / 3.3% / 1.7% |
| 0.5 | 100 / 200 / 400 | 81.0 / 114.7 / 205.1 | 15.8% / 5.0% / 10.0% |
| 1.0 | 100 / 200 / 400 | 87.0 / 134.7 / 225.4 | 25.0% / 11.7% / 10.0% |
| 2.0 | 100 / 200 / 400 | 101.7 / 190.6 / 366.5 | 64.2% / 60.8% / 57.5% |
| 4.0 | 100 / 200 / 400 | 108.0 / 204.0 / 408.0 | 100% / 100% / 100% |

Two things follow, and the first corrects a number published elsewhere in this repo.

**The published compute column is the wrong metric, but the conclusion it was defending survives.**
`MERGING_MECHANISM_SUMMARY.md` reported "generations merge/reject = 44/76". 44 is merge's *node*
count, and a pooled expansion creates no node, so it undercounts merge's real work by 2.27–2.45×
across the four rows, and 76 is `nodes + rejections`, one more than the real count (the root).
Corrected, that row reads 100/74.8. The whole point of `charge_rejections` is
that the filter must not be handed extra compute, and on the measurement that matters it is not:
**in the runs the filter wins it never exceeds pooling's budget** (0% of runs over budget in the
exact case). That is the claim the classical cube result needs, and it holds.

**What does *not* follow is that the filter is the more efficient arm.** An earlier draft of this
section said so; it is an artifact of when each arm stops doing work. `run_search` has **no
solve-based exit** — a terminal leaf is backed up and `continue`d before the `_mark_exhausted`
branch, so a solved run never marks the goal exhausted and instead spins on it, expanding nothing,
until `max_iterations = 20 × budget`. So a filter run that solves early stops consuming
`apply_move` and idles, while pooling (0.000 everywhere) keeps expanding to the last unit of
budget. Split by outcome the ranking inverts: at td6/τ=1 the filter spends 77.3 when it solves and
**103.9 when it fails** — above pooling's 100. Measured under GUCT-Uniform at td6/b40 the charged
filter costs **40.6** against merge's **40.0**, i.e. the filter is the *more* expensive arm there.
The honest reading of the table above is "compute until the run gave up", not an efficiency
comparison.

**The filter does exceed budget, and it is disclosed rather than fixed.** `run_search` charges a
rejection *after* the `expand` call returns, so a final iteration that rejects a run of moves
overshoots by that whole run — bounded at +11 expansions (the most one `expand` call can reject),
with +10 the largest observed. The overshoot lands **mostly** on failing runs — not only at high τ,
but not exclusively either: at td6, τ=0.25 has 11.7% of runs over budget against 9.2% failing, so a
few solved runs overshoot too. At high τ nearly every run fails (58–64% of runs over budget at τ=2, 100% at τ=4,
scoring 0.275 and 0.000). So the collapse is measured under a budget the filter is over-drawing,
not starving on. The exact-relation points, where the filter's advantage is largest, are clean: 0%
of runs over budget.

## What this changes

| claim | before | after |
|---|---|---|
| The filter repairs pooling's catastrophic failure | Held, exact relation only | **Holds under an approximate relation too**, to ~0.8 false-merge |
| A wrong match costs the filter one discarded branch | Stated throughout | **False past a threshold** — it can cost the solution |
| The filter is never worse than not having a relation | Implied, never tested | **False, and now tested directly.** −0.367 to −0.533 vs tree at τ=4, all CIs excluding zero |
| False-merge rate predicts the damage | Claimed earlier in this doc | **Only at fixed noise.** Same 0.81 rate gives 1.000 or 0.750 depending on τ |
| Pooling's failure depends on the relation | Untested | **No.** 0.000 at every relation quality tested |
| The filter costs ~1.7× merge's state generations | Published in the summary | **Wrong metric** (node count vs `apply_move`). With the exact relation the filter stays within budget and neither arm is reliably cheaper; at τ ≥ 1 it overshoots in 10–100% of runs, by at most one `expand` call |

## The first version of this experiment was confounded, and an auto-review caught it

Worth recording, because the confound inverted the finding and it had already been written up.

The first run used the raw integer oracle distance and no terminal carve-out. It produced a
dramatic **cliff** — 1.000 at τ≤1.5, 0.000 at τ≥3 — written up as "the repair falls off a cliff",
complete with a mechanism story. Review found three defects:

1. **τ was a three-position dial, not a continuum.** Parity within a depth bucket meant τ ∈ {0.5,
   1.5} selected identical match sets, as did {3, 6} — confirmed by identical match counts
   (80, 80, 187, 187). The "cliff" was the boundary between two discrete bands, and the first band
   was a transitive relation, i.e. not approximate at all.
2. **The goal state was being swallowed** at wide τ, so the collapse was partly the relation
   eating the answer rather than the search failing. (The per-run event counts were not retained;
   the retained v1 log has the cliff and the match counts, not this diagnostic.) `search.py` spares terminals by
   default, so the arm was doing something the module it stood in for explicitly does not.
3. **Exact merging was still live underneath**, across depth buckets, so the measured relation was
   not the only relation in force and `false_matches` undercounted it.

Fixing all three reversed the result from a cliff to smooth degradation — and restored the
prediction pre-registered in `tasks/todo.md` and then reported as falsified. The intermediate claim
that "false-merge rate does not predict the collapse" was also an artifact: with a continuous
relation it predicts it well, monotonically. (The *second-model* result that a worse projection
produced better pooling is a separate observation and is untouched by this.)

The general lesson is the one this project keeps relearning, in a new costume: **a discrete-valued
key makes a threshold sweep look like a sweep when it is sampling two points.** The grid
{0.5, 1.5, 3, 6} looked like four conditions and was two.

## Does the collapse need a large budget?

Run to settle a caveat raised while writing up `COUNTDOWN_WIDE_TAU_FINDINGS.md`: sweeping τ on
Countdown up to 60× its published value never pushed the filter below its baseline, unlike the
result above. Was that because the mechanism needs a large budget to manifest — the main sweep here
used 100–400, Countdown used 15 — and never got the chance?

Raw budget numbers are not comparable across domains: the cube expands into ~18 moves per node,
Countdown's harness into K=4 candidates, so the same integer means very different amounts of search
coverage. The fairer match is budget picked so **plain tree search lands at the same solve rate**
Countdown's baseline did (0.20 at budget 15). On the cube that is td=3/budget=28 (tree=0.167, the
closest achievable without overshooting — solve rate rises steeply with budget in this range:
0.000 at 24, 0.167 at 28, 0.333 at 30).

| τ | false-merge | tree | pool | filter | filter − tree | filter − pool |
|---|---|---|---|---|---|---|
| exact | — | 0.167 | 0.025 | 0.625 | +0.458 \* | +0.600 \* |
| 0.25 | 0.51 | 0.167 | 0.017 | 0.600 | +0.433 \* | +0.583 \* |
| 0.5 | 0.68 | 0.167 | 0.008 | 0.708 | +0.542 \* | +0.700 \* |
| 1.0 | 0.80 | 0.167 | 0.008 | 0.542 | +0.375 \* | +0.533 \* |
| 1.5 | 0.85 | 0.167 | 0.017 | 0.408 | +0.242 \* | +0.392 \* |
| 2.0 | 0.87 | 0.167 | 0.017 | 0.267 | +0.100 | +0.250 \* |
| 3.0 | 0.90 | 0.167 | 0.025 | 0.083 | −0.083 | +0.058 \* |
| 4.0 | 0.91 | 0.167 | 0.025 | 0.042 | **−0.125 \*** | +0.017 |

`*` marks a CI excluding zero.

**The collapse reproduces easily at this scale.** Pooling stays near the floor throughout
(0.008–0.025, not exactly 0.000 the way it was at larger budgets — some noise from the small number
of expansions available). The filter beats both comparators through τ=2, then falls — significantly
below tree at τ=4 (−0.125, CI [−0.217, −0.033], p=0.047), the same qualitative shape as the main
sweep, compressed into a budget an order of magnitude smaller.

**So Countdown's null is not about scale.** A budget of 28 — comparable in kind to Countdown's 15,
and matched on the metric that should matter (how hard the baseline finds the problem) — is enough
for the cube's mechanism to show up in full. Whatever stopped it from showing up on Countdown, it is
not "not enough compute for the effect to compound." An earlier version of this paragraph blamed
a projection too fine to be made bad enough; that was wrong. Countdown's relation clips to [0,1], so
τ ≥ 1 was already maximally coarse, and the filter survived as a single chain. The difference is
proposal quality — see `PROPOSAL_QUALITY_FINDINGS.md`.

## Caveats

- One domain, one guidance setting. c=0.02 is near-greedy, which maximises the cost of blocking a
  descent step; a more exploratory constant should soften the crossover and is untested.
- Noise magnitude fixed at ±1.0 against a value that changes by 1 per move. τ and noise trade off
  and only τ was swept; the two-dimensional picture is unmeasured.
- The false-merge rates are aggregates over the filter arm. The arms see structurally different
  candidate sets, so the rate is comparable across τ within an arm but not across arms.
- This relation is a *deliberately* degraded oracle. A learned projection fails differently — its
  errors are systematic, not i.i.d. noise — and that difference is not modelled here.
- With n=8 paired puzzles, p=0.0078 is the Wilcoxon **floor** (2/2⁸). It means "all 8 puzzles
  favour the filter", not a precise significance level. Where differences tie at zero they are
  dropped, cutting the effective n — which is why at τ=4 the bootstrap CI excludes zero on all
  three tiers while Wilcoxon returns p=0.125 on td10 (only 4 non-tied puzzles).
- The τ×noise plane is measured at one tier (td8) only.
