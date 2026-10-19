# Second Model: H3 Replicates, "Filtering Beats Pooling" Does Not

## Why this run existed

Every LLM result in this work used Qwen2.5-7B-Instruct with one Ridge projection. The obvious
challenge to all of it: *is this about search, or about that projection?* This run changes the
model and nothing else that can be held fixed.

Designed to be decisive either way. It was — in opposite directions for the two claims.

## Setup

Qwen2.5-1.5B-Instruct, ProsQA, projection fit on the Phase 0 pilot snapshots for that model
(`results/raw_pilot/snapshots_seed0.pkl`, 107 ProsQA records from 14 problems, layer `mid`,
α=100). Same search settings as the 7B run: τ=0.01, K=4, budget 15, c_puct 1.5, temperature 0.8,
40 new tokens/step, chain length 3–5, seed 90, three arms paired by problem (each arm draws its own LM samples — the per-arm seed includes the arm's name; corrected 2026-10-02). 42 held-out
problems, 84 minutes.

Two deliberate differences, both noted rather than hidden:

* **No in-band calibration filter.** No calibration file exists for this model, so all 42 generated
  candidates ran. The 7B run filtered 42 down to 21 in-band. This costs statistical power (floor
  and ceiling problems are concordant and contribute nothing to McNemar) but not validity, and 42
  problems partly offsets it.
* **τ=0.01 was carried over unchanged** rather than recalibrated, so that one number is held fixed
  across models. Its own calibration table puts p40 at 0.0135, so 0.01 sits just below p40 here,
  much as it did for the 7B.

Worth recording: **this projection is materially worse.** LOPO out-of-fold false-merge rate 0.24,
against the 7B's 0.113. A weaker equivalence relation is exactly the regime that should separate
the two mechanisms, since pooling is supposed to be the one that suffers from bad matches.

## Results

| arm | solved | nodes/problem | expansions | sec/problem | merge hits | rejected |
|---|---|---|---|---|---|---|
| `baseline` | 11/42 (26.2%) | 52.5 | 13.7 | 48.2 | 0 | 0 |
| `treatment` (pool) | **21/42 (50.0%)** | 12.0 | 9.2 | 32.6 | 826 | 0 |
| `reject` | 20/42 (47.6%) | 21.0 | 10.7 | 38.8 | 0 | 677 |

| comparison | delta | 95% CI | discordant | p |
|---|---|---|---|---|
| `reject` − `baseline` | **+0.214** | **[+0.071, +0.381]** | 2 vs 11 | **0.0225** |
| `treatment` − `baseline` | **+0.238** | **[+0.048, +0.429]** | 4 vs 14 | **0.0309** |
| `reject` − `treatment` | −0.024 | [−0.167, +0.119] | 5 vs 4 | 1.0000 |

## What replicates: H3

**The equivalence relation pays, on a second model, significantly.** +0.214 at p=0.0225, CI
excluding zero. That is the load-bearing Phase 1 claim and it is not an artifact of the 7B
projection.

## What does not: "filtering beats pooling"

`reject` − `treatment` = −0.024, 5 vs 4 discordant. A dead tie, and if anything pointing the other
way. Across all three settings:

| setting | n | baseline | pool | reject | `reject` − `pool` | discordant | p |
|---|---|---|---|---|---|---|---|
| 7B ProsQA | 21 | 5 | 18 | 19 | +0.048 | 1 vs 2 | 1.0000 |
| 7B Countdown | 19 | 5 | 11 | 16 | **+0.263** | 0 vs 5 | 0.0625 |
| 1.5B ProsQA | 42 | 11 | 21 | 20 | −0.024 | 5 vs 4 | 1.0000 |

Two of three settings are ties. The pooled-across-everything figure is +0.061, CI [−0.037, +0.159],
p=0.33 — and that number mixes models, so it is reported for completeness rather than relied on.

**So the H3′ claim is withdrawn to: pooling adds nothing over filtering, and filtering adds nothing
over pooling.** The Countdown result stands on its own as the one setting where the filter clearly
won, and it should now be read as a single-setting finding rather than the general pattern the
earlier write-ups took it for.

### A prediction of mine that failed

I expected the *weaker* projection to hurt pooling more, since a false match corrupts a pooled value
estimate while costing the filter only a discarded candidate. This model's false-merge rate is more
than double the 7B's (0.24 vs 0.113) — and pooling did marginally *better* than filtering here, not
worse. Whatever governs the filter-versus-pooling difference, false-merge rate as measured on
training data is not a good predictor of it.

## What this leaves standing

| claim | status |
|---|---|
| A learned approximate equivalence relation pays at matched budget (H3) | **Holds on 2 models, 2 datasets, 3 settings** |
| Pooling adds nothing on top of deduplication (H3′) | Holds — but now symmetric: neither adds over the other |
| Filtering *beats* pooling | **Withdrawn.** One setting of three; ties elsewhere |
| Deduplicating without pooling avoids pooling's catastrophic failure mode | Holds (classical cube, charged budget) |
| Answer-exemption matters when the relation is weak | Untested here (`reject_all` not run) |

The mechanism claim from the classical domains — pooling is the risk, deduplication is the benefit —
survives in its weak form: dedup is what pays, and how you cash it in matters only where pooling
breaks down. That is a narrower claim than this project argued at its most confident, and it has now
been narrowed twice by its own tests.

## Caveats

* One additional model, one dataset, one τ. "Not model-specific" is now supported for H3; two points
  is not a trend.
* The unfiltered problem set means 17 of 42 problems were solved by no arm and 7 by all three — 24
  of 42 contribute nothing to any paired test. Real power here is well under n=42.
* The 1.5B's projection was fit on chain-length 2–3 problems while the search ran on 3–5, a mild
  distribution shift the 7B run did not have.
