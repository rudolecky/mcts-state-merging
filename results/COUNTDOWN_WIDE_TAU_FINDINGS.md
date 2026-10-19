# Wide τ on Countdown: the Filter Never Collapses Below Baseline, Even at the Coarsest Relation Possible

## Why this run existed

The repo's longest-standing open item: no LLM sweep had ever forced real false matches. ProsQA's
sweep (τ ∈ {0.003, 0.01, 0.03}) moved match counts by under 3% — it measured insensitivity, not
robustness. The cube's approximate-relation experiment (`APPROXIMATE_RELATION_FINDINGS.md`) then
produced a sharp, falsifiable prediction on a domain with ground truth: past a false-merge rate of
roughly 0.85, the admission filter drops **below** plain search, because a wrong rejection can
discard the state the search needed. This run tests whether that prediction transfers to the domain
the technique is actually for.

## Setup

Published Countdown H3 configuration exactly, τ swept: Qwen2.5-7B-Instruct on MPS, frozen Ridge
projection (layer `final`, α=1000), K=4, budget 15, c_puct=1.5, temperature 0.8, 60 new
tokens/step, 4 numbers drawn 1–10, held-out seed 70, 10 in-band held-out problems (reproduces the
published baseline exactly: 2/10). Arms seeded per `(seed, problem, arm)`,
paired by problem (each arm draws its own LM samples — the per-arm seed includes the arm's name; corrected 2026-10-02).

τ was **picked from the measured calibration table**, not guessed — the same discipline the cube
sweep used:

| τ | percentile | training false-merge rate (LOPO) |
|---|---|---|
| 0.0334 | p50 (published) | 0.076 |
| 0.17 | p60 | 0.164 |
| 0.36 | p70 | 0.193 |
| 0.68 | p90 | 0.311 |
| 1.11 | p99 | 0.346 |
| 2.0 | off-table | — |

Worth flagging before the result: **the training false-merge rate saturates around 0.35**, well
short of the cube's 0.91. τ=2.0 was added past the calibration table specifically to see whether
the rate keeps climbing outside it.

## Results

| τ | baseline | pool | filter | filter − baseline | filter − pool |
|---|---|---|---|---|---|
| 0.0334 | 2/10 | 6/10 | **8/10** | +0.600 [+0.10, +1.00] | **+0.200** [0, +0.50] |
| 0.17 | 2/10 | 6/10 | 7/10 | +0.500 [+0.20, +0.80] | +0.100 [−0.20, +0.40] |
| 0.36 | 2/10 | 6/10 | 5/10 | +0.300 [−0.10, +0.70] | −0.100 [−0.40, +0.20] |
| 0.68 | 2/10 | 5/10 | 4/10 | +0.200 [−0.20, +0.50] | −0.100 [−0.40, +0.20] |
| 1.11 | 2/10 | 5/10 | 4/10 | +0.200 [−0.20, +0.50] | −0.100 [−0.40, +0.20] |
| 2.0 | 2/10 | 5/10 | 4/10 | +0.200 [−0.20, +0.50] | −0.100 [−0.40, +0.20] |

McNemar (reject vs pool): 0-vs-2 discordant at published τ down to 1-vs-2 at wide τ — never
significant on this n=10 slice at any point.

### 1. The filter never falls below baseline — the cube's collapse does not reproduce here

Even 60× past the published τ, the filter floors at **4/10 (40%)**, double the 2/10 baseline. This
is the direct test of the cube's headline finding and it comes out negative: whatever mechanism let
a bad relation on the cube pull search below doing nothing does not fire here, at this budget (15
expansions) and this K (4 candidates per step).

### 2. `reject − pool` looked like it flipped sign at n=10 — it does not survive pooling seeds 71/72 to n=19

At published τ, seed 70 alone (n=10) had the filter beating pooling by +0.200. By τ=0.36 the point
estimate flipped to −0.100. Read at face value that looked like the filter's edge reversing as the
relation widened — but every CI at n=10 included zero, so it was flagged as a trend, not a finding,
and extended with seeds 71 (n=4) and 72 (n=5) at exactly τ=0.17 and τ=0.36, mirroring how the
published headline pooled the same three seeds to n=19.

**The flip does not survive.** Pooled to n=19:

| τ | baseline | pool | filter | filter − baseline | filter − pool | pool − baseline |
|---|---|---|---|---|---|---|
| 0.17 | 5/19 | 11/19 | 13/19 | **+0.421** [+0.158, +0.684] p=**0.0215** | +0.105 [−0.158, +0.368] p=0.69 | +0.316 [−0.053, +0.632] p=0.18 |
| 0.36 | 5/19 | 11/19 | 11/19 | +0.316 [0.000, +0.579] p=0.11 | **+0.000** [−0.263, +0.263] p=1.00 | +0.316 [−0.053, +0.632] p=0.18 |

At τ=0.36 the filter and pool tie exactly (11/19 each, 3-vs-3 discordant) rather than pooling
pulling ahead — the n=10 slice's −0.100 was noise from four extra problems, not the start of a real
trend. And at τ=0.17 the filter now **significantly** beats baseline (p=0.0215, CI excluding zero)
at more than 5× the published τ, which the n=10 slice alone could not show. So the corrected
reading is the opposite of the first: the filter's advantage over pooling holds (or ties) through
τ=0.36, and its advantage over doing nothing is significant well past the published setting.

### 3. τ=1.11 and τ=2.0 are byte-identical — because the relation is already as coarse as it can get

merge_hits (117), rejected (98), nodes/problem, and every solve outcome are **exactly** the same at
τ=1.11 and τ=2.0. An earlier version of this section read that as "the projection's values do not
spread far enough for a wider τ to matter", concluded the filter **could not be made bad enough**
here, and called the null insensitivity rather than robustness. **That was wrong** (corrected
2026-09-24, see `PROPOSAL_QUALITY_FINDINGS.md`).

`FrozenProjection` clips search-time values to `[0, 1]`, so any τ ≥ 1 makes every live step at the
same depth equivalent — the coarsest relation the (depth, kind) bucketing allows. Within-depth
spread in the saved graphs reaches 0.986, so the values are not compressed; τ ≥ 1 simply covers the
whole range. (The calibration table's p99 cutoff of 1.11 exceeds 1 only because calibration uses
unclipped out-of-fold predictions — a second reason it was a poor guide to how far to sweep.) The
saved τ=2.0 graphs show the result: **exactly one step node per depth in all 10 problems** — the
filter's tree is a single chain, with answers exempt — and it still solves 4/10 against the
baseline's 2/10.

So Countdown *did* stress the filter maximally. It survived, and the reason is the proposals: at a
saturated relation the filter keeps whichever child is proposed first at each depth, and here that
is a sample from Qwen2.5-7B rather than a random cube move. A cube run that varies proposal quality
confirms it — with proposals of moderate quality the saturated filter solves 0.975, not 0.000.

## What this does and does not answer

| question | answer |
|---|---|
| Does the cube's "filter falls below baseline" prediction transfer to Countdown? | **No — not observed, at this budget/K.** Floors at 2× baseline instead. |
| Is the filter robust to an arbitrarily bad relation? | **Here, yes.** τ ≥ 1 is the coarsest relation possible (values clip to [0,1]); the filter became a single chain and still beat baseline. The protection comes from the LLM's proposals, not from the relation. |
| Does `reject` still beat `pool` at wide τ? | **Yes, or ties.** At n=19 the apparent flip toward pooling at τ=0.36 (n=10) turns out to be an exact tie; the filter never trails pooling once seeds 71/72 are added. |
| Was the calibration table a reliable guide to how far the sweep needed to go? | **No.** Its cutoffs come from unclipped predictions; at search time values clip to [0,1], so everything past τ=1 is the same maximal relation. |

## Caveats

- **n=10, one seed, at four of the six τ points.** The published headline pooled seeds 70/71/72 to
  n=19; the main sweep only ran seed 70 at all six points to keep it affordable (~2h on MPS), so
  0.0334, 0.68, 1.11 and 2.0 remain n=10-only and likely underpowered. τ=0.17 and 0.36 were
  extended to n=19 (see below) specifically because the n=10 slice showed a sign flip that looked
  interesting and turned out not to survive — the same caution applies to any n=10-only point here
  that has not been re-checked.
- ~~Budget=15 and K=4 are both far smaller than the cube's budget range (100–400) and its
  per-node branching. Whether the null on "filter falls below baseline" is about domain or about
  scale is not separated by this run.~~ **Resolved: it is not scale.** A follow-up cube sweep at
  budget matched to Countdown's own baseline solve rate (td=3/budget=28 gives tree=0.167, next to
  Countdown's 0.20) still shows the collapse: pooling near-zero (0.008–0.025) and the filter
  significantly below tree at τ=4 (−0.125, CI [−0.217, −0.033], p=0.047). Raw budget numbers are
  not comparable across domains — the cube's ~18-move branching needs far more absolute budget than
  Countdown's K=4 to reach any given tree solve rate — but at *matched* solve-rate difficulty the
  mechanism reproduces easily within LLM-harness-scale expansion counts. The null here is about the
  domain, not about the search having too little compute to let the collapse manifest — and
  `PROPOSAL_QUALITY_FINDINGS.md` pins the domain difference on proposal quality. See `APPROXIMATE_RELATION_FINDINGS.md` §"Does the
  collapse need a large budget?".
- No ground truth exists for this projection, so false-merge rate here is the *training* LOPO
  estimate, not a search-time measurement the way the cube's `false_matches`/`total_matches` is.
  §3's saturation claim does not depend on that: it is read directly off the saved graphs (one
  step node per depth) and the [0,1] clip in `apply_projection`.
- Reported as a single (seed, τ) grid without a τ×K or τ×budget cross; whether budget changes the
  picture here the way it moved τ* on the cube is unmeasured.

## Extending to n=19: a checkpoint-fingerprint pitfall worth recording

The natural follow-up — seeds 71/72 at τ ∈ {0.17, 0.36} — was run by pointing all three seeds at one
shared `--output-dir`, on the assumption that the checkpoint (keyed per `(held_out_seed, problem,
arm)`) would separate them automatically. It does not: the directory itself carries a single
fingerprint over the full argument set, and `held_out_seed`/`n_low`/`n_high`/`calibrated_from` are
all part of it (correctly — they change which problems are searched). Seed 71 wrote the fingerprint
first; seed 72 was refused outright at both τ points, failing fast with no wasted LLM compute but
also producing no data. Only sweeping τ within one fixed seed (as the six-point sweep above did) may
share a directory; different seeds each need their own. Fixed by giving seed 72 its own
`--output-dir` and re-running — the corrected n=19 result is in §2 above.
