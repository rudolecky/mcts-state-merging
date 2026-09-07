# GSM8K: Real-Benchmark-Scale H3

Fourth dataset, and the first non-synthetic one — real grade-school math word problems, fetched
directly from `openai/grade-school-math` (pinned commit), not the synthetic domains this
project's H3 result has been confirmed on so far (Countdown, ProsQA). Full design rationale in
the approved plan (`.claude/plans/cheerful-jumping-moler.md` at time of writing). Key, stated
scope reduction: `parse_and_verify` can only check the final numeric answer — GSM8K's
natural-language solutions have no fixed grammar to mechanically re-derive against, unlike every
other dataset in this project. "Solved" here means "got the right final number," not "produced a
verified-sound derivation."

## Stage 0 — module + tests, no network in default suite (done)

Built `datasets/gsm8k.py` (fetch, parse, `GSM8KInstance`, `build_prompt`, `parse_and_verify`).
16 new tests (15 offline via a hand-written fixture, 1 `slow`-marked hitting the real network).
Confirmed live: `openai/gsm8k`'s HF mirror only ships parquet; the original publisher's GitHub
repo ships real jsonl, pinned to commit `3101c7d5072418e28b9008a6636bde82a006892c`. Confirmed
exact canonical counts (7473 train / 1319 test) and train/test disjointness. Zero new
dependencies. Full suite green (132 total). Wired into all three integration points
(`collect.py`, `calibrate.py`, `run_search_experiment.py`), purely additive, no changes needed
to `search.py`/`model.py`/`projection.py`/`probe.py`/`geometry.py`.

## Stage 1 — calibration (done, two batches)

| Batch | Seed | n | Mean pass | In-band | At 0% | At 100% |
|---|---|---|---|---|---|---|
| Smoke | 0 | 8 | 50.0% | 5 (62.5%) | 1 | 2 |
| Full pool | 1 | 20 | 83.3% | 10 (50.0%) | 0 | 10 |

**The healthiest first-look format-compliance result of the whole investigation.** Every single
failure reason across both batches was `wrong answer: X != Y` — zero truncation, zero
unparseable-answer, zero step-line breakage. 400 tokens was enough on the first try, no
correction needed (unlike every synthetic domain, all of which needed at least one prompt/budget
adjustment). Makes sense: GSM8K is a far more "normal" task for a general instruct model than
arithmetic-derivation puzzles or board-game tactics. Batch-to-batch difficulty varied a lot
(50% vs 83% mean pass, half of batch 2 solved perfectly every time) — same variance already seen
across Connect Four's calibration batches, not new.

## Stage 2 — collection, batch 1 (done; real cost turned out to be the story)

Collected the 5 in-band puzzles from the smoke batch: 57 snapshot records, K=4 traces / 3
snapshots / 8 rollouts, 400/200-token trace/rollout budgets.

**Real per-problem cost is far higher than any prior dataset, and highly variable**: 103s, 395s,
246s, 515s, 858s per problem (mean 427s) — roughly 3-7x Connect Four's ~120-140s/problem, and
the single most expensive dataset collected so far by a wide margin. Reading: GSM8K's
natural-language multi-sentence steps are structurally longer and more variable in derivation
length than any synthetic domain's terse, uniform-length grammar — some problems need many more
reasoning steps than others, and each step itself runs longer. Scaling to the ~20-22-problem
trust bar at this rate would be the most expensive single stretch of the entire investigation —
a real, honestly-flagged tradeoff, not a blocker, but a genuine decision point given how much
compute this session has already spent.

**First probe look, n=5 (too small to trust either direction)**: overall Spearman rho was
*negative* at all three layers (-0.39 to -0.45, nominally significant on the *overall*, i.e.
between-problem-confounded, statistic), and the within-problem rho — the number that actually
matters, per this project's own established between/within decomposition — was weak and
inconsistent: -0.15 (mid, p=0.29), -0.002 (3/4, p=0.99), +0.26 (final, p=0.10). None reach
significance. This is not being read as "H1 fails on GSM8K" — n=5/57 records is exactly the
small-sample regime this project has already been burned by once (an earlier rho=0.306 at n=5
evaporated to 0.145 at n=11 in the very first Phase 0 pilot) — it is simply too small to mean
anything yet, in either direction. Recorded here plainly rather than either alarmed over or
smoothed away.

## Entropy-stratified false-merge check (extended to GSM8K, free analysis)

Extended `entropy_check.py` to GSM8K. One real finding before the entropy question itself:
**GSM8K's exact-duplicate rate is even higher than Countdown's or ProsQA's** — the p10 through
p40 percentiles of same-step projected distance all land on exactly 0 (40%+ of same-step pairs
are exact hidden-state duplicates), so the usual "calibrate tau at p40" convention is vacuous
here; used the first nonzero percentile instead (p50, tau=0.0105, false_merge_rate=4.3%,
comfortably under the plan's 15% gate).

At that tau: median-entropy-split false-merge rate is 3.8% (low) vs. 2.6% (high) — close, no
clear direction. Top-quartile split: 2.7% (low) vs. 6.7% (high) — nominally the *opposite*
direction from Countdown/ProsQA (which had false merges concentrate entirely in the low-entropy
stratum, 0% high-entropy false-merge rate in every cell tested). **Not read as a real reversal**:
the underlying counts are tiny (2 false merges out of 75 low-entropy attempts vs. 1 out of 15
high-entropy attempts) — exactly the kind of small-count noise this project has learned not to
over-interpret. What *is* solid: GSM8K's overall false-merge rate (2.6-6.7% across both splits)
clears the plan's 15% gate easily regardless of entropy stratum, matching the other two datasets'
overall conclusion even if the entropy-direction question needs more data to resolve on GSM8K
specifically.

## Stage 2, scaled to the trust bar (n=21) — the promising result did not hold

A third batch (6 in-band puzzles) brought the combined pool to 21 problems / 235 records,
matching this project's own trust bar. Re-ran the probe:

| n | Layer | Within-problem rho | p |
|---|---|---|---|
| 15 | final | 0.238 | 0.008 (significant) |
| **21** | **final** | **0.010** | **0.871** |
| 21 | mid | 0.023 | 0.748 |
| 21 | 3/4 | 0.036 | 0.632 |

**The n=15 result did not replicate — it collapsed, the same trajectory as this project's very
first correlational false alarm** (rho=0.306 at n=5 → 0.145 at n=11, in the original Phase 0
pilot). Real effects strengthen under more data in this project's experience; this one went the
other way, decisively.

**Tested, and disconfirmed, one candidate explanation**: 10 of the 21 problems have very low
within-problem `v_hat` standard deviation (0.04-0.08) — the model solves nearly every rollout at
nearly every snapshot (or the reverse), leaving little variance for a within-problem correlation
to find. Hypothesis: calibration's pass-rate filter measures *fresh, full generations from the
prompt*, while collection's `v_hat` measures rollouts *conditioned on a partial trace already
committed to* — a GSM8K problem could pass the 15-85% fresh-pass-rate band while its
conditional-value-given-partial-prefix collapses to floor/ceiling at most snapshots, plausible
specifically for word problems with one canonical operation sequence (get the setup right and
finishing is near-certain; get it wrong and it's nearly hopeless). **Tested directly**: re-ran
the probe restricted to only the 8-9 problems with real within-problem variance (std ≥ 0.15-0.20).
Signal did not recover — within-problem rho stayed small and non-significant, slightly negative
at some layers. This disconfirms the floor/ceiling explanation as the (or at least the whole)
story; reported plainly rather than reached past to protect the hypothesis.

**Where this leaves GSM8K, honestly**: unlike Countdown, ProsQA, and Connect Four — all of which
had their H3 search test gated on, and cleared, a confirmed positive within-problem probe result
— GSM8K has not cleared that gate at the scale tested so far (n=21, the same bar that was
sufficient for every other domain). This doesn't mean the premise is false here; it means it
isn't yet established, and the well-evidenced move is not to spend Stage 3's much larger compute
budget on an unconfirmed premise.

## Investigating the measurement recipe (free analysis, no new GPU time)

Checked `boundary_kind` first: 235/235 records are clean `"step"` boundaries, zero `"position"`
fallback — boundary detection itself isn't the problem.

**Solution length varies far more than any other dataset in this project**: `step_idx` ranges
from 1-2 (short problems) up to **17** (the always-fails problem rambled through 17 detected
`Step N:` lines). Countdown/ProsQA/Connect-Four all have bounded, roughly uniform solution
lengths by construction; GSM8K's real word problems don't. A "step 3" snapshot means something
structurally different in a crisp 2-step solution than in a rambling 17-step one — pooling
across that heterogeneity plausibly dilutes whatever within-problem signal exists.

Tested directly by comparing the one position every problem has in common (step_idx=1, right
after the first line — 4 records/problem, one per trace, regardless of total solution length)
against everything after it:

| Slice | n records | Layer | Within-problem rho | p |
|---|---|---|---|---|
| step_idx == 1 | 84 | final | 0.250 | **0.040** |
| step_idx == 1 | 84 | mid / 3-4 | 0.066 / -0.009 | 0.64 / 0.96 |
| step_idx >= 2 | 151 | mid / 3-4 / final | 0.032 / 0.087 / 0.017 | 0.76 / 0.43 / 0.89 (all null) |

The one structurally-comparable position shows a nominal signal at one layer; every later,
length-heterogeneous position shows nothing at any layer. Consistent with the dilution
hypothesis, not a proof of it — n=84/21 problems, one layer, p=0.04 is suggestive, not the clean
confirmed result the other three datasets got. Real lead worth following (e.g. re-running the
probe with snapshots taken at a fixed small number of *initial* steps rather than evenly-spaced
across each trace's own length) before concluding either way.

## Testing the fix: fixed-early-step snapshotting (done, promising but still n=5)

Implemented `snapshot_selection="first"` in `collect.py` (new optional param on
`collect_for_dataset`, default `"even"` preserves existing behavior for every other dataset,
zero regressions, 4 new unit tests, full suite green at 136). GSM8K's collection call now takes
the first `max_snapshots_per_trace` boundaries of each trace (steps 1, 2, 3) instead of indices
evenly spread across however many total steps that trace happened to take — fixes what a
snapshot *means* (always "right after step k") independent of a trace's eventual length.

Re-collected the exact same 5 problems from batch 1 with the new recipe (59 records, 2163.8s):

| Layer | Within-problem rho ("even", original) | Within-problem rho ("first", new) |
|---|---|---|
| mid | (p=0.29) | 0.005 (p=0.95, null) |
| 3/4 | (p=0.99) | **0.287 (p=0.040)** |
| final | (p=0.10) | 0.216 (p=0.137) |

At matched n=5, matched problems, the only thing that changed is snapshot selection, and two of
three layers moved toward significance, one crossing it. Real corroborating evidence for the
length-heterogeneity dilution diagnosis, not just a hopeful re-telling of it. **Same caveat as
every other n=5 read in this project**: this needs to hold at scale before it's trusted — the
original "even"-spacing n=15 result also looked encouraging and did not survive n=21. Not yet
scaled up.

## Fixed-early-step recipe, scaled to n=21 — also did not hold

Re-collected all three batches (21 problems total, 239 records) with `snapshot_selection="first"`
and reran the probe:

| Layer | Within-problem rho (n=5) | Within-problem rho (n=21) |
|---|---|---|
| mid | 0.005 (p=0.95) | 0.124 (p=0.076, borderline) |
| 3/4 | **0.287 (p=0.040)** | 0.076 (p=0.300) |
| final | 0.216 (p=0.137) | 0.056 (p=0.410) |

The layer that looked best at n=5 is the weakest at n=21. **This is now two independent
snapshot-selection recipes, each encouraging at small n, each failing to replicate at this
project's own trust bar.** Not one unlucky draw — a repeated pattern, across a real methodology
change that was itself well-motivated and partially corroborated at small n.

## Bottom line on GSM8K's value-signal probe

At the scale and methodology tested (n=21 problems, K=4 traces, 8 rollouts/snapshot, both
"even" and "first" snapshot selection), **GSM8K does not show a confirmed within-problem value
signal**, unlike Countdown, ProsQA, and Connect Four, all of which cleared this same gate
cleanly at comparable or smaller n. This isn't attributed to a single identifiable bug — boundary
detection is clean (100% "step", no fallback), the floor/ceiling hypothesis was tested and
disconfirmed, and the length-heterogeneity fix moved the needle at n=5 but didn't survive
scaling. The honest, current read: whatever combination of real-word-problem structure, rollout
noise at n=8, and this project's specific measurement recipe is happening here, it isn't
resolving with the interventions tried so far. A further fix would need a bigger methodology
change (e.g. substantially more rollouts per snapshot for a less noisy v_hat, or a fundamentally
different snapshot strategy) rather than another iteration at the current scale — a bigger
investment than this investigation has made so far, and not undertaken without a specific new
reason to expect it would work.

## Testing candidate fixes from the literature

Three literature-grounded hypotheses (see the artifact/session discussion for full citations:
["Reasoning Models Know When They're Right"](https://arxiv.org/abs/2504.05419),
["Temporal Predictors of Outcome"](https://arxiv.org/abs/2511.14773),
["Linear Probes Detect Task Format, Not Reasoning Mode"](https://arxiv.org/abs/2606.02907),
[Math-Shepherd](https://arxiv.org/abs/2312.08935)):

**H1 -- wrong model class (reasoning-RL-trained vs. plain instruct).** Tested directly: swapped
Qwen2.5-7B-Instruct for `DeepSeek-R1-Distill-Qwen-1.5B` then `-7B`. Result is a real friction,
not a clean confirm/refute: 1.5B mostly failed outright (8.3% mean pass, 6/8 at zero -- a
capability ceiling, uninformative either way). The size-matched 7B comparison did worse on raw
accuracy than the plain instruct model (22.9% vs. 50-83% across the original batches) *and* ran
5-10x slower per generation (122-363s/item in calibration alone). Likely cause: forcing
R1-Distill into this project's `Step N:`/`Answer:` convention fights its natural extended
`<think>` reasoning -- the cited paper's own methodology almost certainly probed natural
boundaries within the model's organic reasoning stream, not an imposed rigid format. **Not
declared refuted** -- a fair test needs harness changes (parse `<think>...</think>` structure
directly) beyond a model swap, a bigger lift than scoped here. Left open, not pursued further at
this cost/accuracy ratio.

**H3 -- format/depth confound (tested on already-collected data, free).** Checked whether
within-problem `step_idx` (a length/depth proxy) confounds `v_hat` and the probe's own
predictions, across every dataset in this project:

| Dataset | step_idx vs. v_hat (ground truth) | step_idx vs. probe's own prediction |
|---|---|---|
| countdown | rho=-0.105, p=0.164 (null) | -- |
| prosqa | rho=-0.113, p=0.149 (null) | -- |
| connect_four | rho=0.180, p=0.021 | **rho=0.218, p=0.005 -- confound present** |
| gsm8k | rho=0.196, p=0.0025 | rho=0.023, p=0.724 -- confound absent |

Nuanced, real result. **Connect Four's confirmed signal is partly riding on depth** -- the
probe's predictions track "how far into the trace" even independent of state quality (a real
caveat to attach to that dataset's result, not necessarily invalidating it: depth-awareness is
legitimately informative there too). **GSM8K shows the opposite pattern**: the true outcome
correlates with depth, but the probe's predictions don't track it at all -- meaning GSM8K's
hidden geometry fails to pick up even the *easy*, cheaply-decodable depth signal that would have
correlated with the real outcome. This reinforces H1 rather than competing with it: Qwen2.5-7B's
GSM8K hidden states don't seem to encode much task-relevant structure at all, consistent with
"needs a reasoning-trained model," not just more data.

**H2 -- rollout noise (tested).** Bumped `num_rollouts` from 8 to 12 (a first attempt at 24
produced a pathological ~2.3-hour single-problem runtime, an isolated MPS batching outlier, not
a real scaling law -- confirmed by 12 running at completely normal per-problem cost). Same 5
problems, "first"-step recipe, only `num_rollouts` changed:

| Layer | within rho, num_rollouts=8 | within rho, num_rollouts=12 |
|---|---|---|
| mid | 0.005 (p=0.95) | -0.237 (p=0.12) |
| 3/4 | **0.287 (p=0.040)** | 0.051 (p=0.72) |
| final | 0.216 (p=0.14) | 0.103 (p=0.49) |

The one layer that looked significant at 8 rollouts is flat at 12 -- more rollouts didn't clean
up the signal, and the earlier reading looks like n=5 instability rather than something rollout
noise was masking. **Null result, not a fix.**

## Building the native-boundary harness (done) and testing it (n=5, inconclusive)

Built `datasets/gsm8k_native.py` (asks only for `\boxed{}`, no `Step N:` convention; parses
answer from free text) plus a `split_fn` parameter threaded through `score_rollouts`,
`estimate_pass_rate`/`calibrate_pool`, and `collect_for_dataset` (default preserves existing
behavior for every other dataset — 146/146 tests green, zero regression). Full design in
`.claude/plans/cheerful-jumping-moler.md`.

**Stage 1 confirmed the format-fighting diagnosis directly**: DeepSeek-R1-Distill-Qwen-7B's raw
accuracy jumped from 22.9% (forced `Step N:` format) to **75-79%** (native `\boxed{}` prompting)
across two calibration batches — and 100% of failures across both batches were genuine wrong
answers, zero unparseable/missing-answer cases. `native_split`'s regex worked correctly against
real model output on the very first sample tested (manually inspected before trusting it at
scale, per the plan).

**Stage 2 (n=5, two batches merged) probe result**: within-problem rho 0.205 (mid, p=0.39),
0.191 (3/4, p=0.42), 0.063 (final, p=0.80) — directionally positive on two of three layers, not
remotely significant yet. This is the same "too small to tell" state every domain in this
project passed through before its real result emerged one way or the other — not a confirmation,
not a refutation. One collection detail worth flagging: one of the five problems
(`gsm8k_train_4662`) spontaneously produced `Step N:`-shaped lines despite never being asked to
(`boundary_kind="step"` instead of `"position"`) — a minor, real heterogeneity across problems,
though not one that breaks the per-problem-consistent `boundary_kind` invariant this pipeline
relies on.

**Cost, honestly**: per-problem collection cost was 250-650+ seconds even with the smaller
`num_traces=2`/`num_rollouts=4` used here (more conservative than every other GSM8K batch) —
still markedly slower than plain Qwen2.5-7B-Instruct despite the accuracy and format-compliance
wins. Reaching this project's own ~20-22-problem trust bar at this rate would be another
multi-hour investment on top of everything already spent chasing GSM8K's value signal.

**Where this leaves things**: the harness change worked exactly as designed mechanically (format
compliance, accuracy, clean parsing) — the literature-grounded diagnosis behind it holds up. Its
actual claim (does hidden-state value tracking work better on this model class) is still open,
not because it failed, but because n=5 genuinely isn't enough data to say either way yet.

**Scaled to n=12 (one more batch, seed=2, pool=20 → 7 more in-band).** Within-problem rho:
0.117 (mid, p=0.43), **0.230 (3/4, p=0.11)**, 0.126 (final, p=0.39). Still not significant, but
the trajectory is the meaningful part: going n=5→12, none of the three values collapsed toward
zero or flipped sign — the failure mode that killed both plain-instruct-model attempts when they
were scaled. `3/4` layer's p-value moved from 0.42→0.11, trending toward significance rather
than away from it. Not a confirmed result yet, but the first time in this entire GSM8K
investigation that scaling up made a reading *more* encouraging rather than less — the opposite
of the pattern that closed out every prior GSM8K attempt. Real cost so far for this specific
push: ~100 minutes combined (calibration + collection) for 12 problems.

**Pushed to n=18 (two more batches, seeds 3-4), the trust bar this project uses everywhere
else.** Within-problem rho:

| n | mid | 3/4 | final |
|---|---|---|---|
| 5 | 0.205 (p=0.39) | 0.191 (p=0.42) | 0.063 (p=0.80) |
| 12 | 0.117 (p=0.43) | 0.230 (p=0.11) | 0.126 (p=0.39) |
| 18 | 0.155 (p=0.21) | 0.161 (p=0.19) | 0.122 (p=0.35) |

**Not statistically significant at any layer, at any n tested.** But the trajectory is
qualitatively different from every plain-instruct-model attempt in this document: all three
values stayed positive across the entire scale-up, and by n=18 they've converged into a tight
0.12-0.16 band instead of spreading apart or flipping sign — the signature of a real, stable
(if modest) effect, not noise. Every plain-instruct attempt either collapsed toward zero or
reversed sign when scaled; this one did neither. Reading power requirements the honest way: an
effect this size (rho≈0.15) genuinely needs more than 18 problems to clear significance at
conventional confidence — this isn't yet a confirmed result, it's a stable, real-looking,
sub-significance trend.

**Pushed further to n=22 (batch 5, seed=4) — the picture got messier, not cleaner:**

| n | mid | 3/4 | final |
|---|---|---|---|
| 5 | 0.205 (p=0.39) | 0.191 (p=0.42) | 0.063 (p=0.80) |
| 12 | 0.117 (p=0.43) | 0.230 (p=0.11) | 0.126 (p=0.39) |
| 18 | 0.155 (p=0.21) | 0.161 (p=0.19) | 0.122 (p=0.35) |
| 22 | -0.087 (p=0.43) | 0.193 (p=0.09) | 0.039 (p=0.75) |

The n=18 convergence across all three layers did not hold: `mid` flipped negative, `final`
collapsed toward zero. Only `3/4` layer has stayed positive and shown a genuinely improving
trend across every single point measured (0.42→0.11→0.19→0.09) — noisy in absolute terms but
monotonically the best-performing layer, currently at its strongest reading of the whole run at
the largest n. Reported honestly rather than cherry-picking: two of three layers now look like
noise; one continues to look like it might be real. This is exactly the kind of point where a
premature "converging nicely" read (as reported at n=18) needs correcting once more data arrives
— stated here plainly rather than glossed over.

**Pushed to n=26 (batch 6, seed=5) — decisive, and it closes the question.**

| n | mid | 3/4 | final |
|---|---|---|---|
| 5 | 0.205 (p=0.39) | 0.191 (p=0.42) | 0.063 (p=0.80) |
| 12 | 0.117 (p=0.43) | 0.230 (p=0.11) | 0.126 (p=0.39) |
| 18 | 0.155 (p=0.21) | 0.161 (p=0.19) | 0.122 (p=0.35) |
| 22 | -0.087 (p=0.43) | 0.193 (p=0.09) | 0.039 (p=0.75) |
| 26 | -0.021 (p=0.86) | 0.033 (p=0.79) | 0.096 (p=0.42) |

`3/4` layer's best-ever reading (p=0.09 at n=22, the one metric that had stayed consistently
positive and improving through every prior checkpoint) collapsed to p=0.79 at n=26. All three
layers are now flat and indistinguishable from noise. **This is the exact false-alarm signature
this entire project has been calibrated to recognize since its very first correlational result**
(rho=0.306 at n=5 → 0.145 at n=11 in the original Phase 0 pilot) — a reading that looked
increasingly real through several rounds of scaling, then evaporated once enough data arrived.
Unlike that first false alarm, this one got a full, fair, patient hearing across 6 batches and
26 problems before being called.

**Conclusion: H1 does not hold up at scale.** The harness fix was real and worked exactly as
diagnosed mechanically (75-94% raw accuracy vs. 22.9% forced-format, 100% parse compliance across
every batch) — the format-fighting explanation for R1-Distill-7B's earlier poor showing was
correct. But the underlying hypothesis (reasoning-RL training gives hidden states better
value-tracking geometry) does not survive contact with enough data on this task. GSM8K's
within-problem value signal remains unconfirmed on every model and every harness tried in this
entire investigation — plain instruct (forced format), plain instruct (n/a, format was never the
issue there), and now reasoning-RL (native format). **Total cost for the full gsm8k_native
scale-up (26 problems across 6 batches): roughly 5-6 hours of combined calibration + collection
time.** This closes the last open thread in the GSM8K investigation with a real, decisive
answer, not an indefinite maybe.

## Where this leaves the three literature-grounded hypotheses (final)

All three tested to a real conclusion, none produced a confirmed fix:
- **H1** (needs reasoning-RL-trained model): the naive direct-swap test showed real friction
  (worse raw accuracy, 5-10x slower), which turned out to be a harness problem, not a refutation
  -- the native-boundary harness removed it entirely (75-94% accuracy, 100% format compliance
  across every batch). The underlying value-tracking hypothesis got a full, fair test on top of
  that fix: 26 problems, 6 batches, ~5-6 hours of compute. It looked increasingly promising
  through n=22, then collapsed to flat noise at n=26 -- the same false-alarm signature this
  project has recognized since its first correlational result. **Tested and not confirmed**, not
  left open.
- **H2** (rollout noise): tested directly, null -- more rollouts didn't help, and even reversed
  the one promising small-n reading.
- **H3** (depth/format confound): real and present for Connect Four (a caveat worth attaching to
  that dataset going forward), but *absent* for GSM8K on the plain instruct model.

**Final bottom line**: all three hypotheses closed, none rescued GSM8K's within-problem value
signal. Across every model (Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B/7B) and every
harness (forced `Step N:`, native `\boxed{}`) tried in this investigation, no configuration
produced a value signal that survived scaling to this project's own trust bar. This is now a
real, well-evidenced negative result for GSM8K specifically -- not for the project's core H3
finding, which remains solidly confirmed on Countdown and ProsQA independent of anything in this
document.

## Open decision point (updated after both recipes failed to replicate at n=21)

Six collection batches (three "even," three "first," ~3.5 hours of real model time combined)
have now been spent testing GSM8K's within-problem value signal, at real per-problem cost 3-7x
anything else in this project, without a confirmed result either way settling cleanly. Given
that track record, further iteration at the current scale/recipe is not well-justified without a
specific new idea, not just "try again." Real options from here: invest in a substantially
bigger methodology change (many more rollouts per snapshot to shrink v_hat's own sampling noise,
which no test so far has isolated as a candidate cause) if this is worth the compute; accept
GSM8K's honest result as-is — clean format compliance, real cost profile established, value
signal not confirmed after a genuine, multi-angle attempt — and treat that as the finding, same
as Connect Four's Stage 4 was deliberately stood down early once its own detour had answered the
question it existed for; or redirect remaining effort to the other still-open items from this
session (bigger/frontier model scale, a faster inference stack) rather than continuing to spend
on an unconfirmed premise here.
