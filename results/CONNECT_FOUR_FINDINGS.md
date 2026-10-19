# Connect Four Tactical Puzzles: Ground-Truth Transposition Validation

Third dataset domain, added to test a gap Countdown/ProsQA structurally can't close: both of
those datasets validate the learned projection against *value-outcome* proxies (does merging
two similar-value states produce similar downstream correctness?), never against a
ground-truth "these two states are literally the same" label. A board game uniquely provides
that — a real transposition is two move orders reaching the byte-identical board, checkable
exactly, independent of the projection. Full design rationale in the approved plan
(`.claude/plans/cheerful-jumping-moler.md` at the time this was written); this doc tracks
results stage by stage.

## Stage 0 — engine + dataset module, no model (done)

Built `datasets/connect_four_engine.py` (board/gravity/win-check, negamax forced-win solver
with memoization) and `datasets/connect_four.py` (`ConnectFourInstance`, `generate_puzzles`,
`build_prompt`, `parse_and_verify`). 26 new tests, full suite green (113 total at the time).

Two empirical findings from actually running generation, not just designing it:

- **Yield is real but rarer than assumed.** At 5x4/K=3, random openings shorter than ~5 plies
  produce zero exact-3-ply forced wins; even at 8-9 plies it's ~0.4-0.5% of draws. Default
  `max_pre_moves` bumped from 4 to 8 based on this measurement (documented in the module
  docstring). Still cheap wall-clock — pure-Python solves are milliseconds, so even a few
  thousand attempts finish in under a second.
- **Cache-hit-rate diagnostic didn't fire at this shallow depth** — single-puzzle searches at
  K=3-5 are too shallow/narrow to revisit states within one tree, so the solver's memo cache
  showed 0 hits on individual solves. Not a red flag: the domain's transposition property is
  independently confirmed by a dedicated test (two move orders reaching an identical board),
  which is the actual load-bearing claim. Transposition-cache value showing up more at real
  search depth than at K=3-5 is expected, not broken.

## Stage 1 — small calibration-only dry run (done)

Qwen2.5-7B-Instruct, 5x4 board, K=3, `max_pre_moves=8`, pool `n_low=15, n_high=5` (only the low
bucket filled — 15 found, high bucket empty within budget, consistent with path_count>=2
positions being rarer still). `num_samples=6` per instance, both prompt variants
(move-list-only vs. move-list-with-encouraged-scratch-board) run against the *same* 15 puzzles
for a clean A/B, not a confound between "harder puzzles" and "harder prompt." 704.9s total.

| Variant | Mean pass rate | In-band (15-85%) | At 0% | At 100% |
|---|---|---|---|---|
| Strict (move-list only) | 26.7% | 7/15 (47%) | 7 | 1 |
| Scratch-board encouraged | 10.0% | 7/15 (47%) | 8 | 0 |

**Go signal, on the first try — no capability-ceiling repeat.** In-band yield (47%) is healthy,
comparable to or better than Countdown's own num_numbers=3 regime, and the overwhelming
majority of failures are genuine reasoning misses ("column N is not a forcing move," "did not
reach a win"), not format breakage — the model reliably emits parseable `Step N: drop in
column C` lines. A handful of degenerate/garbled continuations did appear (stray
non-English/gibberish tokens: "dropД виньосі в стовп 2", "drop in columnν", "drop in Saturdays'
column 0"), concentrated in the **scratch-board** variant, not the strict one — a plausible
causal story for why scratch underperformed (2.7x lower mean pass rate) rather than helping as
hypothesized: the extra instruction seems to invite more free-form drift, not better board
tracking. **Decision: use the strict move-list-only prompt going forward**, drop the
scratch-board variant.

**Open, unconfirmed hypothesis worth flagging**: "column 2 is not a forcing move" is the
single most common rejection reason, recurring across *multiple different puzzle positions* —
not just one board where column 2 happens to be a natural-looking wrong guess. Column 2 is the
literal center column on a 5-wide board. Plausible mechanism: general Connect Four strategic
knowledge from pretraining ("control the center") may be overriding the model's actual
position-specific tactical calculation, similar in flavor to the Qwen2.5-Math LaTeX-habit
rejection earlier in this project — a prior habit colliding with what the specific task
actually requires. Not confirmed (would need the raw generated text, which calibrate.py doesn't
log — only aggregated reason counts), but worth checking before/alongside Stage 2 if the
pattern persists at larger n, since a systematic center-bias would concentrate false rejections
in a specific, explainable way rather than being random noise.

## Stage 2 — small collection run, plumbing validation (done)

Added `SnapshotRecord.ground_truth_key` (default `None`, backward-compatible), an optional
`ground_truth_key_fn` param to `collect_for_dataset` (decodes the forward trace's own prefix at
each snapshot boundary and calls `connect_four.canonical_state_at`), and
`geometry.ground_truth_merge_confusion` (2x2 confusion of ground-truth exact state equality vs.
the projection's tau-threshold merge decision, skipping any pair touching a `None` key — a
no-op for countdown/prosqa). 6 new tests, full suite green (117 total).

Collected 5 puzzles (strict prompt, same 5x4/K=3 config as Stage 1, no calibration filter —
this stage checks plumbing, not difficulty), Qwen2.5-7B-Instruct, default K=4 traces / 3
snapshots / 8 rollouts per snapshot. 49 snapshot records in 621.7s.

**Plumbing works, and it's already informative, not just a correctness check:**

- 43/49 records got a real `ground_truth_key` (6 `None` from unparseable/illegal prefixes,
  plausible and handled cleanly — no crash, no poisoned keys).
- 26 distinct board states among those 43, several repeated 3x — **real transpositions found in
  actual model-generated traces**, not just the hand-built engine fixture. 22 of 63 same-step
  pairs share an exact ground-truth state.
- `ground_truth_merge_confusion` ran cleanly across 3 layers x 3 tau values (0.01/0.05/0.10):

  | Layer | tau | true_merge | false_merge | missed_merge | precision | recall |
  |---|---|---|---|---|---|---|
  | final | 0.01 | 22 | 1 | 0 | 0.957 | 1.0 |
  | final | 0.05 | 22 | 18 | 0 | 0.550 | 1.0 |
  | mid | 0.01 | 22 | 6 | 0 | 0.786 | 1.0 |
  | 3/4 | 0.01 | 22 | 5 | 0 | 0.815 | 1.0 |

**`missed_merge = 0` in every single cell tested, at every layer and every tau, including the
tightest.** The projection never fails to merge a real transposition in this data — recall is
perfect throughout. Precision is what actually varies with tau (as expected — tighter tau,
fewer false merges), and the `final` layer at tau=0.01 is the standout: 95.7% precision at
perfect recall, on real collected data, not a synthetic test. This is a stronger and more
direct result than anything the value-outcome proxy could ever show, since it's checked against
real ground truth instead of an outcome-based stand-in.

Caveat, stated plainly: n=5 puzzles is far below this project's own ~20-22-problem trust bar —
this is a preview, not a confirmed result. But it's a clean, promising one, and makes Stage 3
(scale to the real trust bar) clearly worth running rather than a speculative next step.

### Baseline check: does the learned projection beat raw (untrained) distance here?

The plan flagged this as a cheap bonus once `ground_truth_key` existed; it turned out to be a
first-order finding, not a footnote. Same 56 same-step pairs, same ground truth, z-scored raw
distance (no learned projection at all) vs. the learned projection, at matched
percentile/tau operating points:

| Method | Operating point | n_would_merge | true_merge | precision | recall |
|---|---|---|---|---|---|
| Raw L2 (z-scored) | p10 (cutoff=0) | 22 | 22 | **1.000** | **1.000** |
| Raw cosine (z-scored) | p10-p30 (cutoff=0) | 19 | 19 | **1.000** | 0.864 |
| Learned projection | tau=0.01 | 23 | 22 | 0.957 | 1.000 |
| Learned projection | tau=0.05 | 40 | 22 | 0.550 | 1.000 |

**Raw, untrained distance matches or beats the learned projection here** — the opposite of
this project's original Phase 0 finding, where raw distance failed completely at predicting
*value* and motivated learning a projection in the first place. Reading: the cutoff=0 cases at
p10-p30 are exact-duplicate hidden states (this project has repeatedly found 17-32% exact
hidden-state duplication across independently-sampled traces), and here they line up almost
perfectly with genuine ground-truth transpositions — meaning when two textually-different move
sequences produce byte-identical hidden states, they represent the same game state almost every
time in this sample. That's a real, nontrivial result on its own (nothing guarantees two
different token sequences encoding the same board collapse to the identical hidden vector), but
it reframes what the learned projection is actually for: it may be necessary for predicting
*value* from geometry, but *exact* symbolic equivalence might already be visible in raw geometry
without training anything.

**Caveat, louder than the one above**: n=5 puzzles / 56 pairs is small enough that "raw distance
gets precision=1.000" could easily be this session's own recurring false-alarm pattern (a
dramatic small-n split that doesn't survive more data) — a `cutoff=0` category built from ~20
pairs is not yet trustworthy. This comparison is now a Stage 3 headline, not a bonus: rerun both
raw-distance and projection confusion matrices at the ~20-22-problem trust bar before drawing
any real conclusion about whether the projection earns its keep on this specific task.

## Stage 3 — full correlational validation at the trust bar (done)

A second, independent calibration+collection batch (seed=2, 40-puzzle pool, 13 in-band —
puzzle-pool difficulty varies batch to batch, same as Countdown/ProsQA's own experience)
combined with Stage 2's original 5 puzzles: **17 problems, 164 snapshot records, 228
same-step pairs with known ground truth, 88 real transpositions** — close to this project's own
~20-22-problem trust bar, reached the same way Countdown's H3 result was (independent batches
accumulated, not one giant run).

**The value-prediction probe is strong and robust across both batches independently and
combined** — this domain's value signal is, if anything, more cleanly linear than
Countdown's/ProsQA's ever was:

| Scale | Layer | Spearman rho (within-problem) | p-value |
|---|---|---|---|
| n=13 (batch 2 alone) | final | 0.789 | <0.0001 |
| n=17 (combined) | final | 0.730 | <0.0001 |
| n=17 (combined) | mid | 0.768 | <0.0001 |
| n=17 (combined) | 3/4 | 0.741 | <0.0001 |

**The raw-vs-projection comparison replicated cleanly, not a small-n fluke.** At the combined
n=17 scale (228 ground-truth pairs, 88 real transpositions), `final` layer:

| Method | Cutoff (p10-30) | precision | recall | Cutoff (p50) | precision | recall |
|---|---|---|---|---|---|---|
| Raw L2 (z-scored, untrained) | exact-zero | 1.000 | 1.000 | loosest | 0.765 | 1.000 |
| Learned projection | near-zero | 0.989 | 1.000 | loosest | 0.752 | 1.000 |

**Recall is 1.000 in every single cell tested — both methods, every layer, every percentile,
both batches independently and combined.** Neither method ever misses a real transposition in
this entire dataset. Precision is where they'd differ if they were going to, and they don't,
meaningfully: at most a couple of percentage points apart in either direction depending on
layer/cutoff, with no consistent winner. **Conclusion, now well-powered rather than a n=5
preview: the learned projection provides no detectable advantage over plain untrained distance
for detecting real transpositions in Connect Four.** This is a genuinely different answer than
the original Phase 0 finding (raw distance completely failed to predict *value*, which is why a
learned projection was built in the first place) — it's domain- and question-specific: learning
a projection appears necessary for *value* prediction (the probe result above), not for *exact
symbolic equivalence detection*, where this domain's geometry already carries the signal for
free. Exact/near-exact hidden-state duplication across independently-sampled traces is, on this
evidence, already a near-perfect ground-truth-transposition detector on its own.

**What this answers, plainly**: the original question was whether this project's learned
merge-decision metric tracks real symbolic equivalence, not just value-outcome similarity. It
does — precision/recall against real ground truth is excellent throughout — but so does the
much cheaper raw-distance baseline nobody had to train. The projection's demonstrated value in
this project remains real (H3's confirmed accuracy gains on Countdown/ProsQA, both of which
lack any ground-truth equivalence signal to check against), but "does the learned metric find
real transpositions" and "is a learned metric *required* to find them" turned out to be two
different questions with two different answers.

## Stage 4 — full H3 accuracy replication: attempted, deliberately stopped

`connect_four` was wired into `run_search_experiment.py` (third dataset branch, zero changes
needed to `search.py`/`model.py`/`projection.py` — confirms the plan's own "purely additive"
integration contract). A held-out batch was drawn (seed 50, pool of 25, 6 in-band after
calibration, only 3 survived after `exclude_ids` also removed overlap with the training set —
a real, expected collision given this domain's documented collision-proneness, exactly why
`exclude_ids` exists here unlike prosqa). Result at n=3: baseline 0/3, treatment 1/3 —
directionally consistent with H3, but far too thin to read anything into (not even worth a
significance test).

**Deliberately stopped here rather than scaling to the ~10-14-problem range this would need**
(2-3 more calibrate+collect+search cycles, on the order of another hour of model time, matching
the actual cost of scaling Countdown's own H3 result). Reasoning: Stage 3 already delivered the
actual novel finding this domain was built for — does the learned merge criterion track real
ground-truth equivalence, not just a value-outcome proxy. It does, and so does plain untrained
distance. Whether merging *also* improves Connect Four search accuracy is a real open question,
but it's the same H3 question already confirmed on two other domains (Countdown, ProsQA) at
proper scale; a third repeat competes for the same scarce compute against genuinely unclaimed
next steps (real-benchmark-scale Phase 1, per `PLAN_GATE_REVIEW.md`'s standing recommendation).
Left explicitly open, not silently dropped, should it become worth revisiting.

## Stage 4, scaled (done) — a third confirmed H3 domain

Revisited once the classical no-LLM arc's own findings made closing this specific gap worth the
compute again. Reused the Stage 3 projection (`raw_qwen7b_connect_four_combined.pkl`, layer
`final`) and the original n=3 batch (seed 50) as the starting point, then added two fresh
held-out batches on the same model/config (`Qwen/Qwen2.5-7B-Instruct`, K=4, budget=10,
`tau_percentile=40.0`): seed 51 (pool 50→9 in-band→3 survived the held-out draw) and seed 52
(pool 82→17 in-band→12 survived) — yield between calibration's in-band count and the
search-time held-out draw is consistently lower than the calibration number alone suggests,
since the search step draws its own disjoint candidate set and only keeps the intersection with
calibration's in-band ids, not simply the calibration batch itself.

**Combined across all three batches, n=18:**

| | solved |
|---|---|
| baseline | 3/18 (16.7%) |
| treatment | **11/18 (61.1%)** |

Contingency: 2 both, 1 baseline-only, **9 treatment-only**, 6 neither -- heavily one-directional,
the same pattern that made ProsQA the cleanest result in the investigation. **McNemar exact
p=0.0215** (1 vs 9 discordant), matching Countdown num_numbers=4's own significance level at a
comparable n. Bootstrap 95% CI on the delta: **[0.167, 0.722]**, comfortably excluding zero.
Mean unique nodes: baseline 39.6, treatment 8.7 -- a 78% reduction, consistent with every other
confirmed H3 domain's dedup magnitude (75-83% across Countdown-4, ProsQA, and this result).

**This closes the open question Stage 4's original stopping point left explicitly on the table.**
Connect Four is now a third confirmed LLM-based H3 domain, at n=18 -- close to but not quite this
project's own 20-22-problem "fully trust" bar, already conventionally significant with a
discordant pattern this decisively one-sided. Given the existing significance and the real
compute cost of further batches (each additional ~12-problem batch took ~25 minutes of model
time on top of ~15-25 minutes of calibration), no further scaling was pursued -- the qualitative
question (does merging help Connect Four's search accuracy, with the LLM and learned projection)
is answered, matching the classical no-LLM arc's own finding for this exact domain's *search
mechanism* (`CONNECT_FOUR_KSAMPLE_FINDINGS.md`: Connect Four's classical UCB1 search is null
under its own single-simulation expansion, but flips positive once K-independent-sampling
removes its structural anti-duplicate guarantee) -- the LLM harness's batched-K-candidate
expansion has exactly that missing guarantee, so this result is the predicted outcome from that
mechanism, not an independent coincidence.
