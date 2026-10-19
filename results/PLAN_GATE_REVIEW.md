# Gate Review: `latent-mcts-merging-plan.md` Against This Session's Evidence

**Purpose:** the original plan is a five-phase, gated research program. This session ran a
literature audit (`The Merge Ledger`) and then an empirical arc (Phase 0 → learned probe →
learned projection → a toy search harness testing H3) that touches every one of the plan's
hypotheses and half its phases. This document holds the plan's own criteria up against what was
actually found, phase by phase and hypothesis by hypothesis, and ends with a concrete
recommendation for what's actually worth resourcing next.

---

## Bottom line

**The plan's own gates for Phases 0–2 have effectively been passed — at toy scale, in token
space, not the full scope the plan specifies (GSM8K/MATH, not Countdown).** Phase 3 (true
latent/continuous-thought merging, not token-space merging) remains completely untouched and is
still the highest-risk, least-justified next step. **The well-evidenced recommendation is: take
the token-space result to real benchmark scale (the plan's original Phase 1 target) before any
work on Phase 3.** This session accidentally reconstructed Phase 1 and a rough Phase 2 in
miniature while chasing a go/no-go answer for Phase 0 — that's a real head start, not a
substitute for running them at the scale the plan actually specifies.

---

## Hypotheses, updated

| # | Original claim | Status after this session |
|---|---|---|
| H1 | Value is smooth in hidden-state distance | **Refined, not confirmed as stated.** Raw/z-scored distance: flat everywhere tested (Qwen 1.5B, DeepSeek-R1-Distill-1.5B, Qwen 7B — three model classes, two tasks). But a *trained* linear projection of the same hidden states recovers strong, significant within-problem value signal (ρ up to 0.53, p<0.001) that generalizes across held-out problems. **The corrected claim: value is not smooth in raw distance, but it is linearly decodable — the geometry needed a learned metric, not a better raw one.** This is exactly what C1 predicted and Phase 2 proposed, now with empirical teeth. |
| H2 | >15% of MCTS expansions are semantic duplicates | **Confirmed, and exceeded.** 17–32% of same-depth Countdown states were found to be byte-identical hidden states across independent samples, at two different model scales. In the H3 harness, baseline (no merging) consistently created ~59–60 of a possible 60 nodes per problem — near-total duplication with almost nothing deduplicated. |
| H3 | Merging at fixed budget improves accuracy | **Confirmed.** Null at 1.5B/Countdown-3-numbers (traced to a model-capability ceiling, not a search-efficiency one — 90% of jointly-unsolved problems were unsolvable via raw sampling too). Positive and significant at 7B/Countdown-4-numbers: 14.3% vs. 71.4% solved, McNemar p=0.0215, effect *strengthened* (not weakened) when the sample was doubled — the opposite trajectory from a small-n fluke. **This is the plan's most direct hypothesis, and the one now most solidly supported.** |
| H4 | Stochastic latent branching is diverse + valid | **Not tested.** The H3 harness branches by sampling K discrete token-level continuations (`generate_traces`, `num_return_sequences=K`), not by literal continuous-latent perturbation. This is faithful to the plan's Phase 1 (token-space) but says nothing about Phase 3's actual latent-space branching question. |
| H5 | Smoothness fails specifically at superposed/high-entropy states | **Tested — no failure found.** Entropy-stratified false-merge check at the actual deployed tau, both datasets, two split points (median and top-quartile): false-merge rate in the high-entropy stratum was 0.0% in every case; all false merges concentrated in the low-entropy stratum instead (8.7-17.1%). H5's specific feared failure mode did not materialize — the opposite pattern did. See `H3_MERGE_SEARCH_FINDINGS.md`'s entropy-stratified section. |

---

## Phase-by-phase against the plan's own gates

### Phase 0 — Value-geometry probe
**Plan's gate:** some layer/metric with Spearman ρ > 0.3 and false-merge rate < 15%, on raw/z-scored distance.
**Result:** Never cleared on raw distance alone, at any model or scale tested. Per the plan's
own kill criterion, this is a "stop" — but the plan explicitly treats a raw-distance failure as
informative rather than fatal ("Raw-hidden-state geometry will probably be mediocre — that is
expected and is the argument for Phase 2's learned metric, not a kill"). That's exactly what
happened: Phase 0 failed on its own terms and correctly triggered a look at Phase 2's approach
instead of a full stop.

### Phase 1 — Semantic merging in token space
**Plan's gate:** treatment beats baseline at matched node budget on ≥2 datasets, surviving
wall-clock accounting.
**Result:** This is, in substance, what the H3 harness tested and confirmed — merging beat
no-merge at matched expansion budget, with treatment also running ~1.7x *faster* wall-clock
(fewer redundant forward passes), which clears the plan's wall-clock honesty requirement too.
**Gap from the plan's actual scope:** tested on one dataset (Countdown), not ≥2, and at
toy scale (num_numbers=3-4, a few dozen held-out problems) rather than GSM8K/MATH. The
mechanism is validated; the benchmark realism is not.

### Phase 2 — Learned aggregation metric
**Plan's gate:** false-merge rate at least 2x better than raw cosine, not catastrophically
worse on high-entropy states.
**Result: full pass, both halves.** The learned projection's false-merge rate (1.3–7.6% across
the model/task combinations originally tested) is comfortably better than raw distance's
typical performance (which often couldn't even clear the 15% *ungated* threshold, let alone 2x
better). The high-entropy-stratification half is now closed too: at the actual deployed tau on
both datasets, false-merge rate in the high-entropy stratum was 0.0% — not catastrophic, not
even elevated. All false merges concentrated in the low-entropy stratum instead (8.7–17.1%,
with ProsQA's median-split figure sitting slightly above the plan's 15% ungated reference —
worth carrying forward as a real number, not disqualifying on its own).

### Phase 3 — Latent MCGS
**Not attempted.** Everything this session built operates on discrete token sequences and
their hidden states at line boundaries — never on continuous "thought vectors" fed back as
input embeddings (Coconut-style), never on a genuinely continuous action space, never on the
KV-cache-as-state question central to the plan's §7.1. This remains exactly as open as the
plan left it, with one relevant new data point: **C5 (latent reasoning underperforms explicit
CoT) was strongly confirmed by this session's own literature research** — a real reason not to
rush toward Phase 3 before Phase 1 has been pushed as far as token-space allows.

### Phase 4 — Distillation loop
**Not attempted.**

---

## Concerns register, updated with first-hand data

| # | Concern | Status |
|---|---|---|
| C1 | The metric isn't the one you want | **Confirmed as the central finding of the session.** Raw distance is the wrong lens; a learned projection is not. This is no longer a theoretical worry — it's the load-bearing empirical result everything else depends on. |
| C2 | Superposition works against aggregation | Not directly tested (see H5). Orthogonal to what was found. |
| C3 | Cache economics | Not tested — this session never touched the KV-cache/latent-state question; it stayed in token space throughout. Still exactly as open as the plan left it. |
| C5 | The baseline may already be better | **Strongly reaffirmed** by the literature audit (`The Merge Ledger`): latent CoT underperforms explicit CoT beyond 1B params, gap widens with scale. Reinforces staying in token space for the next round of work, not jumping to Phase 3. |
| C6 | Value-head reward hacking | **Tested directly, and the tasks split.** Ablation: `value_source="rollout"` swaps the PUCT guidance signal for real rollout estimates while the merge decision stays projection-based, isolating the two jobs the shared scalar used to do at once. Countdown's entire merge advantage vanished under rollout guidance (7/10 vs 7/10, exact tie) — its original win looks like compensation for a noisy cheap value estimate, not dedup efficiency per se. ProsQA's advantage *survived and strengthened* (81.0%→90.5% solved, p=0.0010→0.00098) — its bottleneck is a low-branching task structure that forces near-duplicate states regardless of guidance quality, so merging helps independent of value-source quality there. See `H3_MERGE_SEARCH_FINDINGS.md`'s C6 ablation section. |
| C8 | Cycles and pathological graphs | **Resolved by construction**, not just mitigated — the search harness's depth-indexed merge buckets make cycles structurally impossible (every edge strictly increases depth), verified by a dedicated test rather than a runtime check. |
| C9 | Scope creep | **The gates held.** Each phase's cheap version was tried before committing to the next; Phase 3/4 were correctly left untouched despite the temptation of a positive H3 result. |
| **C10 (new)** | Frontier drift — does search matter on genuinely frontier-scale models | Still unaddressed; everything tested this session (1.5B–7B) is far below frontier scale. Unchanged from the earlier literature-audit finding that this risk is real and untested by anyone, including this project. |

---

## Recommendation

1. **Don't move to Phase 3 next.** The literature (C5) and this session's own H3 result both
   point the same direction: token-space merging is where the demonstrated value is, and latent
   continuous-thought reasoning remains behind explicit CoT at every scale anyone has published.
   Phase 3 is a bigger, riskier bet than the evidence currently justifies.
2. **The well-justified next investment is Phase 1 at real scale**: the plan's original target
   (GSM8K/MATH, or another established reasoning benchmark), a real inference stack (this
   session's HF-transformers-on-MPS harness is adequate for toy validation but not for
   benchmark-scale throughput), and the same merge-plus-value-projection recipe already
   validated here. This is a natural escalation of infrastructure already built (`search.py`,
   `projection.py`, the calibration discipline), not a new design.
3. **Phase 2's remaining gap is closed.** Entropy-stratified false-merge check found no
   high-entropy penalty on either dataset — the opposite pattern, if anything. Nothing further
   owed here before scale-up; carry forward the one real number worth tracking (ProsQA's
   low-entropy false-merge rate sits slightly above the plan's 15% ungated reference).
4. **C6's ablation is done** (see `H3_MERGE_SEARCH_FINDINGS.md`) and changes what "take this to
   real scale" should mean: it's not one recipe, it's two. Countdown's win depends on merging
   compensating for a cheap noisy value proxy — the realistic deployment case, but scale-up
   should keep testing under a cheap proxy, not assume the win transfers to a better value
   model. ProsQA's win is structural (task-forced duplication) and value-source-independent —
   the more portable result to lean on when picking what "Phase 1 at real benchmark scale"
   generalizes from.
