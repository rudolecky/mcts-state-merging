# Phase 1: Admission Filtering Matches Pooling on ProsQA, Without Any of Its Machinery

## Question

The classical domains showed that merging is two mechanisms — deduplicating expansions and
pooling their statistics — and that the second is where the harm lives
(`ARTICLE_DERIVED_MERGING_FINDINGS.md`). That motivated rebuilding
`latent-mcts-merging-plan.md` Phase 1 (v0.2) around an **admission filter**: use the learned
equivalence relation to *decline to expand* a near-duplicate reasoning step, rather than to pool
its statistics into an existing node.

The reason to care is specific to this plan. Everything here rests on an **approximate**
equivalence relation, and the two mechanisms fail differently under approximation:

| | a wrong equivalence costs | error with depth |
|---|---|---|
| pooling | a corrupted value estimate that then steers all later selection | compounds |
| filtering | one discarded candidate | does not accumulate |

So: does filtering actually pay, and does pooling add anything on top of it?

## Setup

The published H3 configuration exactly (`H3_MERGE_SEARCH_FINDINGS.md`): ProsQA,
Qwen2.5-7B-Instruct on MPS, frozen Ridge projection at layer `mid`, τ=0.01, K=4, budget 15,
c_puct=1.5, temperature 0.8, 21 in-band held-out problems (seed 90), verified against ground
truth. Four arms, all seeded per `(held_out_seed, problem_id, arm)` so they are paired on
identical LM draws:

| arm | equivalence relation | on a match | tree shape |
|---|---|---|---|
| `baseline` | none | — | tree |
| `treatment` | τ on projected value | pool statistics into the existing node | DAG |
| `reject` | same | discard the candidate; **answer candidates exempt** | tree |
| `reject_all` | same | discard the candidate, answers included | tree |

3h15m total on one machine.

## Results

| arm | solved | nodes/problem | expansions used | LLM samples | sec/problem | merge hits | rejected |
|---|---|---|---|---|---|---|---|
| `baseline` | 5/21 (23.8%) | 55.7 | 14.0 | 56 | 192.0 | 0 | 0 |
| `treatment` | 18/21 (85.7%) | 10.0 | 8.6 | 34 | 113.4 | 466 | 0 |
| `reject` | **19/21 (90.5%)** | 17.0 | 8.7 | 35 | 124.6 | 0 | 340 |
| `reject_all` | 18/21 (85.7%) | 11.0 | 9.1 | 36 | 128.6 | 0 | 484 |

Paired comparisons (McNemar exact, bootstrap CI on the accuracy delta):

| comparison | delta | 95% CI | discordant | p |
|---|---|---|---|---|
| `reject` − `baseline` | **+0.667** | [+0.476, +0.857] | 0 vs 14 | **0.0001** |
| `treatment` − `baseline` | +0.619 | [+0.429, +0.810] | 0 vs 13 | 0.0002 |
| `reject` − `treatment` | +0.048 | [−0.095, +0.190] | 1 vs 2 | 1.0000 |
| `reject_all` − `reject` | −0.048 | [−0.143, +0.000] | 0 vs 1 | 1.0000 |

### H3 passes, decisively

Filtering beats prefix matching by **+0.667** (5/21 → 19/21), with **14 problems solved only by
the filter and none solved only by the baseline** — a perfectly one-sided discordance. The
equivalence relation pays for itself with no DAG anywhere.

### H3′ fails, which is the interesting half

Pooling adds nothing. `reject` is 19/21 against `treatment`'s 18/21, a difference of one problem
(1 vs 2 discordant, p=1.0) — statistically indistinguishable, and if anything pointing the wrong
way for pooling. Every benefit merging showed on this domain is attributable to *not
re-expanding* duplicates, not to sharing their statistics.

That is exactly what the classical domains predicted, now confirmed on the LLM harness with a
learned, approximate equivalence relation rather than an exact one. On the second dataset it goes
further: `COUNTDOWN_ADMISSION_FILTER_FINDINGS.md` finds the filter *beating* pooling (+0.263, 0
vs 5 discordant), with pooling failing to clear the baseline significantly there at all. Pooled
across both datasets the filter is ahead by +0.150, CI [+0.025, +0.275].

**The multi-parent fraction in the filtering arms is 0.000 by construction**, and it is reported
rather than asserted: no node in either filtering arm ever held statistics from more than one
path, so the compounding failure mode had no opportunity to occur and the result was obtained
without it.

### Sparing verifiable terminals is free, and slightly positive

`reject` (answers exempt) 19/21 vs `reject_all` (answers filtered too) 18/21. One problem, not
significant — but the direction is the predicted one, the CI's upper bound touches 0, and the
mechanism is concrete: an answer can be *checked against ground truth* for one verifier call on
generation already paid for, so filtering it on an approximate projected-value match discards a
possibly-correct solution on the weakest signal available. Pooling cannot make this choice at all
(the node is shared, so the newcomer's own correctness is discarded either way). Cheap to keep,
so keep it.

### Both treatments are cheaper, not just better

`baseline` spent 14.0 of its 15 expansions and 192s per problem; the filter spent 8.7 and 124.6s.
The treatments are faster **because they solve earlier** — a solved search stops consuming
expansions. So on this domain the wall-clock accounting, which reversed the verdict on ARC,
reinforces it: the filter is better *and* ~35% cheaper than the baseline.

Against pooling the cost comparison is closer and goes the other way by a little: 124.6s vs
113.4s (+10%), and 17.0 nodes vs 10.0. Both are expected. The filter keeps duplicates out
instead of collapsing them, so it holds more distinct nodes; and a rejected candidate still paid
for its own generation before being discarded.

## τ sweep: the filter is flat across a 10× range, pooling is not

τ=0.01 was calibrated for the pooling arm and the filter inherited it, so the obvious worry is a
knife-edge. Three τ points, same 21 problems, `baseline` reused (it never consults
`find_merge_target`, so its result cannot depend on τ — which is why the checkpoint shares one
baseline across a sweep):

| τ | `baseline` | `treatment` (pool) | `reject` | `reject` − `baseline` | `reject` − `treatment` | matches fired |
|---|---|---|---|---|---|---|
| 0.003 | 5/21 | 18/21 | **19/21** | +0.667, p=0.0001 | +0.048 (1 vs 2) | 338 rej / 465 merged |
| 0.01 | 5/21 | 18/21 | **19/21** | +0.667, p=0.0001 | +0.048 (1 vs 2) | 340 rej / 466 merged |
| 0.03 | 5/21 | 17/21 | **19/21** | +0.667, p=0.0001 | +0.095 (1 vs 3) | 331 rej / 462 merged |

**The filter does not move: 19/21 at every τ, +0.667 at p=0.0001 throughout.** Pooling slips from
18 to 17 at the widest τ, so the gap between them widens with τ (+0.048 → +0.095).

That is the predicted asymmetry showing up as a robustness property. As τ grows the relation
admits more false matches; a false match costs pooling a corrupted value estimate that then steers
selection, and costs the filter one discarded candidate. The arm whose errors do not propagate is
the arm that tolerates a badly-set threshold.

The match counts also explain *why* the sweep is so flat here — 338/340/331 rejections across a
10× τ change means a 10× wider threshold selects almost the same pairs. ProsQA's projected values
are crisply separated (consistent with `PROSQA_GROUND_TRUTH_TRANSPOSITION_FINDINGS.md`: precision
0.993, recall 1.000 at τ=0.01 against ground-truth state identity). So this sweep demonstrates
*insensitivity*, and does **not** demonstrate that the filter survives a genuinely bad relation —
that would need a domain where the distances actually spread, and Countdown's `reject_all` result
is the warning that such a domain exists.

**τ=0.1 was not run** (stopped early); a fourth point wide enough to force real false matches on
ProsQA would be the informative one and is unfinished. Its 3 completed shards are left in the
checkpoint, so resuming that point costs only the remaining 39.

**Do not read the wall-clock column across τ.** `baseline` is a reused shard (192s, measured once),
and the treatment arms were timed under different machine load at different points in the sweep —
`reject` reads 149s / 125s / 185s across the three points, non-monotonically. That is contention,
not a τ effect. Only the within-run comparisons in the tables above are trustworthy on time.

## Caveats

- **n=21, one dataset, one model.** Phase 1's gate requires ≥2 datasets; this is one of them.
  The `reject` − `treatment` comparison in particular is powered only to detect a large
  difference, so "pooling adds nothing" here means "adds nothing detectable at n=21", not
  "provably adds nothing".
- **`baseline` is 5/21 here against 6/21 in the published two-arm run.** Per-arm seeding changed
  the RNG streams, so this is a fresh sample of the same configuration rather than a
  reproduction. The published `treatment` was 17/21 against 18/21 here — both arms moved by one
  problem in the same direction, which is the expected size of that resampling.
- **Matched budget is not fully delivered**, and now says so out loud: 16/21 `reject` problems
  and 18/21 `treatment` problems spent fewer than the full 15 expansions, because `run_search`
  can exit on its iteration cap once a terminal leaf dominates PUCT. The arms are matched *per
  expansion* in LLM samples (K each), not in aggregate expansions. This affects all arms
  including the published ones, and it favours the baseline if anything, since the baseline is
  the arm that actually spends its budget.
- **τ=0.01 was calibrated for the pooling arm** and the filter inherited it unchanged. The sweep
  above shows this does not matter across 10× on this dataset, but also shows why: ProsQA's
  distances barely spread, so the sweep tests insensitivity rather than robustness to a bad
  relation.

## What this changes

`latent-mcts-merging-plan.md` v0.2 predicted this shape and can now be updated from prediction to
result: **Outcome B is live, B′ is not.** Concretely, the plan's Phase 3 can drop the DAG from its
critical path — with it goes §7.1's state-definition problem (a filter never continues generation
from a merged node, so the transition function is never ill-defined), §7.4's visit-count pooling
rule, cycle handling, and C3's cache-invalidation concern.

The second dataset Phase 1's gate requires is done (`COUNTDOWN_ADMISSION_FILTER_FINDINGS.md`):
+0.579 at p=0.0074, and the filter beats pooling there rather than tying it. **The gate is met on
both datasets.**

What is genuinely unfinished, in priority order:

1. **A τ point wide enough to force false matches**, on a domain where distances actually spread.
   The sweep here shows insensitivity, not robustness — ProsQA's distances barely move across 10×.
   Countdown is the candidate domain, and its `reject_all` result (filtering verifiable answers
   costs 5 of 19 problems) is direct evidence that false matches there are consequential.
2. **A third dataset**, since the two here are both small (n=21, n=19) and both single-model.
3. **The `reject` − `treatment` p-values sit at the floor McNemar can return** for their
   one-sided discordances (0.0625–0.0703). More problems, not a different test, is what would
   move them.
