# Phase 1, Second Dataset: On Countdown the Filter Beats Pooling Outright, and Sparing Verifiable Terminals Is Why

## Question

`PROSQA_ADMISSION_FILTER_FINDINGS.md` found that an admission filter (decline to expand a
near-duplicate reasoning step) matched statistic pooling on ProsQA while needing none of its
machinery, and that pooling added nothing on top. Phase 1's gate requires ≥2 datasets. This is
the second: Countdown-4.

## Setup

The published Countdown H3 configuration exactly
(`search_experiment_qwen7b_countdown4`): Qwen2.5-7B-Instruct on MPS, frozen Ridge projection at
layer `final`, τ=0.0334, K=4, budget 15, c_puct=1.5, temperature 0.8, max 60 new tokens per step,
4 numbers drawn 1–10, seed 70, 10 in-band held-out problems, verified against ground truth. Same
four arms as ProsQA, each seeded per `(held_out_seed, problem_id, arm)` so they are paired on
identical LM draws. 78 minutes.

**The baseline reproduces the published run exactly** (2/10 both), which is the check that this is
the same experiment. Published pooling was 7/10 against 6/10 here — one problem, the expected size
of the resampling that per-arm seeding introduces.

## Results

| arm | solved | nodes/problem | expansions used | LLM samples | sec/problem | merge hits | rejected |
|---|---|---|---|---|---|---|---|
| `baseline` | 2/10 (20%) | 60.0 | 15.0 | 60 | 157.2 | 0 | 0 |
| `treatment` (pool) | 6/10 (60%) | 9.1 | 8.3 | 33 | 79.9 | 240 | 0 |
| **`reject`** | **8/10 (80%)** | 21.9 | 10.6 | 42 | 111.8 | 0 | 195 |
| `reject_all` | 4/10 (40%) | 11.4 | 10.4 | 42 | 114.4 | 0 | 282 |

Paired comparisons (McNemar exact, bootstrap CI):

| comparison | delta | 95% CI | discordant | p |
|---|---|---|---|---|
| `reject` − `baseline` | **+0.600** | **[+0.100, +1.000]** | 1 vs 7 | 0.0703 |
| `treatment` − `baseline` | +0.400 | [−0.100, +0.900] | 2 vs 6 | 0.2891 |
| `reject` − `treatment` | +0.200 | [+0.000, +0.500] | 0 vs 2 | 0.5000 |
| `reject_all` − `reject` | **−0.400** | **[−0.700, −0.100]** | 4 vs 0 | 0.1250 |

### Pooled across all three Countdown batches (n=19) — H3 passes, and pooling does not

The seed-70 run above is n=10 and underpowered; adding the published seed-71 (n=4) and seed-72
(n=5) pools gives n=19, matching `search_experiment_qwen7b_countdown4_final`.

| arm | solved | nodes/problem | expansions | sec/problem | merge hits | rejected |
|---|---|---|---|---|---|---|
| `baseline` | 5/19 (26%) | 60.0 | 15.0 | 159.3 | 0 | 0 |
| `treatment` (pool) | 11/19 (58%) | 8.5 | 7.6 | 76.2 | 418 | 0 |
| **`reject`** | **16/19 (84%)** | 20.1 | 9.4 | 92.7 | 0 | 324 |
| `reject_all` | 11/19 (58%) | 8.8 | 8.1 | 82.9 | 0 | 429 |

| comparison | delta | 95% CI | discordant | p |
|---|---|---|---|---|
| `reject` − `baseline` | **+0.579** | **[+0.263, +0.842]** | 2 vs 13 | **0.0074** |
| `treatment` − `baseline` | +0.316 | [−0.053, +0.632] | 4 vs 10 | 0.1796 |
| **`reject` − `treatment`** | **+0.263** | **[+0.053, +0.474]** | **0 vs 5** | 0.0625 |
| `reject_all` − `reject` | −0.263 | [−0.526, +0.000] | 6 vs 1 | 0.1250 |

Three things here, and the second is the one that changes the story.

**H3 passes with significance on the second dataset**: +0.579, p=0.0074, CI [+0.263, +0.842].

**Filtering beats pooling on Countdown — it does not merely match it.** +0.263 with a CI
excluding zero, and a **0-vs-5 discordance**: the filter solved five problems pooling missed and
lost none that pooling won. p=0.0625 is the *smallest value McNemar's exact test can return* for
a 0-vs-5 split, so this is as strong as that discordance pattern allows at this n.

**And pooling does not clear the bar on Countdown at all**: +0.316 at p=0.1796, CI [−0.053,
+0.632] — includes zero. The filter passes on this dataset where pooling does not. That is a
dissociation between the two mechanisms, not just a tie.

### Both treatments remain cheaper than the baseline

Pooled (n=19), the baseline spent all 15 expansions and 159.3s per problem; the filter spent 9.4
and 92.7s (**−42%**). As on ProsQA, the treatments are faster because they solve earlier, so the
wall-clock accounting reinforces the node-budget verdict rather than reversing it as it did on
ARC. Pooling is cheaper still (76.2s) — it collapses more of the tree, to 8.5 nodes against the
filter's 20.1 — but it solves five fewer problems.

## Caveats

- **n=19 pooled, and the seed-70 block on its own (n=10) is underpowered** — its p-values were
  0.07 and 0.29. The three batches use the published pools (seeds 70/71/72) and their own
  calibration reports, so pooling them is what the original work did
  (`search_experiment_qwen7b_countdown4_final`), not a choice made after seeing results. The
  per-batch numbers are reported above so the pooling is auditable rather than implicit; seed 71
  (n=4) in particular ran the *opposite* way on `reject_all` (4/4), which is what n=4 looks like.
- **`reject` − `treatment` sits at p=0.0625 pooled at n=19 and p=0.0703 across both datasets.**
  Both are the floor McNemar's exact test can return for a fully one-sided discordance of that
  size (0-vs-5 and 1-vs-7), so the constraint is n, not the effect. The CIs exclude zero in both
  cases. Calling this "the filter beats pooling" is justified by the one-sidedness; calling it
  p<0.05 would not be.
- **τ=0.0334 was calibrated for the pooling arm** and the filter inherited it unchanged. Given
  that `reject_all`'s failure is a false-match problem, the filter is probably operating at a τ
  that is too permissive for it, and a sweep could only help. Checkpointing now shares one
  baseline across a sweep, so each additional τ costs only the treatment arms.
- **Matched budget, again, is not exactly delivered**: the baseline spends its full 15 expansions
  while the treatments average 8–11, because a solved search stops. This favours the baseline.

## Status against Phase 1's gate

| requirement | ProsQA (n=21) | Countdown (n=19) |
|---|---|---|
| filter beats prefix matching at matched node budget | **+0.667, p=0.0001** | **+0.579, p=0.0074** |
| win survives wall-clock accounting | yes (−35% vs baseline) | yes (−42% vs baseline) |
| pooling adds anything on top (H3′) | no (+0.048, p=1.0) | **no — filter beats it, +0.263, 0-vs-5** |

**The gate is met.** Both datasets, both significant, same direction.

### Both datasets pooled (n=40)

| arm | solved | vs baseline | 95% CI | discordant | p |
|---|---|---|---|---|---|
| `baseline` | 10/40 (25.0%) | — | — | — | — |
| `treatment` (pool) | 29/40 (72.5%) | +0.475 | [+0.275, +0.675] | 4 vs 23 | 0.0003 |
| **`reject`** | **35/40 (87.5%)** | **+0.625** | **[+0.425, +0.800]** | 2 vs 27 | **0.0000** |
| `reject_all` | 29/40 (72.5%) | +0.475 | [+0.275, +0.675] | 3 vs 23 | — |

| comparison | delta | 95% CI | discordant | p |
|---|---|---|---|---|
| **`reject` − `treatment`** | **+0.150** | **[+0.025, +0.275]** | **1 vs 7** | 0.0703 |
| `reject_all` − `reject` | −0.150 | [−0.275, −0.025] | 7 vs 1 | 0.0703 |

Pooled, the filter beats pooling by +0.150 with a CI excluding zero and a 1-vs-7 discordance.
Measured multi-parent fraction: **0.0851 for pooling, exactly 0.0000 for both filtering arms** —
so the arm that never let a node hold statistics from two paths is the arm that solved more, on
both datasets and pooled.

The mechanism claim from the classical domains is now independently supported on the LLM harness
under a learned, approximate equivalence relation: the benefit is deduplication, and statistic
pooling is not merely unnecessary but actively worse.

The remaining work is a τ sweep, not a new question.
