# Connect Four Guidance-Quality Ablation: The Null Does Not Flip

## Question

`H3_MERGE_SEARCH_FINDINGS.md`'s C6 ablation found Countdown num_numbers=4's merge benefit was
guidance-driven: swapping PUCT's value source from a noisy shared projection to an honest sampled
rollout collapsed a real 71.4%/14.3% split down to an exact tie. `CROSS_DOMAIN_MECHANISM_CHECK.md`
confirmed Countdown num_numbers=3's null stays null under both guidance qualities (capability
ceiling, not guidance). Connect Four's own null (`CLASSICAL_MCTS_FINDINGS.md`) has only ever been
tested under one guidance quality: an honest full random rollout. This asks the inverse: **degrade
Connect Four's value estimate and see if merging starts helping, the way C6 showed for
Countdown-4.**

## Setup

Extended `classical_mcts.py` in place (not a new module -- backward-compatible optional field,
same pattern as GSM8K-native's `split_fn`, verified by the existing test suite staying green with
zero changes): `ClassicalMCTSConfig.guidance_depth_cap` (default `None` = today's honest full
rollout). When set to a small int, `simulate()` caps the random rollout at that many plies,
returning a neutral **0.5** instead of continuing if no decisive win occurred by then -- the
classical analog of a regression-shrunk, less-discriminating value proxy (the H3 doc measured the
real projection's held-out predictions as literally biased toward the mean under distribution
shift). A decisive win found within the cap still returns the honest 1.0/0.0. Merge criterion
(exact board-state equality) is untouched throughout -- only the shared value estimate both
conditions use for UCB1 selection/backup changes, mirroring C6 exactly.

Reused the identical puzzle set and budgets from the closed main Connect Four sweep (`--n-low 30
--seed 0 --budgets 10 25 50 100`) for direct comparability. Swept `guidance_depth_cap` ∈
{None (control), 8, 4, 2} -- from honest full rollout down to a lookahead barely longer than a
single exchange.

## Results

| guidance | budget=10 | budget=25 | budget=50 | budget=100 |
|---|---|---|---|---|
| None (control) | base/treat 0.000/0.000 | 0.465/0.465 | 0.993/0.993 | 1.000/1.000 |
| cap=8 | 0.000/0.000 | 0.305/0.305 | 0.993/0.993 | 1.000/1.000 |
| cap=4 | 0.000/0.000 | 0.256/0.256 | 0.994/0.994 | 1.000/1.000 |
| cap=2 | 0.000/0.000 | 0.227/0.227 | 0.986/0.986 | 1.000/1.000 |

**Every single cell is an exact tie.** Not a small, noisy null like some of the other domains'
weaker points -- baseline and treatment solve rates are byte-identical at all 16 (guidance ×
budget) combinations tested, `wilcoxon_p=nan` throughout (zero variance in the diffs). The control
run exactly reproduces the closed findings doc's own numbers, confirming the extension didn't
regress anything. Degrading guidance *does* have a real, visible effect -- baseline's budget=25
solve rate drops steadily as the cap shrinks (0.465 → 0.305 → 0.256 → 0.227), confirming the
degraded value estimate is genuinely worse at guiding search, not a no-op -- it just affects both
conditions identically, never creating a gap between them.

## Mechanism: why this differs from Countdown, decisively

This isn't a weaker or noisier version of C6's finding -- it's a structurally different answer,
and the reason is worth being precise about. C6's guidance-driven redundancy in the LLM setting
came from a specific mechanism this project's own `classical_mcts.py` docstring already
flagged as absent in classical MCTS: **batched K-candidates-per-expansion sampling.** The LLM
harness draws K independent continuations at every expansion; when the shared value estimate is
noisy, search wanders more broadly across those K-candidate draws, hitting far more
near-duplicate continuations across different expansions/branches than a well-guided search
would -- redundancy that scales with how much the search's *attention* wanders, which noisy
guidance directly controls.

Classical MCTS never had that batching. It's single-simulation-per-iteration: one untried move
tried per visit, tracked explicitly via `untried_moves`, guaranteeing every node's own moves are
each explored exactly once before any repeat -- a **structural** guarantee that has nothing to do
with the quality of the value estimate guiding *which* untried node gets visited next. Worse
guidance in classical MCTS can only change the *order* nodes get expanded in (and, as the
budget=25 solve-rate drop shows, that order matters for how fast the tree reaches a solution) --
it cannot make baseline start creating duplicate nodes, because duplicate-node creation was never
possible for baseline in the first place (Connect Four's moves are irreversible; classical
MCTS's baseline is a genuine merge-free tree keyed by move-path by construction). There is no
redundancy latent in this setup for bad guidance to unlock, at any depth cap.

**This sharpens C6's own finding into a more general, mechanistic claim**: guidance quality only
creates a redundancy-driven merge opportunity when the search's *expansion* mechanism itself lacks
a structural anti-duplicate guarantee -- true of LLM batched-K-sampling, false of every classical
MCTS domain in this project (all four use single-simulation, untried-move-tracked expansion,
identically). The C6 guidance-vs-structure axis and this project's whole classical-arc
redundancy-density axis are not actually two independent variables -- guidance quality is a
*modulator* of redundancy that only has something to modulate when the expansion mechanism can
generate redundancy at all.

## Status

Closed. Full budget × guidance-cap grid, exact ties throughout, reconfirmed against a live control
run of the original closed result. This decisively answers the question the plan set out to test:
Connect Four's null is not guidance-quality-dependent, unlike Countdown num_numbers=4's positive
result -- and the reason is a clean, mechanistic one about what kind of expansion procedure makes
guidance quality relevant to redundancy at all.
