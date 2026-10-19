# Both Catastrophes Need Random Proposals

## Why this run existed

The approximate-relation cube sweep (`APPROXIMATE_RELATION_FINDINGS.md`) found a floor: with a bad
enough relation the filter falls below plain tree search, to 0.000 at τ=4. The Countdown sweep
(`COUNTDOWN_WIDE_TAU_FINDINGS.md`) did not reproduce it, and that write-up explained the null as
"the projection could not be made bad enough". **That explanation is wrong.**

### Correction: Countdown's relation *was* pushed to its worst possible form

`FrozenProjection` clips search-time values to `[0, 1]` (`projection.py`, `apply_projection`). So
any τ ≥ 1 already makes every live step at the same depth equivalent — the coarsest relation the
(depth, kind) bucketing allows. The byte-identical τ=1.11 and τ=2.0 runs are that clip, not a
bounded spread. (The calibration table's p99 cutoff of 1.11 exceeds 1 only because calibration uses
unclipped out-of-fold predictions.) The saved τ=2.0 graphs confirm what the filter degenerates into:

| arm | max step nodes at any depth, per problem | solved |
|---|---|---|
| filter, τ=2.0 | 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 | 4/10 |
| filter, τ=0.0334 (published) | 2, 4, 7, 3, 3, 2, 1, 5, 5, 3 | 8/10 |
| baseline | 40, 32, 40, 28, 39, 40, 40, 38, 31, 39 | 2/10 |

At τ=2.0 the filter's tree is a **single chain** in every problem — one step node per depth,
answers exempt — and it still beats the baseline. So Countdown did stress the filter maximally. The
question becomes why a single chain survives there and dies on the cube.

**Hypothesis.** At a saturated relation the filter keeps whichever child is *proposed first* at
each depth. Cube proposals are uniformly random moves (`rng.choice(untried_moves)`); Countdown's are
samples from Qwen2.5-7B. The floor may be about proposal quality, not the relation alone.

## Design

Cube, UCB1 + MC-average, oracle guidance, c=0.02, td8/budget 200, 8 puzzles × 15 repeats, relation
noise ±1.0, rejections charged. New option `expand_noise`: `None` keeps the random order; a float σ
makes `expand` try first the untried move with the lowest *oracle distance of the child + N(0, σ)* —
a proposal policy whose quality is a dial (σ=0 perfect; siblings differ by at most one distance step,
so σ=2 is mostly noise). **All three arms get the same policy.** Scoring children is not charged to
the budget, mirroring the LLM harness drawing all K candidates up front. `expand_noise=None` is a
strict no-op against the committed module: 3240 paired runs, 0 graph-signature mismatches.

## Results

| proposals | τ | tree | pool | filter |
|---|---|---|---|---|
| random | exact | 0.400 | 0.000 | 1.000 |
| random | 1.5 | 0.400 | 0.000 | 0.642 |
| random | 4 (saturated) | 0.400 | 0.000 | **0.000** |
| σ = 2 | exact | 0.025 | 0.283 | 1.000 |
| σ = 2 | 1.5 | 0.025 | 0.092 | 0.958 |
| σ = 2 | 4 (saturated) | 0.025 | 0.150 | 0.150 |
| σ = 1 | exact | 0.000 | 0.917 | 1.000 |
| σ = 1 | 1.5 | 0.000 | 0.958 | 0.992 |
| σ = 1 | 4 (saturated) | 0.000 | **0.975** | **0.975** |
| σ = 0.5 | exact / 1.5 / 4 | 0.000 | 1.000 | 1.000 |
| σ = 0 | exact / 1.5 / 4 | 0.000 | 1.000 | 1.000 |

The random rows reproduce the earlier sweep exactly (0.400 / 0.000 / 1.000 at exact, 0.000 at τ=4).

### 1. The filter's floor needs weak proposals

At the saturated relation the filter solves 0.000 with random proposals, 0.150 at σ=2, **0.975 at
σ=1** and 1.000 at σ ≤ 0.5. A relation that merges everything at a depth is catastrophic only when
the one survivor per depth is a random move. With proposals of moderate quality the chain it leaves
is a good chain — which is exactly what Countdown showed.

### 2. So does pooling's catastrophe

Pooling — 0.000 at every budget and relation quality in every earlier cube run, the project's one
catastrophic-pooling result — climbs to 0.283 at σ=2, 0.917 at σ=1 and 1.000 at σ ≤ 0.5. **The
cube's pooling failure is a property of uninformed expansion order**, not of pooling as such. With
informed proposals pool and filter coincide, as they do everywhere else pooling works.

### 3. Tree search gets *worse* with better proposals, in this MCTS variant

Tree drops from 0.400 (random) to 0.025 (σ=2) and 0.000 (σ ≤ 1). Diagnostics on 20 runs:

| proposals | solved | duplicate-state fraction | undo-to-grandparent fraction | max depth | closest distance reached |
|---|---|---|---|---|---|
| random | 8/20 | 0.70 | 0.17 | 7.0 | 1.1 |
| σ = 2 | 0/20 | 0.75 | 0.22 | 5.8 | 2.4 |
| σ = 0 | 0/20 | 0.82 | 0.29 | 5.2 | 2.8 |

The search goes shallower and spends more of itself on duplicates and on moves that undo the
previous move. This module only descends through fully expanded nodes and backs up MC averages; a
plausible reading is that best-first expansion makes a node's average fall monotonically as worse
children are added, so near-greedy UCB keeps switching between siblings before any is finished, and
the reversible cube supplies a stream of duplicate children to waste budget on. **That mechanism is
not verified.** What matters here is the consequence: every arm that deduplicates escapes it, and
"filter − tree" in the policy rows measures tree's collapse, not the filter's merit. The comparisons
that carry this document's argument are filter and pool against their own random-proposal rows.

## What this changes

| claim | before | after |
|---|---|---|
| Countdown could not stress the relation enough | Stated in the wide-τ write-up | **Wrong.** τ ≥ 1 is the coarsest relation possible (clip to [0,1]); the filter became a single chain and still beat baseline |
| A bad enough relation drives the filter below doing nothing | Measured on the cube | **Only with random proposals.** With σ ≤ 1 the saturated filter solves 0.975–1.000 |
| Pooling is catastrophic on the cube | The one catastrophic-pooling result | **Only with random proposals.** 0.917 at σ=1, 1.000 at σ ≤ 0.5 |

Both catastrophic results in this project live in a regime — expansion in uniformly random order —
that LLM search never operates in. For the latent programme that lowers the risk the cube suggested.
It also raises a question the Countdown numbers already hint at: a single chain with no real
deduplication solved 4/10 against the baseline's 2/10, so part of the filter's gain there may be
**forced depth rather than deduplication**. A random-rejection arm at a matched rejection rate would
separate the two; it has not been run.

## Caveats

- One tier (td8/b200), one exploration constant (c=0.02), oracle guidance throughout.
- The proposal policy is the oracle plus Gaussian noise. An LLM's errors are systematic and
  correlated across siblings, not independent noise; σ is a dial, not a model of a real policy.
- The tree-search degradation is observed and unexplained, and it is specific to this MCTS variant
  (full expansion before descent, MC-average backup, no priors). PUCT-style search, where the policy
  enters selection as a prior rather than as expansion order, may behave differently.
- n = 8 puzzles; p = 0.0078 is the Wilcoxon floor for 8 pairs.
