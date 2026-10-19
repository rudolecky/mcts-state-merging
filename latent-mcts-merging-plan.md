# Latent-Space MCTS with State Merging — Implementation & Validation Plan

**Status:** draft v0.2 · research proposal, not yet resourced
**Context:** follow-up to the test-time-compute-distillation thread (Coconut / CODI / KaVa lineage)

> **v0.2 revision note.** v0.1 assumed "merging" meant one thing: identify equivalent
> reasoning states and *pool their search statistics*. The classical-domain work since then
> (`results/ARTICLE_DERIVED_MERGING_FINDINGS.md`) shows that merging is two mechanisms, and
> that they can be separated:
>
> 1. **deduplication** — do not spend expansions on a state already in the tree;
> 2. **pooling** — average every path's returns into one shared node.
>
> Deduplication is where the benefit lives. Pooling is where the risk lives. An arm that
> deduplicates *without* pooling (discard the duplicate candidate; keep the tree a tree)
> fixed the Rubik's Cube case completely — 1.000 solve rate at every budget and difficulty
> where pooling gives exactly 0.000 — and beat pooling on Blocksworld, this project's
> largest pooling win, at every budget (0.370 vs 0.158 at budget 200; 1.000 vs 0.602 at
> 1000).
>
> That matters *most* for this plan, because everything here rests on an **approximate**
> equivalence relation. A learned projection cannot do better than approximate, and the
> asymmetry between the two mechanisms under approximation is stark:
>
> | | wrong equivalence costs | error behaviour with depth |
> |---|---|---|
> | pooling | a corrupted value estimate that then steers all later selection | compounds |
> | filtering | one discarded candidate | does not accumulate |
>
> So v0.2 rebuilds Phase 1 around **admission filtering** rather than pooling, which
> retires or downgrades three of the plan's four high-severity concerns (§10 C1, C2, C3)
> without weakening the claim being tested. Phases 0 and 2 are unchanged; Phase 3's DAG
> machinery becomes optional rather than load-bearing.

---

## 1. Thesis

Token-space tree search over reasoning steps has a structural inefficiency: node identity
is exact prefix match, so two paraphrases of the same reasoning step become two distinct
subtrees and get explored twice. If reasoning states were represented in a space where
"same meaning" is "nearby," we could **stop re-exploring them** and reallocate that budget
to genuine breadth.

Note what that sentence no longer says. The v0.1 version said "we could merge them, turning
the search tree into a DAG and pooling their statistics." Turning the tree into a DAG is one
way to cash in the equivalence relation, and it is the expensive, fragile way. Simply
refusing to expand a state you already hold gets the budget back without ever letting an
approximate equivalence corrupt a value estimate.

So the revised proposal is: **search over latent reasoning states with an equivalence
relation used as an admission filter, feeding an expert-iteration distillation loop** —
with statistic pooling kept as a *second, separately-gated* treatment arm rather than as the
mechanism the project depends on.

This plan is deliberately front-loaded with cheap experiments that can kill the idea. The
expensive parts (Phase 3+) should not be built until Phases 0–2 pass.

### 1.1 What would make this worth doing

| Outcome | Claim | Why it matters |
|---|---|---|
| **A** | Latent search beats token MCTS at matched compute | New search substrate; publishable |
| **B** | Semantic *diversity enforcement* beats prefix matching in token space | Cheap win, applies to existing pipelines today. **Confirmed on 2/2 datasets: +0.625 pooled, p<0.0001** |
| **B′** | Pooling adds something on top of filtering | The v0.1 claim. **Refuted on 2/2: filtering is ahead by +0.150, CI [+0.025, +0.275]** |
| **C** | Neither, but the value-geometry results are informative | Negative result on latent reasoning's search-compatibility |

Outcome B is the realistic base case and is worth the project on its own. It is also a
**weaker and cheaper claim than v0.1's** — "do not re-explore near-duplicates" needs no DAG,
no representative-parent choice, no visit-count pooling rule, and no cycle handling — and it
is the direction that independently works in practice: the ARC Prize 2025 results were
dominated by refinement and evolutionary program synthesis, which maintain population
diversity, not by methods that share value estimates between equivalent states.

B′ is what v0.1 called B. Keeping it as a separate arm is the point: if filtering wins and
pooling adds nothing, that is a clean, publishable mechanism result, and it is the outcome
the classical domains predict.

---

## 2. Prior art to ground against

Read before writing code; each has a specific bearing on a design decision.

**Search + LLM reasoning**
- AlphaZero / expert iteration — the amortize-then-search-again loop this ultimately serves
- Tree of Thoughts — establishes step-level branching over discrete text
- rStar-Math, ReST-MCTS\* — MCTS with outcome-verified / PRM-scored steps; the token-space
  baselines to beat
- Monte Carlo *Graph* Search, transposition tables in game-playing MCTS — the DAG machinery
  already exists; we are supplying a new equivalence relation, not a new algorithm
- Licks, "MCTS with state merging for RL in Regular Decision Processes" (AIxIA 2023 DC) —
  merges by *statistical* distinguishability of empirical suffix distributions rather than exact
  state equality, i.e. the closest existing analogue to a soft equivalence relation; its stated
  limitation is the one we inherit, that a distinguishing suffix is deep and low-probability to
  sample, so the cost of the merge test itself is the bottleneck (bears on §8 C1 and the Phase 0
  false-merge gate). Its §2 is also a compact map of the MCTS-abstraction literature.

**Latent reasoning**
- Implicit CoT / stepwise internalization — curriculum that deletes trace tokens
- Coconut — feed last hidden state back as next input embedding; source of the
  superposition / BFS-like-behavior claim that this plan partly attacks
- CODI — self-distillation, explicit run supervises latent run, aligned at the answer
- KaVa — treats the KV cache as the reasoning trace; **most relevant to the state-definition
  problem in §7.1**
- System-1.5 — dynamic depth routing; a later-stage integration target

**State abstraction theory**
- Bisimulation metrics (Ferns et al.) — the *correct* notion of "same state" for planning
- DeepMDP / DBC — learned bisimulation-ish embeddings in deep RL; closest methodological template
- ε-aggregation error bounds — the theory that says merging is safe, and exactly under
  what condition it isn't (see §8, Concern C1)

---

## 3. Falsifiable hypotheses

- **H1 (value smoothness).** Latent reasoning states that are close under some learnable
  metric have similar expected outcome value. Formally: ∃ embedding φ such that
  |V(s₁) − V(s₂)| ≤ L·‖φ(s₁) − φ(s₂)‖ with usable L over the operating range.
  *If H1 fails, everything downstream fails.* Aggregation bounds have no teeth without it.
- **H2 (redundancy is real and large).** In token-space MCTS on math, a non-trivial fraction
  (target: >15%) of expanded nodes are semantic duplicates of existing nodes.
- **H3 (deduplication converts to accuracy).** *Not expanding* semantic duplicates at fixed
  node budget improves final accuracy, not just tree compactness. This is the load-bearing
  hypothesis, and it is testable without any DAG machinery.
- **H3′ (pooling adds something).** Sharing statistics between equivalent states beats merely
  declining to re-expand them. Separately gated, and the classical domains predict it fails:
  under exact equivalence pooling was neutral-to-catastrophic, and under an approximate
  equivalence it should be worse, not better.
- **H4 (latent branching is viable).** A stochastic latent step-proposal distribution can
  generate diverse-but-valid continuations, with diversity controllable and validity
  measurable.
- **H5 (superposition hurts smoothness).** Value smoothness degrades specifically at states
  the model is using to hold multiple candidate paths — i.e. H1's failure is not uniform but
  concentrated exactly where latent reasoning claims its advantage.

H5 is the interesting one. It is a *prediction of failure* and confirming it is a real result.

**What the H3/H3′ split buys.** H1 is a *smoothness* hypothesis, and it is load-bearing for
pooling because ε-aggregation bounds are what license averaging two states' values together.
Filtering needs something weaker: only that two states within τ are *interchangeable enough
that exploring both is a poor use of budget*. A metric can satisfy the second while failing
the first — which is roughly what Phase 0 found (raw-distance H1 fails; a learned projection's
value decodability does not). So the H3 arm remains testable in exactly the regime where H1's
strict form is already known to be shaky, and H3′ is the arm that needs H1 to hold.

---

## 4. Phase 0 — Value-geometry probe (go/no-go, ~1 day GPU)

**Purpose:** test H1 before building anything. This is the single highest-value experiment
in the plan and must run first.

**Procedure**
1. Pick a math reasoning model with a strong CoT baseline (e.g. Qwen2.5-Math-7B-Instruct)
   and GSM8K + MATH-500.
2. For each problem, sample CoT traces; snapshot hidden state at reasoning-step boundaries
   (last-token residual stream at layer ℓ, sweep ℓ over {mid, 3/4, final}).
3. For each snapshot state *s*: roll out N=16 continuations to a final answer, score against
   ground truth → empirical V̂(s) ∈ [0,1].
4. Build all within-problem state pairs. Regress |V̂(s₁) − V̂(s₂)| on ‖h₁ − h₂‖
   (cosine and L2, raw and after whitening).

**Read the result as:**
- Spearman ρ, and the conditional distribution of |ΔV| given small distance
- **The quantity that actually matters:** P(|ΔV| > 0.3 | distance in bottom decile) — the
  false-merge rate. A high Spearman with a fat tail of near-but-divergent pairs is still fatal.
- Layer sensitivity: does any layer give usable geometry?

**Gate:** proceed only if some layer/metric gives ρ > 0.3 **and** false-merge rate < 15%.
Raw-hidden-state geometry will probably be mediocre — that is expected and is the argument
for Phase 2's learned metric, not a kill. The kill condition is *no* layer showing signal,
which would mean outcome value is not a smooth function of the residual stream at all.

**Deliverable:** `phase0_value_geometry.ipynb` + a one-page memo with the scatter plot,
per-layer table, and an explicit go/no-go call.

---

## 5. Phase 1 — Semantic diversity enforcement in token space (the cheap real win)

**Purpose:** H2 and H3, with *no latent machinery and no DAG at all*. If an equivalence
relation cannot pay for itself here — where a good metric is available off the shelf — it
will not pay in latent space where the metric must be learned.

**Already implemented.** `search.py` carries all four arms
(`SearchConfig.merge_enabled` / `reject_duplicates` / `reject_answer_nodes`), and
`run_search_experiment.py --arms` selects them. The combination
`merge_enabled=True, reject_duplicates=True` is refused in `__post_init__`: filtering is an
alternative to pooling, not a modifier of it.

### 5.1 The four arms

| arm | equivalence relation | on a match | tree shape |
|---|---|---|---|
| `baseline` | none (τ=0) | — | tree |
| `treatment` | τ on projected value | pool statistics into the existing node | DAG |
| `reject` | same | discard the candidate; **answers exempt** | tree |
| `reject_all` | same | discard the candidate, answers included | tree |

`baseline` vs `reject` tests H3. `reject` vs `treatment` tests H3′ — and is the comparison
this revision exists to make. `reject` vs `reject_all` measures something specific and
worth its own line: **how much the equivalence relation costs when it is allowed to overrule
a verifier.**

That last distinction is the one design decision here that is not a free choice. A candidate
that reaches an *answer* can be checked against ground truth for one verifier call, on
generation already paid for. Filtering it on an approximate projected-value match instead
discards a possibly-correct solution on the strength of the single signal this project has
repeatedly found unreliable. Pooling cannot avoid this — the node is shared, so the
newcomer's own correctness is discarded either way — but filtering can, and the default does.
`reject_all` exists so the cost of *not* exempting them is measured rather than assumed.

### 5.2 Setup

- Step-level PUCT over discrete reasoning steps, node = token prefix, exactly as the existing
  H3 harness does it. Value from outcome reward where ground truth exists; PRM secondary.
- Equivalence: the frozen learned projection already in use (`projection.py`), thresholded at
  the calibrated τ, compared only within the same `(depth, kind)` bucket. An off-the-shelf
  sentence encoder is the alternative relation to sweep against it — the point of Phase 1 is
  that the relation is *cheap and available*, not that it is the right one.
- Sweep τ. **τ=0 (relation disabled) must reproduce `baseline` exactly** — the harness
  neutrality control, and it is a unit test, not just a run
  (`test_reject_duplicates_is_inert_when_tau_is_zero`).

### 5.3 Measure

- Accuracy @ matched node-expansion budget — the primary comparison, *not* wall-clock
- Accuracy @ matched wall-clock — the honest secondary. Note the cost profiles differ by arm
  and in opposite directions: pooling saves the expansion but pays for the equivalence test;
  filtering pays for both the test and the discarded candidate's generation. On the ARC
  domain the filter cost up to +37% wall-clock for a node count barely below the tree's, so
  this column can reverse the node-budget verdict and must be reported alongside it.
- Duplicate rate by depth (`graph.merge_hits` / `graph.rejected`) → tests H2
- Effective branching factor, unique states per unit compute
- **Multi-parent fraction**, which is 0 by construction in the filtering arms. Report it: it
  is the property that makes them immune to the compounding failure mode, and it should be
  visible in the output rather than asserted in prose.
- Over-merge audit: hand-label 100 matched pairs for genuine equivalence. Cheaper to interpret
  than in v0.1, because under filtering a mislabelled pair costs one candidate rather than a
  corrupted subtree.

**Gate:** `reject` beats `baseline` at matched node budget on ≥2 datasets, with the win
surviving wall-clock accounting at some τ. `treatment` is *not* required to beat `baseline`
for the phase to pass — that is now H3′, reported either way.

### 5.4 What this retires

- **No representative-parent choice**, so **no KV-cache invalidation.** v0.1's "note on cache
  invalidation" and Concern C3 simply do not arise: nothing is merged, so no branch loses its
  prefix sharing. The filtering arms keep every node's own cache.
- **No visit-count pooling rule.** §7.4's open problem ("naive summing over-weights
  early-merged nodes") does not exist here.
- **No cycles.** Aggregation is what introduces edges a tree cannot have; filtering adds
  none, so C8 is retired and the assertion-based cycle tripwire is unnecessary rather than
  merely satisfied.

### 5.5 Result — the gate is met on both datasets

Two datasets at their published H3 settings, four arms each, all arms seeded per
`(held_out_seed, problem_id, arm)` so they are paired on identical LM draws. Write-ups:
`results/PROSQA_ADMISSION_FILTER_FINDINGS.md`, `results/COUNTDOWN_ADMISSION_FILTER_FINDINGS.md`.

| arm | ProsQA (n=21) | Countdown (n=19) | pooled (n=40) | multi-parent fraction |
|---|---|---|---|---|
| `baseline` | 5/21 | 5/19 | 10/40 (25.0%) | — |
| `treatment` (pool) | 18/21 | 11/19 | 29/40 (72.5%) | 0.085 |
| **`reject`** | **19/21** | **16/19** | **35/40 (87.5%)** | **0.000** |
| `reject_all` | 18/21 | 11/19 | 29/40 (72.5%) | 0.000 |

| comparison | ProsQA | Countdown | pooled |
|---|---|---|---|
| `reject` − `baseline` | **+0.667, p=0.0001** | **+0.579, p=0.0074** | **+0.625, p<0.0001** |
| `treatment` − `baseline` | +0.619, p=0.0002 | +0.316, p=0.1796 (n.s.) | +0.475, p=0.0003 |
| `reject` − `treatment` | +0.048 (1 vs 2) | +0.263 (0 vs 5) | +0.150, CI [+0.025, +0.275] (1 vs 7) |
| `reject_all` − `reject` | −0.048 (0 vs 1) | −0.263 (6 vs 1) | −0.150, CI [−0.275, −0.025] (7 vs 1) |

**H3 passes on both datasets**, significantly, and the win survives wall-clock accounting on both
(the filter is 35–42% *cheaper* per problem than the baseline, because a solved search stops
spending expansions). That is the gate.

**H3′ fails, and on Countdown it fails hard.** Pooling does not add anything on top of filtering;
pooled across datasets the filter is *ahead* by +0.150 with a CI excluding zero and a 1-vs-7
one-sided discordance. On Countdown alone, pooling does not even clear the baseline
significantly (p=0.18) while filtering does (p=0.0074) — a dissociation between the two
mechanisms, not a tie. The measured multi-parent fraction is 0.085 for pooling and **exactly
0.000** for both filtering arms, so the arm that never let a node hold statistics from two paths
is the arm that solved more.

This is the classical-domain mechanism claim reproduced on the LLM harness under a *learned,
approximate* equivalence relation rather than an exact one.

**The answer-exemption is load-bearing, not a detail.** Filtering answer candidates too
(`reject_all`) costs −0.150 pooled, with 7-vs-1 discordance, and on Countdown it costs 5 problems
and lands *below* pooling. An answer can be checked against ground truth for one verifier call on
generation already paid for; discarding it because its projected value sits within τ of another
answer throws away a checkable solution on the least reliable signal in the system. Pooling cannot
make this distinction at all — the node is shared, so the newcomer's own correctness is discarded
either way. The two datasets bracket the size: 1 problem where the equivalence relation is decent
(ProsQA), 5 where it is not (Countdown).

**Caveat on the H3′ p-values.** `reject` − `treatment` is p=0.0703 pooled and p=0.0625 on
Countdown. Both are the floor McNemar's exact test can return for a fully one-sided discordance of
that size, so the constraint is n, not the effect; the CIs exclude zero in both cases. "The filter
beats pooling" is justified by the one-sidedness. "p<0.05" would not be.

**τ sweep (ProsQA, three points across 10×).** τ=0.01 was calibrated for the pooling arm, so the
worry was a knife-edge:

| τ | `baseline` | `treatment` | `reject` | `reject` − `baseline` | `reject` − `treatment` |
|---|---|---|---|---|---|
| 0.003 | 5/21 | 18/21 | **19/21** | +0.667, p=0.0001 | +0.048 |
| 0.01 | 5/21 | 18/21 | **19/21** | +0.667, p=0.0001 | +0.048 |
| 0.03 | 5/21 | 17/21 | **19/21** | +0.667, p=0.0001 | +0.095 |

The filter does not move — 19/21 at every τ — while pooling slips at the widest, so the gap
*widens* with τ. That is the predicted asymmetry appearing as a robustness property: as τ admits
more false matches, pooling pays with corrupted estimates and the filter pays one candidate each
time.

Read honestly, though, this shows **insensitivity rather than robustness**: match counts barely
change across the 10× range (338/340/331 rejections), because ProsQA's projected values are
crisply separated (precision 0.993 against ground-truth state identity —
`PROSQA_GROUND_TRUTH_TRANSPOSITION_FINDINGS.md`). A τ wide enough to force real false matches was
not reached. Countdown's `reject_all` result is the evidence that such a regime exists and matters,
and testing the filter there is the top open item.

---

## 6. Phase 2 — Learned aggregation metric

**Purpose:** replace off-the-shelf similarity with something bisimulation-flavored: distance
that predicts *outcome divergence*, not surface similarity.

**Objective.** Contrastive/regression hybrid on state pairs harvested in Phase 0:
- pull together states whose rollout answer-distributions agree (low TV distance)
- push apart states whose distributions diverge
- auxiliary head predicting V̂(s) directly, to keep the embedding value-aware

Train φ as a light MLP head on frozen hidden states first. Only fine-tune the backbone if the
frozen version shows signal — backbone tuning risks degrading the reasoning ability we're
searching over.

**Validation (must precede any use in search)**
- Held-out false-merge rate at the operating threshold, vs. Phase 0 raw-distance baseline
- Calibration: does predicted-equivalence probability match observed answer agreement?
- **H5 test:** stratify by an entropy/superposition proxy (next-token entropy at the snapshot,
  or spread of rollout answers). Report false-merge rate for high- vs low-entropy states
  separately. Expect the metric to be materially worse on high-entropy states.

**Gate:** false-merge rate at least 2× better than raw cosine, and — critically — not
catastrophically worse on the high-entropy stratum, since those are the states search
actually spends its time on.

---

## 7. Phase 3 — Latent search (graph optional)

Only if 0–2 pass. Three sub-problems, each independently able to sink the phase.

"MCGS" in v0.1 assumed the graph. Under v0.2 the graph is the H3′ arm: the default latent
search is a *tree* with an admission filter, and §7.4's DAG machinery is built only if H3′
earns it. That removes §7.1's state-definition problem from the critical path too — see
below.

### 7.1 State definition

The last hidden vector is **not** the transformer's state; the full KV cache is. Merging on the
final vector merges a lossy projection and silently discards the difference, leaving the
transition function ill-defined on the merged node.

**This is an H3′ problem, not an H3 problem.** A filter never has to continue generation from
a merged node, so the transition function is never ill-defined: the surviving candidate keeps
its own cache and the discarded one is simply gone. The projection only has to be good enough
to *decide* equivalence, not to *represent* the merged state. The three options below are
therefore required only if pooling is pursued.

Options, in order of preference:
1. **KaVa-style compressed cache as the state.** Merge on a learned compression of the cache;
   representative cache is then a well-defined object in that space. Most principled, most work.
2. **Merge on φ(h), keep both caches, run transitions from the higher-value parent.** Pragmatic;
   introduces a documented approximation. Requires an ablation measuring how much the choice of
   representative parent changes outcomes.
3. **Cache averaging.** Cheap, almost certainly wrong (attention is not linear in the cache).
   Try it only to quantify how wrong.

Decide this before writing the search loop; it determines the node data structure.

### 7.2 Branching / proposal distribution

Merging solves identity, not action enumeration. ℝ^d has no natural finite action set.
Candidates, cheapest first:
- **Hybrid (recommended start).** Branch over top-k discrete tokens to commit a direction,
  then continue in latent space. Keeps a finite, interpretable branching factor; sacrifices
  purity for tractability.
- **Noise injection** on the fed-back hidden state, σ learned or annealed. Simple; diversity
  is unstructured and may leave the manifold.
- **Stochastic latent head** — a small proposal network emitting a distribution over next
  thought vectors, trained to match the diversity of explicit-CoT continuations.

**Required diagnostic for whichever is chosen:** decode proposed latent steps back to text
(nearest-token or a trained probe) and check validity + diversity by hand on a sample. A
proposal distribution that generates off-manifold garbage will look like healthy exploration
to the search and produce silent quality collapse.

### 7.3 Value estimation

Cheapest credible path: value head on the residual stream, trained on outcome labels from
Phase 0 rollouts. This is the objection I would *not* lead with — a value head on hidden
states is natural and data is the only real obstacle. Options if it underperforms:
- Decode the latent state to text at leaves only, score with an existing PRM (expensive but
  reuses trusted machinery, and gives a calibration reference for the latent head)
- Truncated rollouts to outcome reward, no learned critic (highest variance, zero reward-hacking)

Track value-head calibration continuously. A miscalibrated critic inside a merging search
compounds: bad values corrupt aggregated statistics, which corrupt selection, which changes
the state distribution the critic sees.

### 7.4 Search loop

Standard PUCT over a DAG. Implementation notes that matter:
- Log the full graph for every solved problem — post-hoc graph analysis is where the
  interesting findings will come from. One such analysis is already done and negative: DAG
  in-degree predicts solution-path membership at 0.84–0.87 discrimination on its own but
  **at chance once conditioned on visit count**, so the graph structure carries nothing UCB1
  does not already have (`ARTICLE_DERIVED_MERGING_FINDINGS.md`, item E). §13's open question
  about heavily-merged regions marking reasoning bottlenecks is answered: not usefully.

The following are required **only for the H3′ (pooling) arm** — the filtering arm is a tree
and needs none of them, which is most of why it is the cheaper thing to build:

- Backup along *all* paths to a merged node, not one — the point of the DAG
- Guard against cycles introduced by aggregation (merging can create loops that pure trees
  cannot have); detect and break them
- Visit-count pooling on merge must be principled; naive summing over-weights early-merged nodes

---

## 8. Phase 4 — Closing the distillation loop

The point of the search is to be an improvement operator for expert iteration:
search → harvest improved traces → distill into weights → search again from a stronger base.

**Plan**
- Harvest (input, answer) pairs from successful searches; self-consistency as unsupervised filter
- Distill both ways and compare: token-CoT targets vs. latent targets (CODI-style, explicit
  run supervising latent run)
- Two iterations minimum — one round tells you nothing about whether the loop compounds

**Watch for:** diversity collapse across iterations. Expert iteration is known to narrow the
policy; if round 2's search finds less than round 1's despite a stronger base, the loop is
eating itself. Track entropy of sampled solutions and pass@k (not just pass@1) every round.

---

## 9. Metrics and protocol

**Primary**
- Accuracy @ matched node-expansion budget
- Accuracy @ matched wall-clock and matched FLOPs (report all three; they will disagree, and
  the disagreement is the story)

**Diagnostic**
- False-match rate (held-out, and stratified by state entropy)
- Duplicate/match rate by depth (`merge_hits` in the pooling arm, `rejected` in the filtering
  arms)
- **Multi-parent fraction** — 0 by construction in the filtering arms, and reported rather
  than assumed, since it is the property that rules out the compounding failure mode
- Effective branching factor; unique states per unit compute
- Value-head calibration (reliability diagram, ECE)
- Solution diversity / pass@k across distillation rounds

**Datasets.** GSM8K (saturating — use for plumbing, not claims), MATH-500, AIME-style held-out
set, plus one non-math domain (e.g. a code or logic benchmark) to test whether merging
generalizes beyond arithmetic's unusually clean step structure.

**Hygiene**
- ≥3 seeds on every headline number, report variance; single-seed search results are noise
- Fixed decode params across arms
- τ=1.0 / merging-disabled control in every merging experiment
- Contamination check on any held-out set
- Pre-register the Phase 1 and Phase 3 primary comparison before running it

---

## 10. Concerns register

Ordered by severity. Each needs an owner and a designed test, not just acknowledgement.

### C1 — The metric isn't the one you want (**high → medium under filtering**)
L2/cosine in the residual stream is not task-relevant distance. The correct notion is
bisimulation-flavored and must be learned, which adds a component with its own failure modes.

v0.1's severity came from the second sentence of its own analysis: "over-merging biases value
estimates in a way that **compounds with depth**; under-merging just gives independent samples
with extra machinery — so the error is asymmetric." That asymmetry is a property of *pooling*,
not of the equivalence relation. Under filtering, an over-eager match costs one discarded
candidate and nothing propagates: the error stops being asymmetric and stops compounding.

Which is why the mitigation list is now short. **Mitigation:** conservative τ, and exempt
verifiable terminals (§5.1) so the relation can never overrule ground truth. Depth-dependent τ
and un-merge audits were mitigations for the compounding case and are no longer needed for the
filtering arms. *Test:* Phase 0 + Phase 2 false-merge rate, still, because C1 remains
load-bearing for H3′.

### C2 — Superposition works against aggregation (**high → medium under filtering**)
Aggregation bounds require value smoothness in the metric. A deliberately non-committal latent
state is exactly where smoothness fails: two mixtures with similar means can resolve to
different answers. **The property that makes latent reasoning attractive is the property that
makes its value landscape non-Lipschitz.**

v0.1 recorded "no clean mitigation" here, and for pooling that stands: averaging two superposed
states' values together is precisely the operation the non-Lipschitz landscape breaks.
Filtering never performs that operation. It can still make a *wrong call* on two superposed
states — declining to explore one that would have resolved differently — but the cost is one
unexplored branch at a known budget, not a corrupted estimate steering later selection.

So C2 downgrades from "may be fatal to the whole substrate" to "bounds how much the filter can
win". The v0.1 concession — "possibly exclude high-entropy states from merging, which shrinks
the benefit toward zero" — is also cheaper here: excluding high-entropy states from
*filtering* costs the budget saving on those states and nothing else.
*Test:* H5 stratified analysis in Phase 2, unchanged. It remains the deciding test for H3′.

### C3 — Cache economics (**high → retired for the filtering arms**)
Prefix sharing is most of what makes step-level search affordable, and *merging* breaks it: a
merged node has several parents and only one cache, so some branch loses its prefix sharing.

Filtering does not merge, so nothing loses its cache. C3 is retired for the H3 arms and
remains open only for H3′.

What replaces it is a smaller, different cost that must still be measured: a filtered
candidate has already been generated, so its generation is paid for and thrown away. On the
ARC domain that came to +37% wall-clock against a node count barely below the baseline's. So
the *test* survives even though the concern does not — **mandatory wall-clock and FLOPs
accounting in every phase, alongside the node-budget number** — and the mitigation shifts from
"compress the cache" to "make the equivalence test cheap, and test before generating where the
proposal distribution allows it".

### C4 — Auditability regression (**medium**, and a real cost)
Reasoning in tokens is reasoning you can inspect. Compiling deliberation into opaque vectors
trades transparency for speed. A *searched* latent trace is worse still: neither the steps nor
the search's reasons for preferring them are legible.
*Mitigation:* build latent→text decoding probes as first-class tooling in Phase 3, not an
afterthought. Treat "can we read what the search did" as a deliverable.

### C5 — The baseline may already be better (**medium**)
Benchmarked head-to-head on modern models, latent reasoning does not consistently beat plain
CoT; several careful studies find uneven gains, and Coconut-style approaches can underperform
zero-shot CoT on strong recent LLMs. We may be building search on a substrate that is behind
its discrete competitor.
*Mitigation:* Phase 1 is deliberately latent-free so there is a deliverable even if latent
reasoning does not carry its weight. Always report token-CoT and token-MCTS baselines.

### C6 — Value-head reward hacking (**medium**)
A learned critic inside a search is a reward model under adversarial optimization pressure —
the search will find its blind spots by construction.
*Mitigation:* outcome reward wherever ground truth exists; periodic re-grounding of the critic
against verified outcomes; monitor the gap between critic score and true outcome as search
depth grows.

### C7 — Off-manifold proposals (**medium**)
Noise-injected or head-generated latent steps may leave the region the model was trained on,
producing states whose rollouts are degenerate in ways the value head fails to notice.
*Mitigation:* decode-and-inspect diagnostic (§7.2); density/likelihood filter on proposals.

### C8 — Cycles and pathological graph structure (**low → retired for the filtering arms**)
Aggregation can introduce cycles impossible in a tree, breaking backup. Filtering introduces
no edges a tree cannot have, so there is nothing to detect. Open only for H3′.
*Mitigation (H3′ only):* cycle detection in the graph; unit tests on synthetic graphs.

### C9 — Scope creep (**medium**, process)
This plan touches state abstraction, latent reasoning, search, PRMs, and expert iteration.
Each is a project. The gates exist to prevent building Phase 3 because Phase 3 is interesting.
*Mitigation:* gates are hard. A failed gate ends the project or reduces it to writing up the
negative result.

---

## 11. Kill criteria

Written down in advance so they can't be renegotiated later:

- **Phase 0:** no layer/metric with ρ > 0.3 and false-merge rate < 15% → **stop.** Publish the
  value-geometry negative result; it's a genuine contribution about latent reasoning's search
  compatibility.
- **Phase 1:** semantic *filtering* does not beat prefix matching at matched node budget on ≥2
  datasets → **stop.** The equivalence relation doesn't pay even with a free metric, and it
  will not pay with a learned one. (Pooling failing is not a kill: that is H3′, and the
  classical domains already predict it.)
- **Phase 2:** learned metric no better than raw cosine, or catastrophically worse on
  high-entropy states → **stop at Phase 1's result** and write up C2 as the finding.
- **Phase 3:** latent MCGS underperforms token MCTS at matched FLOPs after a fair tuning budget
  → publish the comparison; do not proceed to Phase 4.
- **Phase 4:** round-2 search finds less than round-1 → loop does not compound; report and stop.

---

## 12. Sequencing and rough cost

| Phase | Work | Time | Compute | Gate |
|---|---|---|---|---|
| 0 | Value-geometry probe | 2–3 d | ~1 d × 1 node | ρ, false-merge rate |
| 1 | Token-space diversity enforcement | 1–2 wk | moderate, many search runs | accuracy @ node budget |
| 2 | Learned aggregation metric | 2–3 wk | light (frozen backbone) | false-merge rate, H5 |
| 3 | Latent MCGS | 6–10 wk | heavy | accuracy @ FLOPs |
| 4 | Distillation loop, 2 rounds | 4–6 wk | heavy (training + search) | compounding |

Phase 0 is an afternoon of GPU time to de-risk a month of work, and Phase 1 is where the
realistic deliverable lives. Treat everything from Phase 3 on as conditional.

Phase 1's estimate came down because the filtering arms need no DAG backup, no
representative-parent handling, no visit-count pooling rule and no cycle detection — and
because the arms themselves are already implemented and unit-tested in `search.py` /
`run_search_experiment.py --arms`. What is left for Phase 1 is running it and a second
dataset, not building it.

---

## 13. Open questions

- Is there a principled way to define aggregation on the KV cache rather than on a projection
  of it? (Blocking for §7.1 option 1.)
- Can the threshold be made *adaptive* — learned per-state from predicted value variance
  rather than a global τ? Cheaper to try under filtering: a per-state τ that is wrong costs
  candidates, not corrupted estimates. AB-MCTS's framing (treat width-vs-depth as a bandit
  decision) suggests the version worth building — per-node arms for "expand a new candidate"
  vs "accept a near-duplicate" — but it is gated behind H3 passing.
- ~~Does the DAG structure itself carry signal? A heavily-merged region may indicate a reasoning
  bottleneck worth surfacing to the distillation targets.~~ **Answered: no, not usefully.**
  In-degree discriminates on-solution-path nodes at 0.84–0.87 alone and at chance once
  conditioned on visit count, on two classical domains with ~670 solved graphs
  (`ARTICLE_DERIVED_MERGING_FINDINGS.md`, item E). It re-reads where the search already spent
  budget.
- Does merging help more on problems with many equivalent solution paths (combinatorics) than
  on linear-chain problems (arithmetic)? A per-problem-type breakdown may be the real result.
- If C2 holds, is there a *different* latent representation designed for search-compatibility —
  trained to be committal and smooth rather than superposed?
