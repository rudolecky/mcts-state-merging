# State Merging in MCTS Is Two Mechanisms, and Only One of Them Is the Risk

**Draft.** Everything below is reproducible from this repository; commands and commit hashes are in
§8. Numbers superseded during the work are marked as such rather than quietly replaced — the record
of what was withdrawn is part of the result.

---

## Abstract

Monte Carlo Tree Search implementations that detect transpositions bundle two distinct operations:
they **deduplicate** expansions (do not re-expand a state already in the tree) and they **pool**
statistics (average the returns of every path reaching a state into one shared node). Prior results
in this codebase — merging helps enormously on Blocksworld, is catastrophic on the Rubik's Cube,
is null under AlphaZero-style PUCT — all measured the bundle.

We separate them with a third search arm that deduplicates without pooling: compute the same
equivalence key, but *discard* a duplicate expansion rather than merging into the existing node, so
the tree stays a tree and no node ever holds statistics from more than one path.

Across five classical domains under two backup rules, and a step-level LLM reasoning search on two
datasets and two models, we find: **pooling is where the risk lives and the half that can fail
catastrophically; deduplication carries the benefit; and where pooling works, how you cash the
deduplication in does not matter.** On the Rubik's Cube under Monte-Carlo-average backup, where
pooling solves 0% at every budget, deduplication-without-pooling solves 98–100% within a
compute-matched budget while plain tree search reaches 0–53%. Everywhere pooling already works, the
two tie.

Two claims in earlier drafts of this work did not survive its own later tests, and the reasons
generalize past merging: a budget convention that silently stopped measuring what it was chosen to
measure, and a result that held on one model and not a second.

---

## 1. The question, and why the existing answer was confounded

The standing question in this codebase: *does merging improve search accuracy at matched compute,
and when does it not?* The accumulated answer was domain-indexed and unsatisfying — large win on
Blocksworld, catastrophic on the cube, null on Connect Four, null under PUCT — with the
explanation resting on the backup rule (`RUBIKS_CUBE_FINDINGS.md`) and on whether plain tree search
gets trapped without it.

That explanation is about a bundle. Merging does two things at once:

1. **Deduplication.** A state already in the tree is not expanded again, so budget goes to states
   the search has not seen.
2. **Pooling.** The node is shared, so every path reaching it contributes to one visit count and
   one value estimate.

These have different failure modes. Pooling makes a node's statistics a function of paths that did
not go through the parent currently being evaluated, and under an approximate equivalence relation
a false match corrupts an estimate that then steers all later selection — an error that compounds
with depth. Deduplication has no such channel: a wrong match costs one unexplored branch.

## 2. Method: a third arm

`reject_duplicates` computes the same equivalence key as the merge arm but, on a hit, **discards the
candidate and tries another move**. Node keys stay path-based. The three arms:

| arm | equivalence key | on a match | tree shape |
|---|---|---|---|
| `tree` | none | — | tree |
| `merge` | state / τ-threshold | pool statistics into the existing node | DAG |
| `reject` | same | discard the candidate | tree |

Implemented in five classical modules and the LLM harness. `merge_enabled` and `reject_duplicates`
together are refused: the filter is an alternative to pooling, not a modifier of it.

**Budget accounting.** This is load-bearing and we got it wrong first (§7.1). A rejected move never
becomes an expansion, but it costs a real state generation. `charge_rejections` counts rejections
against the expansion budget, and **every comparative number below is from a charged run** —
except ARC-AGI, whose module has no charge option (its reject arm ran ~12% over tree, so its null
is conservative). The charged runs were first produced by ad-hoc driver scripts; the runners
now take `--charge-rejections` (§9), and without it they reproduce the *uncharged* numbers.

## 3. Classical domains: exact equivalence

| domain | backup rule | pooling's state | `reject` − `merge`, charged |
|---|---|---|---|
| Rubik's Cube | UCB1 + MC-average | **catastrophic**, 0.000 everywhere | **+0.98 to +1.00**, p=0.0078 |
| Blocksworld | UCB1 + MC-average | works (large win over tree) | +0.007, +0.077 (p=0.58, 0.31) |
| Blocksworld | GUCT-Uniform | works | +0.000, +0.008, −0.042 (n.s.) |
| Cube / 8-puzzle / Sokoban | GUCT-Uniform | at parity with tree | null at every budget |
| ARC-AGI program synthesis | UCB1 + MC-average | null | null (uncharged) |

The one place deduplication-without-pooling buys something no other arm gets (oracle-guided,
c=0.02, 8 puzzles/tier, 15 repeats, every point verified inside budget):

> **Accounting note (2026-09-21).** In the "generations merge/reject" column below, the merge side
> is a *node* count and the reject side is `nodes + rejections`; neither is an `apply_move` count.
> A pooled expansion creates no node, so merge's compute is understated — by 2.27–2.45× on the cube
> rows, magnitude unmeasured on the Blocksworld rows, where the reject column being exactly
> budget+1 suggests the two arms are compute-matched rather than the filter being costlier. See
> `MERGING_MECHANISM_SUMMARY.md`.

| tier / budget | `tree` | `merge` | `reject` (charged) | generations merge / reject |
|---|---|---|---|---|
| td6 / 100 | 0.533 | 0.000 | **1.000** | 44 / 76 |
| td8 / 200 | 0.400 | 0.000 | **1.000** | 84 / 108 |
| td10 / 200 | 0.000 | 0.000 | **0.983** | 84 / 166 |
| td10 / 400 | 0.367 | 0.000 | **1.000** | 163 / 166 |

Pooling solves nothing here at any budget or difficulty. Two capping mitigations (limiting parents
per merged node, limiting visits) both failed as a cliff in earlier work. Not pooling fixes it
outright, and beats plain tree search as well.

**The backup-rule fix and the no-pooling fix are redundant.** GUCT-Uniform (Full-Bellman-minimum
backup, LCB1-Uniform selection) independently removes the same catastrophe, and applying the filter
on top of it adds nothing on any of four domains.

## 4. LLM reasoning search: learned, approximate equivalence

Step-level PUCT over reasoning states; equivalence is |Δ| < τ on a frozen Ridge projection of the
hidden state. Arms seeded per `(seed, problem, arm)` and paired by problem (each arm draws its own LM samples — the per-arm seed includes the arm's name; corrected 2026-10-02).

**Rejection is free in this setting**, which is the crucial asymmetry with §3: `expand_node` draws
all K candidates and computes every hidden state *before* the reject decision, so declining one
causes no extra generation and no extra forward pass.

| setting | n | `baseline` | `merge` | `reject` | `reject` − `baseline` | `reject` − `merge` |
|---|---|---|---|---|---|---|
| Qwen2.5-7B, ProsQA | 21 | 5 | 18 | **19** | +0.667, p=0.0001 | +0.048 (1v2) |
| Qwen2.5-7B, Countdown | 19 | 5 | 11 | **16** | +0.579, p=0.0074 | +0.263 (0v5) |
| Qwen2.5-1.5B, ProsQA | 42 | 11 | 21 | 20 | +0.214, p=0.0225 | −0.024 (5v4) |

**The equivalence relation pays, on both models, significantly.** That is the load-bearing claim and
it replicates.

**Neither mechanism beats the other.** One setting of three favours the filter; two are ties. An
earlier draft claimed "filtering beats pooling" from the first two settings; the second model
withdrew it (§7.2).

Multi-parent fraction is 0.085 for pooling and exactly 0.000 for the filtering arms — reported
rather than asserted, since it is the structural property that rules out the compounding failure
mode.

**Answer-exemption.** Filtering candidates that reach a *verifiable terminal* costs −0.150 pooled on
the 7B runs, and 5 of 19 problems on Countdown, where that variant lands below pooling. An answer
can be checked against ground truth for one verifier call on generation already paid for;
discarding it because its projected value sits within τ of another answer throws away a checkable
solution on the least reliable signal available. Pooling cannot make this distinction at all — the
node is shared, so the newcomer's own correctness is discarded either way.

## 5. Negative results

| tested | outcome |
|---|---|
| ARC's small merge effect is due to sparse binary reward | **Not supported.** Graded cell-level reward leaves merge diffs inside noise and makes the *baseline* worse at high budget (0.342 → 0.300). |
| A coarser, observational-equivalence merge key finds more useful equivalences | **Buys merges, not accuracy.** ~2× the merge rate, solve rate flat to slightly worse. |
| DAG in-degree predicts solution-path membership, so it is a free exploration prior | **Refuted.** 0.84–0.87 discrimination alone; 0.405 and 0.511 conditioned on visit count. Visit count fully explains it, and UCB1 already has that number. |
| Pooling yields lower-variance self-play training targets | **Predicted null** by the above — pooling is the half that does not pay, and PUCT creates no duplicates to deduplicate. Dropped rather than run. |

## 6. τ sensitivity

On ProsQA the filter is flat at 19/21 across τ ∈ {0.003, 0.01, 0.03} while pooling slips 18 → 17.
Read honestly this demonstrates **insensitivity, not robustness**: match counts barely move
(338/340/331 rejections across a 10× range) because that projection separates states crisply
(precision 0.993 against ground-truth state identity). A τ wide enough to force real false matches
was never reached on ProsQA. It has since been reached, on both Countdown (§10.2) and the cube
under a controlled relation (§10.3) — see `COUNTDOWN_WIDE_TAU_FINDINGS.md` and
`APPROXIMATE_RELATION_FINDINGS.md`.

## 7. What went wrong, and what it cost

We consider this the most transferable part. Four diagnostics silently stopped measuring what they
were chosen to measure. All four were invisible in the output, and all four were caught only by
adversarially re-checking the measurement rather than the hypothesis.

### 7.1 A budget convention that stopped applying

"Budget counts expansions" was chosen when every arm spent one state generation per expansion. The
filter can *decline* work, so its rejections were free and it ran at ~2× the state generations of
its comparators. Measured on the cube under GUCT-Uniform at budget 40 by counting `apply_move`
directly: merge spends 40.0, the uncharged filter 93.2 — 2.3×. (An earlier version of this sentence
gave 19.6 against 94.3, comparing merge's *node* count with the filter's `nodes + rejections`. Same
conclusion, and the ratio is unchanged, but it is the error this very section is about.)

We verified this once, on one domain, found the arm inside budget (80 + 92 = 172 < 200), and
generalized. **Cost: two committed claims retracted** — "the filter beats merging on Blocksworld"
and "the two fixes are not redundant." Both were entirely accounted for by the extra compute.

### 7.2 A result that held on one model

"Filtering beats pooling" rested on two settings with one model and one projection. A second model
tied. We had also predicted that the *weaker* projection (false-merge rate 0.24 vs 0.113) would hurt
pooling more, since a false match corrupts a pooled estimate while costing the filter one candidate.
Pooling did marginally better. **Training-set false-merge rate does not predict which mechanism
wins.**

### 7.3 Nondeterminism the size of the effect

`legal_moves` returned `list(<set of tuples containing strings>)`. Set iteration follows hash order
and Python randomizes string hashes per process, so every ARC search depended on which process it
ran in. Re-running a saved configuration from its own arguments and seeds disagreed on 2 of 12
per-task solve rates.

**Cost: every ARC number in the repository was process-dependent.** On republish, a merge effect of
+0.067 became +0.008, and a borderline merge *penalty* of −0.023 at p=0.058 became −0.000 at p=0.87.
The injected noise was the same size as the effects being measured.

### 7.4 Statistics that flattered the hypothesis

Three, all in one analysis: raw AUCs compared across features pointing in opposite directions (a
control at 0.28 discriminates as well as one at 0.72); quantile strata collapsing on a discrete
confounder, leaving the confounder inside each bin as apparent conditional signal (2 usable strata
of 10 requested); and a global permutation null destroying the within-graph clustering it was
introduced to respect, pinning p at its floor. The conclusion survived; every number moved.

### 7.5 Tests that could not fail

Three of ours: one reimplemented in its own body the logic it tested, so deleting the production
rule left it green; one compared a hardcoded dict against itself; one used a 9-key stand-in for a
30-key argument parser, so adding a key to an exclusion set by mistake passed the suite.

### 7.6 One bug, made twice

"This node produced nothing new" is not "this node leads nowhere." Marking a node exhausted when its
remaining moves were all rejected pruned live subtrees — dropping a 3-block Blocksworld solve rate
to 3/20 in one module, and 3/10 in a second module where we made the identical mistake independently
weeks later. Any arm that can decline an expansion needs this checked explicitly.

## 8. Limitations

- **Two models, two datasets** in §4; small n (21, 19, 42). The `reject` − `merge` p-values that are
  not ties sit at the floor McNemar's exact test can return for their one-sided discordances.
- The 1.5B run had no in-band calibration filter, so 24 of 42 problems are concordant across all
  arms and real power is well below n=42.
- τ was swept on only one dataset, in a regime where it barely binds.
- §3's cube result is oracle-guided (exact BFS distance). Under a learned or noisy value the picture
  is untested.
- The filter costs memory: one node per expansion, ~1.7× pooling's node count at matched budget.

## 9. Reproduction

Branch `merging-mechanism-split`, 13 commits, 520 tests. Findings documents are tracked in git
(`results/*.md`); pickles and checkpoint shards are not.

| result | entry point |
|---|---|
| Classical arms | `run_heuristic_merge_experiment_rubiks.py --reject --charge-rejections`, `run_classical_mcts_blocksworld_experiment.py --reject-duplicates --charge-rejections` |
| GUCT-Uniform crossing | `run_guct_uniform_blocksworld_experiment.py --reject-duplicates --charge-rejections`, `run_guct_uniform_rubiks_experiment.py --reject --charge-rejections`, `run_guct_uniform_generalization_experiment.py --reject --charge-rejections` |
| LLM search | `run_search_experiment.py --arms baseline treatment reject reject_all` |
| Analysis | `analyze_search_experiment.py --baseline-key X --treatment-key Y` |
| In-degree | `run_dag_indegree_analysis.py` |

Per-`(problem, arm, τ)` checkpointing makes the multi-hour LLM runs interruptible; the fingerprint
covers the projection file's contents and a content hash of the search code, not just argument
values.

## 10. What would change the conclusion

1. ~~**A third model.**~~ **Done; it exposed a budget leak, now fixed** (`THIRD_MODEL_BUDGET_LEAK_FINDINGS.md`,
   `LEAK_FIX_RERUN_FINDINGS.md`). Deduplicating searches whose frontier empties idled with budget
   unspent. Fixed opt-in (`widen_on_starvation`) and both 1.5B settings re-run: on ProsQA the fix
   recovered four problems for pooling (pool − baseline +0.333, p=0.0013; filter unchanged at +0.214);
   on Countdown-3 it changed nothing and both arms trail the baseline (n.s.). A random-rejection
   control at the filter's rate loses to the filter 2 vs 11 discordant (p=0.022), so the relation
   is deduplicating. Pool vs filter shows no ordering. Both 7B settings re-run: unchanged problem
   for problem, so the 7B Countdown filter win stands; filter beats random rejection 20 vs 3
   (p=0.0005) across all four settings.
2. ~~**A τ wide enough to force real false matches**, on a domain where distances actually
   spread~~. **Done.** `results/COUNTDOWN_WIDE_TAU_FINDINGS.md`: six τ points, published to 60×.
   The filter never fell below baseline (floors at 4/10 vs 2/10). The widest two points are
   byte-identical because search-time values clip to [0,1]: τ ≥ 1 is the coarsest relation
   possible, the filter's tree became a single chain (one step node per depth, every problem), and
   it still beat baseline. A cube sweep over proposal quality explains why: the filter's floor — and
   pooling's cube catastrophe — both need uniformly random proposals; with a moderate proposal
   policy the saturated filter solves 0.975 (`PROPOSAL_QUALITY_FINDINGS.md`). An apparent
   `reject − treatment` sign flip past τ≈0.36 at n=10 was chased with seeds 71/72 to n=19 and did
   **not** survive: filter and pool tie exactly at τ=0.36, and the filter significantly beats
   baseline at τ=0.17 (p=0.0215).
3. ~~**A domain where pooling is broken *and* the equivalence relation is approximate.** Every
   setting where pooling failed catastrophically had an exact relation; every setting with an
   approximate relation had working pooling. The cell that would most sharpen the mechanism claim
   is empty.~~ **Done.** `results/APPROXIMATE_RELATION_FINDINGS.md` (cube, UCB1+MC-average, oracle
   relation + calibrated noise): pooling is 0.000 at every relation quality; the filter degrades
   gracefully from a measured false-merge rate of 0.5 up to ~0.8, then **crosses below plain tree
   search** past τ≈1.3–1.9 (rising with budget) and is significantly worse by τ=4. So the repair
   does not need an exact relation, but it is not unconditionally safe either — a bad enough
   relation makes the filter worse than doing nothing, which item 2 above then failed to reproduce
   on Countdown at LLM-harness budget/K.
