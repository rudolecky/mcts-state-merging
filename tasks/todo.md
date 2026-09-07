# Article-derived merging improvements

Source: ARC-AGI/MCTS landscape review. Items lettered as in the review reply.

## Plan

- [x] **0. ARC goal/merge-key inconsistency (defect).** `canonical_key` is order-independent,
      `is_goal` reads `ctx[-1]` -> merge arm can cache `is_terminal=False` on a node a goal
      ordering later merges into. Fix: goal-ness must be a function of the merge key
      (any-index match). Verify: regression test that two same-key states agree on `is_goal`,
      plus a search-level test.
- [x] **A. Graded ARC reward.** Binary exact-match reward is the sparsest in the repo and ARC
      has the smallest merge effect; pooled statistics pool zeros. Add per-example graded
      progress (exact -> 1.0, same-shape grid -> matching-cell fraction, else 0.0), meaned over
      examples, maxed over context indices. Verify: unit tests on the scorer, `--reward-mode`
      arm in the runner.
- [x] **B. Observational-equivalence merge key.** Current key is (tag, values) pairs; the same
      concrete value reached under different static tags makes two observationally identical
      states distinct. Add `--merge-key-mode values` (tag-free). Verify: merge rate rises,
      false-merge audited against ground truth.
- [x] **C. Third arm: reject-duplicates (admission filter).** Compute the merge key but discard
      a duplicate child instead of pooling into it. Gets distinctness with zero backup-rule
      exposure. Domains: ARC, Blocksworld (merge = big win), Rubik's (merge = catastrophe).
      Verify: unit tests per domain + real runs on all three.
- [x] **E. DAG in-degree as signal.** Does merge in-degree predict on-solution-path membership?
      Post-hoc on fresh cheap Blocksworld/Sokoban merge runs. Verify: AUC + permutation null.
- [x] **G. Wall-clock accounting.** Node savings != compute savings; `canonical_key` is O(state).
      Report per-arm elapsed alongside node counts.
- [x] **D. Self-play target quality (code only).** Thread self-play merge on/off through
      `run_selfplay_train.py` so tree- vs DAG-derived training targets can be compared at
      matched self-play budget. Full two-arm training run is heavy -> left for the user to launch.
- [ ] F. Bandit tau. Explicitly gated behind C's result. Not implemented.

## Review

Full results: `results/ARTICLE_DERIVED_MERGING_FINDINGS.md`. Test suite: 473 passed.

**What paid off.** Only item C, and it paid off large. Splitting merging into deduplication and
statistic-pooling and keeping only the first:

* Rubik's Cube: 1.000 at every budget/difficulty, where merging is 0.000 and tree needs 2-4x the
  budget to catch up. Independently verified against the BFS table -- genuinely solved cubes at
  optimal depth 10, and the win survives charging rejections to the budget.
* Blocksworld: beats *merging* at every budget (0.370 vs 0.158 at budget 200; 1.000 vs 0.602 at
  1000), on the domain where merging is this project's biggest win.
* ARC: null, like everything else there.
* Costs up to +37% wall-clock on ARC (a rejected move still pays a full `apply_move`), against a
  node count barely below the tree's -- the node column and the compute column are nearly unrelated
  on that domain.

**What did not.**
* Item A (graded ARC reward): merge diffs turn uniformly positive under graded reward but every one
  is inside the noise, and the *baseline* gets worse at high budget (tree 0.342 binary -> 0.300
  graded at budget 5000). Cell-level grid overlap is a misleading gradient in program space. Reward
  density does not explain ARC's small merge effect.
* Item B (observational-equivalence key): roughly doubles the merge rate (782 vs 449 merge hits at
  budget 5000), solve rate statistically unchanged. Real equivalences, not useful ones.
* Item E (DAG in-degree): AUC 0.83-0.86 unconditionally, **0.47-0.48 conditioned on visit count**.
  Not a usable prior -- it is a re-reading of where the search already spent budget.
* Item 0 (goal/merge-key defect): real, fixed, and worth zero on the classical ARC arm's 12 tasks
  (identical results with the old and fixed `is_goal`). It does touch one published GUCT-Uniform
  ARC number: the budget-500 borderline negative (-0.023, p=0.058) becomes a flat null (+0.003,
  p=0.570). That is a fresh sample, not a clean A/B -- changing which states are terminal changes
  RNG consumption -- so it reads as "the marginal negative does not replicate under corrected goal
  detection".

**A pre-existing bug found while chasing item 0, and the most consequential thing here:** ARC
searches were **not reproducible across processes**. `legal_moves` returned
`list(<set of tuples containing strings>)`, so move order followed randomized string hashes; the
same seed gave different `rng.choice`/`rng.shuffle` results per process. Re-running a saved config
from its own args and seeds disagreed on 2 of 12 per-task solve rates. Fixed with
`sorted(moves, key=repr)`, pinned by a cross-subprocess test. **Every ARC number in the repo
predating this is process-dependent, `ARC_AGI_FINDINGS.md` included.** It also explains the
published-vs-rerun drift I initially misattributed to my own `is_goal` change.

**Corrections to my own claims, from the review:**
* Item 0's defect is **unreachable**, not merely unobserved: a non-final-goal state's own prefix is
  already terminal and so never expanded. Kept as an invariant repair, re-documented honestly.
* The GUCT-ARC heuristic had the same inconsistency and it *is* reachable (h-driven search, no
  rollouts). Fixed -- and it lifts both arms equally, leaving the merge-vs-tree diff identical
  (-0.000 and +0.017 before and after). Also inert for the comparison.
* Item E's conditional AUC needed three statistical fixes before it meant anything: orientation-free
  discrimination (depth points the other way, so raw AUCs are not comparable), frequency-based
  strata (quantile bins collapsed to 2 of 10 on visit counts, leaving the confound inside each bin),
  and a within-graph permutation null (the global shuffle destroyed the clustering it was added to
  respect and pinned p at the floor). The conclusion survived; the numbers moved.

**Three bugs I introduced and then fixed, all caught by running things rather than reading them:**
1. First version of `_mark_exhausted` marked any node whose remaining untried moves were all
   rejected. But `select()` returns nodes that have BOTH untried moves and existing children, so
   this orphaned live subtrees -- Blocksworld's 3-block reject solve rate fell to 3/20. Fixed by
   requiring no live children first (20/20 after).
2. The any-index `is_goal` made the per-rollout-step goal check O(context length) instead of O(1),
   slowing the ARC experiment ~5x. Fixed with `last_value_is_goal`/`last_value_progress`, exactly
   equivalent inside a rollout since each step appends one value.
3. `merge_enabled=True, reject_duplicates=True` silently meant three different things across the
   three modules. Now refused in `__post_init__`.

**Not run.** Item D's `--selfplay-merge/--no-selfplay-merge` switch is in place; the two-arm
training comparison is a real GPU/CPU spend and was left for the user to launch. Item F stays gated
behind C, as the review itself specified.

**Best follow-up this opens:** the reject arm has never been crossed with GUCT-Uniform's
Full-Bellman-minimum backup. The backup-rule swap and the no-pooling arm now look like two fixes
for the same underlying problem, and nobody has run them together.

---

# Phase 1 redesign: admission filtering instead of pooling

Follow-on from the finding that merging is two mechanisms and only pooling carries the risk.
Redesigns `latent-mcts-merging-plan.md` Phase 1 around an admission filter, and implements it.

## Plan

- [x] **Plan document (v0.2).** Rewrite thesis + outcomes (B becomes "semantic diversity
      enforcement"; the old B becomes B′, a bonus arm); split H3 into H3 (deduplication pays)
      and H3′ (pooling adds something); rewrite §5 Phase 1 around the four arms; mark §7's DAG
      machinery and §7.1's state-definition problem as H3′-only; downgrade C1/C2 from high to
      medium and retire C3/C8 for the filtering arms, with the reason in each case; update the
      Phase 1 kill criterion and cost; answer two §13 open questions from the in-degree result.
- [x] **Implement the arm in the LLM harness.** `SearchConfig.reject_duplicates` +
      `reject_answer_nodes`, `__post_init__` refusing merge+reject, graph counters,
      `expand_node` filter branch. `run_search_experiment.py --arms
      {baseline,treatment,reject,reject_all}`; `analyze_search_experiment.py
      --baseline-key/--treatment-key`.
- [x] **Answer candidates exempt by default.** An answer can be *verified*; filtering it on an
      approximate projected-value match discards a possibly-correct solution on the weakest
      available signal. `reject_all` is the symmetric-with-merge variant, so the cost of not
      exempting them is measured rather than assumed.
- [x] **Run it.** Done, 3h15m, four arms on ProsQA at the published H3 settings. Results in
      `results/PROSQA_ADMISSION_FILTER_FINDINGS.md`:

      | arm | solved | delta vs baseline | p |
      |---|---|---|---|
      | baseline | 5/21 | — | — |
      | treatment (pool) | 18/21 | +0.619 | 0.0002 |
      | **reject** | **19/21** | **+0.667** | **0.0001** |
      | reject_all | 18/21 | +0.619 | 0.0002 |

      **H3 passes** (filter beats prefix matching, 14 discordant problems to 0). **H3′ fails**
      (`reject` − `treatment` = +0.048, p=1.0) — pooling adds nothing detectable, so merging's
      whole benefit on this domain is deduplication. Multi-parent fraction 0.000 in both
      filtering arms, so the compounding failure mode never had the chance to occur. Both
      treatments are also ~35% *cheaper* than the baseline, because a solved search stops
      spending expansions.

- [x] **Per-problem checkpointing**, added before the run after two interruptions discarded 50
      and 90 minutes. Shards are keyed per *(problem, arm)* rather than per problem, because
      per-arm seeding makes each arm independent of which others ran — so extending a finished
      three-arm run costs only the fourth arm. Atomic writes (temp + rename), a truncated shard
      reads back as absent and is recomputed, and a checkpoint from different search arguments is
      refused rather than mixed in. 12 unit tests.

- [x] **Second dataset (Countdown-4) — the gate is now met.** Three published pools (seeds
      70/71/72), n=19: baseline 5, pooling 11, **reject 16**, reject_all 11.
      `reject` − `baseline` = **+0.579, p=0.0074**. `reject` − `treatment` = **+0.263, 0-vs-5
      discordant** — the filter beats pooling here rather than tying it, and pooling does not
      clear the baseline significantly at all on this dataset (p=0.18). Write-up:
      `results/COUNTDOWN_ADMISSION_FILTER_FINDINGS.md`.

      **Both datasets pooled (n=40):** baseline 10, pooling 29, **reject 35 (87.5%)**.
      `reject` − `baseline` +0.625 p<0.0001; `reject` − `treatment` +0.150, CI [+0.025, +0.275],
      1-vs-7. Measured multi-parent fraction 0.085 for pooling, **0.000** for both filtering arms.

      **The answer-exemption turned out to be load-bearing, not a detail:** −0.150 pooled to
      filter answers too, worth 1 problem on ProsQA and 5 on Countdown, where `reject_all` lands
      *below* pooling. A design decision made on principle before the data existed.

- [x] **tau sweep (ProsQA, 3 points across 10x).** `reject` is flat at 19/21 for
      tau ∈ {0.003, 0.01, 0.03}, +0.667 at p=0.0001 throughout; pooling slips 18 → 17 at the
      widest, so the gap widens with tau. That is the predicted asymmetry as a robustness
      property. **But read honestly it shows insensitivity, not robustness** — match counts
      barely move (338/340/331), because ProsQA's distances are crisply separated. tau=0.1 was
      not reached (stopped); its 3 shards remain in the checkpoint.

- [ ] **Top open item: a tau wide enough to force real false matches, on a domain where
      distances spread.** Countdown is the candidate — its `reject_all` result is direct evidence
      that false matches there cost real problems. Then a third dataset (both current ones are
      small and single-model), and more n to move the `reject` − `treatment` p-values, which sit
      at the floor McNemar can return for their one-sided discordances.

## Review outcomes (second review agent)

**One high-severity bug of mine, found by review and confirmed by measurement.** An expansion
that admits nothing left its node `has_been_expanded=True` with no children — a state the
pre-existing arms cannot reach. `select_leaf` needs `node.children` to descend, so it kept
returning that node and `run_search` kept re-expanding it: K fresh LLM generations, filtered
again for the same reason. Measured at 28–58% of the arm's expansion budget spent producing
nothing, and invisible in every recorded statistic. **The first launched run was killed and
discarded because of this.** Fixed with a `no_admissible_children` flag that selection skips
and `run_search` propagates lazily up the tree — backing up the node's own `guide_value`, not
0.0, since 0.0 would tell the parent a merely-duplicate continuation is worthless and
reintroduce exactly the value corruption this arm exists to avoid.

**Two more real ones:**
* The filter dropped the duplicate's *prior share* while merge accumulates it, so
  `treatment - reject` was measuring pooling **plus** a changed exploration weighting — a 1.2
  absolute gap in a PUCT term whose Q component spans [0,1]. Now a rejected duplicate credits
  its prior to the surviving sibling when that sibling is already this parent's child: shares
  the policy signal, not `n_visits`/`w_value`, which is the split being tested.
* `depth_index` is populated at registration and `kind` is later mutated to `"exhausted"`
  without reindexing, so dead ends stayed live match targets. Harmless-ish under merge (you get
  an edge to the dead end); under filtering it destroys a candidate in favour of a state already
  proven to go nowhere. Now skipped.

**Also fixed:** no per-arm seeding (arms were consuming one RNG stream in order, so adding an
arm changed every later arm's draws and no arm could be re-run alone — now seeded per
`(held-out seed, problem, arm)`, which additionally pairs the arms on the same LM draws, as the
McNemar test already assumed); `expansions_used`/`llm_samples` now recorded, because
"matched budget" is not delivered whenever `run_search` exits on its iteration cap instead;
`--arms` rejected duplicates; the analysis script's saturation heuristic was measuring the
treatment working rather than saturation.

**Two of my new tests were weak and the review said so.** The tau=0 neutrality test asserted
only node counts, so it would have passed under any divergence preserving them — now a full
graph-snapshot comparison, plus a companion where the relation is live but declines. And the
test that constructed the invariant-violating state asserted it as *desired* with no coverage
of what the harness then did with it — replaced by a direct per-node "expanded at most once"
invariant across all four arms, which is precisely what the leak violated. Two fixtures I
wrote to expose the leak end-to-end did not actually exercise it (they terminated for
unrelated reasons); rather than ship tests that pass vacuously, I dropped them and pinned the
invariant instead.


## Checkpoint infrastructure review (third review agent)

**No high-severity findings, and nothing that invalidates a reported result.** The reviewer
verified empirically (not by inspection) that the load-bearing assumption holds — the `baseline`
arm's result really is tau-independent, so sharing one baseline shard across a sweep is correct —
and that the fingerprint's four exclusions are exactly the keys that cannot change a search.

Fixed from its findings:

* **The fingerprint covered argument *values* but not the files they name or the code that reads
  them.** `--projection-data` at a stable path could be regenerated, moving every node's
  `proj_value` — and since `guide_value` defaults to it, that changes the *baseline* arm too, not
  only the arms that match. Now folds in a content hash of the projection file and of
  `search.py`/`model.py`/`projection.py`. Deliberately content-based rather than git HEAD + dirty
  flag: this tree runs dirty for days, so the git version is the same string across every edit and
  would not discriminate the changes that matter.
* **The assembled filename distinguished tau but not the arm set** — so "add a fourth arm cheaply",
  which the runner's own output recommends, would have replaced a four-arm file with a one-arm
  file. Both keys are now in `results_filename()`, which `main()` calls and the test calls.
* **`load_arm` swallowed every exception silently.** A moved class or a permissions problem would
  make all shards unreadable, print nothing, and read to the user as "no checkpoint" — inviting
  deletion of hours of recoverable work. Now names the reason. Rows also carry a schema version, so
  a stale-schema shard is recomputed instead of raising a KeyError *after* the results file is
  written.
* Temp filenames are process-unique (a parallel sweep sharing one output dir would otherwise
  interleave bytes and silently lose the shared-baseline saving); the resume count uses `exists()`
  instead of unpickling every graph; a fingerprint mismatch now names the differing keys instead of
  leading with "delete it".
* **The analyser now states that reused shards carry the original run's timings** — in a tau sweep
  the baseline's wall-clock was measured once, so cross-run time comparisons are indicative only.
  This was already disclosed in prose in the ProsQA write-up; now it prints next to the number.
* `--tau-percentile` is dead (never read; `calibrate_tau` hardcodes its own percentiles). Documented
  as such rather than left with help text asserting a use it does not have.

**Three of my tests were vacuous and the reviewer was right about all three.** The filename test
reimplemented the naming rule in its own body, so deleting the rule from `main()` left it passing —
it could not detect the regression its docstring was about. The schema test compared a hardcoded
dict against itself. And the fingerprint tests used a 9-key stand-in against a ~30-key real parser,
so adding a key to the exclusion set by mistake passed the whole suite. All three now exercise
production code, including an exhaustive check that the set of non-invalidating arguments equals
`_FINGERPRINT_IGNORED` exactly.

**The top item was not a bug but a missing guard**, and it is now closed:
`test_uses_tau_matches_which_arms_actually_depend_on_tau` runs every arm at two taus and asserts
the answer matches `_uses_tau`. Previously that inference lived in one module while the invariant
it depended on lived in another, with nothing linking them — a future arm reaching
`find_merge_target` by a new route would have silently made every sweep point report the first
point's result.

**Migration:** adding the schema field and changing the fingerprint made 247 existing shards look
stale. Migrated in place and verified that all six (dataset, tau) configurations still reproduce
their reported solve counts exactly, so ~7h of computed search remains resumable — which matters
because the top open item (a wider tau on Countdown) reuses that baseline.


---

# Wrap-up: the three closing steps

## 1. Committed

Branch `merging-mechanism-split`, six commits. Left uncommitted deliberately: `prosqa.py`,
`test_prosqa.py`, `run_prosqa_reconvergence_probe.py`, `phase0-followup-plan.md`, the vendored
arc-dsl README, and `data/` — all pre-existing work from before this thread, not mine to commit.

**`results/` is gitignored, so no findings doc is version-controlled** — that is the repo's existing
convention (nothing under `results/` has ever been tracked), but it means the five write-ups from
this thread and ~7h of shards are still unprotected. Force-adding just the markdown would be a
one-line change to that convention if wanted.

## 2. Republished the two ARC experiments under deterministic move ordering

`results/ARC_DETERMINISTIC_REPUBLISH.md`. The published verdicts survive; two specific numbers do
not:

* Classical merge-vs-tree at budget 5000: **+0.067 → +0.008** (8x smaller, CI now symmetric about
  zero). Budgets 500 and 2000 reproduce their diffs exactly.
* GUCT-Uniform ARC at budget 500: the borderline merge *penalty* of −0.023 at p=0.058 — the closest
  thing to a significant merge penalty under that backup rule anywhere in this project — becomes
  **−0.000 at p=0.87**. It does not survive.

Node-reduction figures are unchanged in character (10–16% vs the published 11–13%), so the
deduplication was always real; it was the accuracy claim at the top budget that carried hash noise.

Also corrected: earlier in this thread I attributed the GUCT-ARC shift to the `is_goal` fix. Wrong
— run side by side under deterministic ordering, old and new goal semantics are identical. The
mover was the nondeterminism.

## 3. Crossed the reject arm with GUCT-Uniform's backup rule — the fixes are NOT redundant

`results/GUCT_UNIFORM_REJECT_CROSSING_FINDINGS.md`. Blocksworld, 30 puzzles, 10 repeats, both
treatment arms at identical settings:

| budget | tree | merge | reject | reject−merge | p |
|---|---|---|---|---|---|
| 100 | 0.033 | 0.233 | **0.323** | **+0.090** [+0.057, +0.123] | **0.0005** |
| 300 | 0.033 | 0.603 | **0.690** | **+0.087** [+0.017, +0.160] | **0.0380** |
| 1000 | 0.033 | 0.907 | 0.897 | −0.010 [−0.053, +0.030] | 0.8031 |

Under a backup rule that by itself already removes merging's harm elsewhere, declining to pool
*still* beats pooling at tight budgets, and ties once budget is ample. So the two fixes address
overlapping but distinct failures: the backup rule stops pooled statistics from destroying the
search, and the filter recovers the budget that re-deriving duplicate states wastes. Hence the gap
is largest exactly where budget is scarcest and vanishes where it is not.

Cost: the filter keeps one node per expansion (101/301/1001 vs 66.7/188.3/589.7), ~1.7x the nodes
at matched expansion budget.

**Testable prediction this generates:** the crossing should be *null* on the cube, 8-puzzle and
Sokoban, where GUCT-Uniform already brought merge and tree to parity, leaving no gap for the filter
to widen. If it is not null there, this explanation is wrong. Cheap to check — those are
seconds-to-minutes domains.

**And I made the same bug twice**, in two modules, independently: closing/marking a node that
"produced nothing new" as though it "leads nowhere", which silently deletes reachable search space.
Dropped Blocksworld's 3-block solve rate to 3/20 in the classical module and 3/10 here. Any future
arm that can decline an expansion needs this checked explicitly.


---

# RETRACTION and consolidation (final)

## The falsification test worked, but not the way I expected

Predicted: crossing the reject arm with GUCT-Uniform's backup rule should be **null** on the cube,
8-puzzle and Sokoban, since GUCT-Uniform already brought merge and tree to parity there.

Uncharged, it looked like a decisive **refutation** — reject won by +1.000 at six of twelve cube
budget points and +0.900 on Sokoban, while tree and merge were exactly equal everywhere. Then the
obvious check: is reject getting extra compute?

It was. **A rejected expansion never becomes an expansion, so under "budget counts expansions" it
was free — but it costs a real `apply_move`.** Cube at td=6/budget=40: merge spends 19.6 state
generations, reject spends 94.3. 2.4x, same nominal budget.

Added `charge_rejections` to all six affected configs and re-ran everything:

* **Blocksworld GUCT crossing: GONE.** Was +0.090 (p=0.0005) / +0.087 (p=0.038). Charged:
  +0.000 / +0.008 / −0.042, all p>0.67. **The claim I committed in 97fe924 is retracted.**
* **Falsification test: null once charged**, exactly as predicted. So the prediction was right and
  the earlier experiment supporting the opposite was the flawed thing.
* **Classical UCB1 Blocksworld: also does not survive.** +0.007 (p=0.58), +0.077 (p=0.31).
* **Classical UCB1 cube: survives**, every point inside budget (44/76, 84/108, 84/166, 163/166
  generations). merge 0.000, reject 0.983–1.000, also beating tree. This is the load-bearing
  classical result now.
* **LLM results unaffected** — there rejection is genuinely free, because `expand_node` draws all K
  candidates and computes their hidden states *before* the reject decision. That asymmetry is the
  actual reason the technique suits the reasoning search specifically, and it was not something I
  had articulated before this audit forced it.

**Corrected claim, narrower than what I argued for most of this thread:** the filter is a repair for
pooling's catastrophic failure mode, not a general improvement on merging. Where pooling works it
ties. Where pooling is broken it fixes it — and so does swapping the backup rule, which makes the
two repairs **redundant, not complementary**.

## Consolidated and version-controlled

`results/MERGING_MECHANISM_SUMMARY.md` is now the single authoritative statement, superseding the
individual docs where they disagree. All 34 markdown findings are tracked in git via a
`results/*` + `!results/*.md` pattern — they were untracked because `results/` is gitignored for
the pickles, which meant every conclusion and retraction lived only as a local file.

Note on that: my first gitignore attempt used `results/` + `!results/*.md`, which silently does
nothing because git will not descend into an excluded directory. And then my *verification* of the
fix used `git check-ignore -v`, which exits 0 on a negation match too, so it reported the fix as
broken when it was working. Third diagnostic failure of the session.

## What I would do next

1. A tau wide enough to force real false matches, on Countdown (where `reject_all` losing 5 of 19
   is evidence they bite) rather than ProsQA (where the sweep only showed insensitivity).
2. A second model — every LLM result here is Qwen2.5-7B with one projection.
3. Re-audit any remaining classical merge-vs-tree finding before quoting it comparatively. The
   accounting flaw only affects arms that decline work, so the pre-existing tree/merge results
   should be unaffected — but after this session that should be checked, not assumed.
