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

- [x] **Wide tau on Countdown.** Six points (published tau to 60x it, picked from the measured
      LOPO calibration table). Result: **the cube's "filter falls below baseline" does not
      reproduce** -- floors at 4/10 vs a 2/10 baseline even 60x past published tau. But
      `reject - treatment` flips sign past tau~0.36 (favouring pooling), never significant on this
      n=10 slice. And tau=1.11/2.0 are byte-identical -- the (depth,kind) bucket saturated, so the
      run could not distinguish "the filter is robust" from "this projection cannot be made bad
      enough to test it here", which is exactly the risk flagged before running it. Write-up:
      `results/COUNTDOWN_WIDE_TAU_FINDINGS.md`. Not committed.

- [x] **Is Countdown's null about scale?** Budgeted the cube down to match Countdown's own baseline
      solve rate (td=3/budget=28 -> tree=0.167, next to Countdown's 0.20) rather than matching raw
      budget numbers (not comparable -- cube's ~18-move branching vs Countdown's K=4). **No** --
      the collapse reproduces easily: pooling near-zero throughout, filter significantly below tree
      at tau=4 (-0.125, CI excludes zero, p=0.047). So the mechanism does not need the cube's larger
      budgets to show up; Countdown's null is a property of that domain/projection, not of scale.
      Folded into both write-ups.

- [x] **Sign-flip extension, seeds 71/72 at tau in {0.17, 0.36}, n=19.** Hit a checkpoint-
      fingerprint pitfall first: sharing one --output-dir across different held_out_seed values is
      refused (the directory fingerprint includes held_out_seed/n_low/n_high/calibrated_from, not
      just tau), so seed 72 needed its own dir. Fixed, re-ran. **Result: the flip does not survive
      more power.** At n=19: tau=0.17 filter significantly beats baseline (+0.421, p=0.0215, CI
      excludes zero) with filter-vs-pool still mildly favouring filter (+0.105, n.s.); tau=0.36
      filter and pool tie EXACTLY (11/19 each, delta 0.000) rather than pool pulling ahead as the
      n=10 slice suggested. So the apparent sign flip toward pooling was noise from 9 extra
      problems, not a real trend -- the filter's edge over pooling holds or ties throughout, and
      its edge over baseline strengthens to significant at 5x the published tau. Folded into
      `COUNTDOWN_WIDE_TAU_FINDINGS.md` and the authoritative summary.

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

---

# The empty cell: pooling broken AND the relation approximate

Every catastrophic-pooling result here used **exact** state identity (the cube under UCB1 +
MC-average). Every **approximate** relation (learned projection + tau) sat in a domain where
pooling worked. So the cell that decides whether the filter's repair depends on the relation being
exact has never been filled — and it is the cell the latent programme lives in, since a learned
relation is always approximate.

**Question.** On a domain where pooling is catastrophic, does the filter still repair it when the
equivalence relation is approximate, and how does that degrade as the relation gets worse?

**Design.** Rubik's cube, UCB1 + MC-average backup (pooling = 0.000 with an exact key). Replace
exact state identity with a relation built to mirror the LLM harness's: two states match if they
sit at the **same depth** and their **oracle distance-to-goal differs by less than tau**. That is
the same shape as `search.py`'s `find_merge_target` (same (depth, kind) bucket, |Δ 1-D value| < tau),
so the cube becomes a controlled stand-in for the learned-relation setting with ground truth
available.

Ground truth is the point: two states matched under this relation are usually *not* the same state,
so the **false-merge rate is directly measurable** rather than estimated from held-out data.

## Plan

1. [x] Approximate keying in `classical_mcts_rubiks.py`: `key_tau` (None = exact, unchanged),
       depth-bucketed index, match helper, wired into both the merge and reject arms.
       → verify: exact mode byte-identical to before on fixed seeds.
2. [x] Instrument `false_matches` / `total_matches` so each run reports its own measured
       false-merge rate. → verify: unit test with a hand-built case.
3. [x] Unit tests: approximate match fires; false matches counted; reject arm still a tree; exact
       mode a no-op. → verify: suite green.
4. [x] Sweep tau across three arms with rejections charged. → verify: tau=0 reproduces the
       published exact-key numbers.
5. [x] Findings doc + auto-review agent.

**Prediction, written before running.** Pooling stays at 0.000 throughout — it is already broken
with a perfect relation and a worse relation cannot help it. The filter degrades *gracefully*:
a false match costs it one discarded branch, so its solve rate should fall smoothly with the
false-merge rate rather than collapsing. If instead the filter collapses as soon as the relation
is approximate, the repair depends on exactness and the whole latent programme inherits that
limitation — which would be the most consequential negative in this project.

**Do not commit** without an explicit ask (memory rule).


## Result: the prediction held — but only after the first version was thrown out

**Pre-registered:** pooling stays at 0.000; the filter degrades *gracefully*, because a false match
costs it one discarded branch.

**First run said the opposite** — a cliff, 1.000 at τ≤1.5 and 0.000 at τ≥3 — and I wrote that up as
the prediction being falsified. The auto-review found the design confounded three ways:

1. **τ was a three-position dial, not a continuum.** Oracle distance changes by ±1 per move, so
   within a depth bucket every gap is even; τ ∈ {0.5, 1.5} selected identical match sets, as did
   {3, 6} (match counts 80, 80, 187, 187). Worse, the first band was *transitive* — exact identity
   on a coarsened space, i.e. not an approximate relation at all.
2. **The goal state was being swallowed** (34–44 events at τ=3), while `search.py` spares terminals
   by default. The arm was doing something the module it stood in for explicitly does not.
3. **Exact merging was still live underneath**, across depth buckets, so the measured relation was
   not the only relation in force and `false_matches` undercounted it.

Fixed by adding continuous per-state noise (SHA-256, not `hash()`, which is per-process random),
sparing the goal by default (`reject_goal`), and creating nodes directly in the `key_tau` branch so
nothing merges by exact identity underneath. Re-ran everything.

**Corrected result: graceful degradation, as predicted.** 8 puzzles × 15 repeats, arms paired on
identical seeds, rejections charged, noise ±1.0.

| τ | false-merge | td6/100 | td8/200 | td10/400 | tree baseline |
|---|---|---|---|---|---|
| exact | — | 1.000 | 1.000 | 1.000 | 0.533 / 0.400 / 0.367 |
| 0.25 | 0.50–0.57 | 0.908 | 0.958 | 0.983 | ″ |
| 0.5 | 0.66–0.72 | 0.875 | 0.942 | 0.883 | ″ |
| 1.0 | 0.78–0.80 | 0.642 | 0.850 | 0.875 | ″ |
| 2.0 | 0.86 | 0.142 | 0.242 | 0.275 | ″ |
| 4.0 | 0.909 | 0.000 | 0.000 | 0.000 | ″ |

Pooling is **0.000 in all 18 cells**. `filter − pool` is significant throughout: p=0.0078 at
τ ≤ 1, and at τ=2 p=0.0312 (td6, td8) / 0.0078 (td10). With n=8 paired puzzles 0.0078 is the
Wilcoxon floor (2/2⁸) — it means all 8 puzzles favour the filter, not a precise level.

Three things fall out, and two of them retract lines from the first write-up:

1. **The repair does not need an exact relation.** At a measured false-merge rate of 0.5 the filter
   still solves 0.91–0.98; at 0.78 it is 0.64–0.88, above the tree baseline on all three tiers.
   That is the answer the latent programme needed — the mechanism is not an artifact of exactness.
2. **But the filter can fall below doing nothing.** Above tree on all three tiers at τ=1, below on
   all three at τ=2, so the crossing is bracketed to 1 < τ* < 2 and not located; by τ=4 it is
   0.000 against a tree baseline of 0.367–0.533. (`filter − tree` is not significance-tested —
   the sweep only tests `filter − pool` — so this is secure at τ=4 and suggestive at τ=2.) Pooling was already on the floor and
   could not fall; the filter could. "A wrong rejection costs one discarded branch and does not
   accumulate" is **false past a threshold** — when the rejected branch is on the solution path, the
   cost is the solution.
3. **False-merge rate does predict it, monotonically** — within each tier both series move together
   across all 18 cells (pooled across tiers they do not, since the tiers have different baselines).
   **Retracted:** the first write-up's claim that it does not was an artifact of the 3-position dial.
   The second-model result it was paired with (a worse projection produced *better* pooling) stands
   on its own and is untouched by this.

**Verification.** `key_tau=None` is a strict no-op: **3240 paired runs across 108 configurations**
(extended after review to cover `value_source="oracle"`, which the original grid omitted despite it
being the setting every arm uses), zero mismatches on full graph signatures, and exact mode
reproduces the published cube node counts (44/76, 84/108, 163/166) to the digit.

**The recurring lesson, in a new costume:** a discrete-valued key makes a threshold sweep look like
a sweep when it is sampling two points. The grid {0.5, 1.5, 3, 6} looked like four conditions and
was two — and the collapse I read as a mechanism was the boundary between them.

Write-up: `results/APPROXIMATE_RELATION_FINDINGS.md`. Not committed — no ask this turn.

## Post-write-up review: my own compute correction was half wrong

The verification agent confirmed the 44 → 100 correction (a pooled expansion creates no node;
`new_nodes+merge_hits+rejected` equalled the monkeypatched real `apply_move` count in 80/80 runs;
the derived td10/b200 row is now directly measured). But it killed the interpretation I attached:

* **"The filter is the cheaper arm" is false** — an artifact of `run_search` having **no
  solve-based exit**. A terminal leaf is backed up and `continue`d before `_mark_exhausted`, so a
  solved run spins on the goal expanding nothing until `max_iterations = 20 × budget`. The filter
  stops spending once it wins; pooling, which never wins, spends to the last unit. Split by
  outcome the ranking inverts (td6/τ=1: 77.3 solved vs **103.9 failed**, above pooling's 100), and
  under GUCT-Uniform at td6/b40 the charged filter costs 40.6 against merge's 40.0.
* **My stated mechanism was wrong too** — "merge scores 0.000 so it never exits early". No arm
  exits early on a solve. Right conclusion, wrong reason.
* The surviving claim is the one that matters: **the filter never exceeds pooling's budget in the
  runs it wins**, which is exactly what `charge_rejections` must guarantee.

Also corrected, verified independently: the GUCT doc claimed merge "does not even spend its budget
(20 generations against 40) — the more compute-efficient arm". Counting `apply_move` directly,
merge spends **exactly** its budget in all six rows and the charged filter marginally more. And the
same mislabelled cube table was still published uncorrected two tables further down in that file,
plus in `PAPER_DRAFT.md` §7.1 — the section whose subject is this exact error.

Remaining known gaps, not fixed: `filter − tree` is never significance-tested; the crossing is
bracketed by a 5-point grid, not located.

---

# Round after the brief: four questions, and a correction

- [x] **Countdown correction.** `apply_projection` clips to [0,1], so τ >= 1 is the coarsest relation
      the bucketing allows. Saved tau=2.0 graphs: exactly one step node per depth in 10/10 problems --
      the filter became a single chain and still solved 4/10 vs baseline 2/10. "Could not be made bad
      enough" was wrong; it was made as bad as possible. Calibration cutoffs >1 come from unclipped OOF
      predictions.
- [x] **Proposal quality (cube, td8/b200).** New `expand_noise` option (None = random order, default
      unchanged: 3240/3240 no-op). Saturated-relation filter: 0.000 random -> 0.150 sd2 -> 0.975 sd1 ->
      1.000 sd<=0.5. Pooling (exact): 0.000 -> 0.283 -> 0.917 -> 1.000. **Both catastrophes need random
      proposals.** Tree unexpectedly degrades with better proposals (0.400 -> 0.000; shallower, more
      duplicates/undo moves) -- unexplained, variant-specific; filter-tree not meaningful there.
      Write-up: `results/PROPOSAL_QUALITY_FINDINGS.md`; folded into summary, cube, Countdown, paper docs.
- [ ] **Random-rejection control (LLM).** Single chain with no real dedup beat baseline 4/10 vs 2/10,
      so part of the filter's Countdown gain may be forced depth. Matched-rate random rejection would
      separate them. Not run. Note: changing search.py invalidates every checkpoint (fingerprint hashes
      it), so the baseline must be recomputed.
- [x] **Third model: Qwen2.5-1.5B on Countdown-3 (n=40).** 3B (3/60 in-band) and Math-7B (0/30,
      format) unusable. Raw: baseline 15, pool 10, filter 11 -- looked like deduplication hurts.
      Cause: a **second budget leak** -- a search whose every reachable leaf is a terminal answer spins
      to the iteration cap with budget unspent; 26/23 of 40 dedup runs stopped short and unsolved.
      Also present in the published 7B runs. **Compute-matched re-analysis** (baseline truncated to the
      arm's actual expansions; exact because PUCT has no budget-dependent schedule; validated 59/59):
      every published H3 gain gets stronger (7B Countdown pooling +0.316 n.s. -> +0.579 p=0.001);
      third model turns slightly positive (n.s.). Filter-vs-pool: another tie (+0.025) and confounded
      by pooling leaking more. Write-up: `results/THIRD_MODEL_BUDGET_LEAK_FINDINGS.md`.
- [ ] **Fix the LLM-harness leak** (design choice: widen on starvation / stop on verified answer /
      force expansion), then re-run filter-vs-pool. Every LLM number changes.
- [x] **Re-audit of classical merge-vs-tree compute claims.** Per charged expansion tree and merge do
      identical work in all 13 modules; no tree-vs-merge solve rate changes. But merge's node count
      undercounts its generations 1.1-2.4x, and ~15 older docs read node counts as compute. Fixed:
      consolidated erratum in the summary's section 4; inline corrections where a mechanism was wrong
      (BLOCKSWORLD "tree running out of room", H3 "compressed into a quarter" mixing early stopping);
      errata banner on ARTICLE_DERIVED; GUCT crossing Blocksworld column relabelled as node counts
      (arms are compute-matched); ARC row marked uncharged in summary and paper.
      **Reproduction gap closed:** no runner passed charge_rejections, so the paper's commands
      reproduced uncharged numbers. Added --charge-rejections to the five runners (default off; the
      Blocksworld runners write a separate _charged file); smoke-tested.
      **Footgun fixed:** key_tau on a tree arm pooled silently; now refused, test added.
      **Not fixed:** guct_uniform_arc.py breaks early when the root's last untried moves fail to
      execute (stopped 23/120 tree and 16/120 merge runs at budget 500) -- fixing it changes published
      GUCT-ARC numbers and needs a re-run.

---

# Testing "What would change the conclusion" (2026-10-02)

Scratchpad was wiped between sessions (it lives in /tmp): all scratch run shards and the brief's local
HTML are gone; every number survives in results/*.md. New runs write under results/.

## Plan
- [x] **Leak fix in search.py** (opt-in, default off): `widen_on_starvation` re-samples at the selected
      node, charged, ONLY when no unexpanded step node exists (structural starvation explained 41/41
      short unsolved published runs, incl. 5/42 baseline runs on 1.5B ProsQA). `stop_on_solve` ends a
      solved search (cannot change solved). Runner flags + `widenings` recorded (row schema 2).
- [x] **Random-rejection control**: `random_reject` arm drops step candidates (answers exempt) with a
      hashed, deterministic draw at `--random-reject-rate`, set to the filter's *realized* rate in the
      same leak-fixed run. Separates deduplication from forced depth.
- [x] Tests: 8 new; 5 mutants each caught (frontier guard, fake re-sample, answer exemption,
      exhausted-marking guard, stop_on_solve). 542 pass. Real-model smoke test: every arm spends full budget.
- [x] **1.5B settings re-run with the fix** (`results/LEAK_FIX_RERUN_FINDINGS.md`). Validation held:
      ProsQA baseline and filter identical to published on 42/42; only pool changed (+4, 0 lost).
      Pool − baseline +0.333 (p=0.0013) on ProsQA; Countdown-3 unchanged, arms trail baseline (n.s.).
      Random rejection at the filter's rate: +0.119 / −0.225 (p=0.004); filter beats it 11 vs 2
      (p=0.022) -> the relation deduplicates. Truncation compute-matching retracted as a substitute.
- [x] **7B settings done** (Docker quit; detached runs + vitest throttle). Both unchanged problem for
      problem; filter vs random 20 vs 3 across four settings (p=0.0005). Was: **7B settings blocked by memory**: float32 7B needs ~28 GB; local fana-train MLX servers hold
      ~33 GB of 64. First attempt thrashed 75 min, stopped. Ready: `results/leakfix/driver.sh cd7b prosqa7b`.
- [x] **Ops lessons**: scratchpad is /tmp (wiped between sessions) -> runs now write to results/;
      a 9.5 GB CPU job beside a 7B model caused the thrash; `grep -v` without --line-buffered hid driver
      progress; killing a driver's child just advances the driver.
- [x] **GUCT-ARC early stop** fixed (mirrors the other GUCT ports); regression test fails on old code
      (1 of 20 expansions) and passes now. Re-running the deterministic 30-task experiment.
- [x] **Correction:** LLM arms are NOT paired on identical LM draws (arm name is in the seed; root
      expansions differ in 10/10 published problems). Paired by problem only -- McNemar still valid,
      comparisons just noisier than claimed. Fixed in 6 docs + the runner comment.
- [x] **Review of the leak fix found 3 gaps** (iteration-cap leak with a frontier present: 4 runs;
      widen target at max_depth; random_reject without widening could end at root). Fixed + 5 tests,
      6 mutants caught, 547 pass. Both 1.5B settings re-run in full (`results/leakfix2/`): **328/328
      outcomes identical** to the first version; unsolved runs with budget unspent 4 -> 0.
