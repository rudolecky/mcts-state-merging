# ProsQA's First Ground-Truth Transposition Validation, and the Forced Reconvergence Probe That Reversed It

## Question

Following the discussion of a path-conditioned learned merge criterion (merging based on the
trajectory leading to a state, not just the state's own embedding), the agreed first step was
cheaper than designing any new model: does the *current* criterion (a Ridge projection of the
hidden state, thresholded at the already-deployed tau=0.01) actually have a measurable false-merge
problem on ProsQA, and if so, do the false merges share a path-shaped signature a path-encoder could
exploit? Before this, ProsQA had no real ground-truth transposition check at all -- its existing
"false merge" number (11.3% at tau=0.01, `H3_MERGE_SEARCH_FINDINGS.md`) is a value-outcome proxy
(`|ΔV-hat| > 0.3`) that can only flag a *suspicious* merge, never confirm a *false* one against real
state identity. Connect Four has exactly this kind of check (`CONNECT_FOUR_FINDINGS.md`, n=17
problems, 228 pairs, precision ~0.95-1.0, recall 1.000); ProsQA did not, until this.

## Method

New `prosqa.canonical_state_at(instance, step_bodies_so_far)`, mirroring `connect_four`'s own
ground-truth key function: keys on the *current entity reached* (not the full fact-sequence), since
facts are never consumed -- two prefixes reaching the same entity are provably interchangeable for
everything that follows, the same Markov property the classical domains' exact board/puzzle state
has. Fails closed (returns `None`) on any unparseable step, off-fact-set fact, or non-contiguous
jump, exactly like `connect_four`'s version.

Collected 18 fresh positive ProsQA instances (chain_len 3-5, reconverge_fraction=0.4 -- the same
generation parameters the deployed H3 run used), K=4 traces each, using the already-deployed
projection (`qwen7b_prosqa_projection.json`, layer `mid`) and tau (0.01). 8 of the 18 instances have
`path_count >= 2` -- a real, by-construction reconverging shortcut edge, a second genuinely
different fact-sequence reaching the same downstream entity as the main chain.

**A methodology correction made before running anything**: `geometry.build_pairs`'s
`same_step_only=True` default (inherited from Connect Four, where a real transposition is by
definition same-move-count) is the wrong restriction here -- ProsQA's real transposition source, the
reconverging shortcut, is explicitly a *different* number of steps to reach the same entity. Ran the
confusion matrix both ways: `same_step_only=False` as the headline (so cross-depth transpositions
aren't excluded from the pairing window before the check even starts), `same_step_only=True` kept as
a secondary, Connect-Four-comparable sanity number.

## Results

191 snapshots collected, 119 with a real ground-truth key (72 skipped -- malformed/non-contiguous
prefixes, expected for free-form generation).

| pairing | n_pairs | true_merge | false_merge | missed_merge | correct_non_merge | precision | recall |
|---|---|---|---|---|---|---|---|
| same_step_only=False (headline) | 423 | 144 | 1 | 0 | 278 | 0.993 | 1.000 |
| same_step_only=True (secondary) | 224 | 144 | 1 | 0 | 79 | 0.993 | 1.000 |

**Both pairings give the identical precision/recall, off one single false-merge pair.** This is
close to Connect Four's own benchmark (precision 0.989, recall 1.000 at n=17/228) -- the currently-
deployed ProsQA merge criterion shows essentially no measurable false-merge problem in this sample.

**The one false merge found, in full**:
```
[pq_rompus95_quimpus99_yes] trace3@step3='brimpus98' vs trace3@step4='quimpus99'
  i: (rompus95 is a dumpus96, dumpus96 is a wampus97, wampus97 is a brimpus98)
  j: (rompus95 is a dumpus96, dumpus96 is a wampus97, wampus97 is a brimpus98, brimpus98 is a quimpus99)
```
Record `j`'s path is record `i`'s path plus exactly one more step -- **the same single trace, one
step apart, not two different routes to the same place.** The projection placed these two hidden
states within tau even though the ground-truth entity moved forward. This is ordinary embedding
imprecision between adjacent reasoning steps of one continuous derivation, not a route-ambiguity
case. Path history would show `j` is literally `i`'s own history plus one token further -- there is
no alternate-route signal here for a path-encoder to exploit; the two "paths" are nearly identical by
construction, so this single false merge does not motivate the design under discussion.

**Path-shaped signature check** (false-merge pairs vs. the full same-problem pair pool): with n=1
false merge, no real distributional signature is measurable (`step_idx_diff`: 1.0 vs. pool mean 1.36;
`same_trace_fraction`: 1.0 vs. pool mean 0.18 -- a 5.6x lift, but off a single data point, not a
trend). **Reported honestly as inconclusive, not as a discovered pattern** -- one pair is not enough
to characterize anything.

## The real limitation: the interesting case never got tested

Despite 8/18 instances having a genuine reconverging shortcut, **zero pairs in either pairing (same-
step or cross-depth) had matching ground-truth keys at different depths** -- `true_merge` is 144 in
both rows, identical, meaning every true-positive pair the check actually found was already same-
step. The free-form K=4 sampling here never happened to produce, within one problem, both a
shortcut-route trace and a main-chain-route trace landing on the same entity at different depths --
the one scenario this whole investigation is actually about (do two genuinely different histories
reaching the same state get correctly recognized as one state) was never exercised by this sample,
despite the dataset supporting it by construction.

## Conclusion (from the offline check alone -- superseded below, kept for the record)

**No evidence found that ProsQA's current state-only merge criterion has a false-merge problem
worth fixing.** Precision/recall are both excellent, and the sole false merge found isn't a case a
path-conditioned criterion would plausibly help with. On the strength of this alone, there's no
measured gap that motivates designing and training a path-encoder.

**But this isn't a clean negative either** -- the one case that would actually test the premise
(genuine alternate-route reconvergence, same entity, different depth, different trace) never
appeared in the collected pairs, so the real question ("does the current criterion correctly
recognize a true cross-route transposition") remains untested, not answered no. A decisive follow-up
would need to bias collection toward eliciting shortcut usage specifically (e.g. filter to instances
where the shortcut is the shortest viable route, or directly construct synthetic prefixes using the
shortcut fact rather than relying on free-form sampling to find one) before this question can be
closed either way.

**This follow-up was run immediately (below), and it reverses this conclusion**: forced directly,
the same criterion misses 39/40 genuine transpositions. Read the section below as the operative
result; this section is kept only to show the reasoning trail, not as standing guidance.

## Follow-up: forcing the untested case directly (`run_prosqa_reconvergence_probe.py`)

The gap above was closed immediately, not left open. New `prosqa.find_reconvergent_routes(facts,
start)` DFS-enumerates every simple path from `start` over an instance's own fact graph and returns
the first entity reachable via >=2 distinct-length paths -- the exact reconverging-shortcut
structure `_build_component` builds, found generically from the graph itself rather than from
generator-internal state.

40 fresh positive instances, `reconverge_fraction=1.0` (all 40 qualified). For each, both routes'
step-body text were constructed directly (no sampling) and each fed through one teacher-forced
forward pass (`build_chat_prompt_ids` + a hand-encoded continuation + `hidden_states_for_sequence`,
reusing `find_step_boundaries` unchanged -- confirmed generation-agnostic, a pure decode/text scan,
so it works identically on synthetic continuations) to get the real hidden state and projected value
at each route's endpoint. A self-check (`canonical_state_at` agrees on both routes' entity, every
case) confirmed the harness itself was sound before trusting any result.

**Result: 1/40 recognized as one state (recall = 0.025).** The two independent validations disagree
sharply: the offline check above reported recall 1.000 because it never actually paired a real
reconvergence case; forced directly, the deployed criterion (Ridge projection, tau=0.01) almost
never recognizes them. **This is the real number** -- the offline one was measuring the wrong thing
entirely, not a weaker version of the same thing.

**On mechanism, checked rather than assumed**: hop-count difference between the two routes
correlates only weakly with the projection distance (Pearson r=0.15 across the 40 cases; mean
`proj_short - proj_long` = 0.049, not a clean or dominant effect). So this isn't simply "the
projection penalizes more steps taken" -- the miss is more idiosyncratic than one obvious confound
explains. The single recognized case had `dist=0.0000` because *both* routes' projected values hit
the clip ceiling (1.0) -- plausibly a saturation artifact, not genuine semantic agreement; treated as
a coincidental hit, not a real success case.

**What this does and doesn't establish**: it establishes, with real forced evidence (not absence of
data), that the current single-snapshot scalar projection fails to recognize genuine cross-history
transpositions almost every time. It does *not* yet establish that the fix specifically needs to be
*path-conditioned* (as opposed to just a richer single-state representation -- e.g. this project's
own Connect Four validation already found the learned scalar projection is no better than untrained
raw hidden-state distance at detecting real transpositions there). The natural, cheap next check
before designing a path-encoder: rerun these same 40 cases comparing raw hidden-state distance (no
Ridge collapse) against the same ground truth -- if raw distance also misses badly, that points at
the single-snapshot representation itself being insufficient (motivating path/history conditioning
specifically); if raw distance does much better, the fix is cheaper than a new architecture (stop
collapsing to one scalar), not evidence for path-conditioning per se.

## Status

**Reopened, and now motivated by real evidence rather than closed on an absence of it.** The offline
check found no problem; forcing the actual case the whole question is about found a large, real gap
(1/40 recognized). Next step before any path-encoder design: the raw-distance re-check above, to
separate "needs path history" from "needs a less lossy single-state metric" -- not yet run, not
scoped without the user's go-ahead first.

## Critical files

- `src/mcts_phase0/datasets/prosqa.py` -- `canonical_state_at`, `find_reconvergent_routes`
- `src/mcts_phase0/collect.py` -- wired `ground_truth_key_fn=prosqa.canonical_state_at`
- `src/mcts_phase0/geometry.py` -- `SnapshotRecord.step_bodies`, `ground_truth_merge_confusion(return_pairs=...)`
- `src/mcts_phase0/run_prosqa_ground_truth_validation.py` -- the offline check (recall 1.000, the wrong test)
- `src/mcts_phase0/run_prosqa_reconvergence_probe.py` -- the forced probe (recall 0.025, the real number)
- Raw output: `results/prosqa_ground_truth_validation/{records.pkl,confusion.json}`,
  `results/prosqa_reconvergence_probe/results.json`
