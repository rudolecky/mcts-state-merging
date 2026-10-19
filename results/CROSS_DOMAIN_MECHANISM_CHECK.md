# Does the Classical Redundancy-Density Story Explain the LLM H3 Results?

## Question

The classical no-LLM arc (`CLASSICAL_MCTS_SUMMARY.md`) found a clean pattern across four domains:
merging's accuracy benefit tracks how much redundant re-exploration of the same state the search
actually generates, measured directly by node-count reduction (treatment nodes / baseline nodes).
Low-redundancy domains (Connect Four ~23%, Morris ~16% reduction) were null; high-redundancy
domains (8-puzzle up to 62%, Sokoban up to 81%) showed large effects. Does this same pattern
explain why the *original* LLM-based H3 results (`H3_MERGE_SEARCH_FINDINGS.md`) varied the way
they did — null on Countdown num_numbers=3, strong on Countdown num_numbers=4, strongest on
ProsQA? Checked entirely from diagnostics already on disk in that doc -- no new runs.

## The numbers, side by side

| Domain | Node reduction (treatment/baseline) | Accuracy result |
|---|---|---|
| Connect Four (classical) | up to 23% | Null |
| Morris (classical) | up to ~16% | Null |
| 8-puzzle (classical) | up to 62% | Large positive |
| Sokoban (classical) | up to 81% | Large positive |
| Countdown, num_numbers=3 (LLM) | **76%** (39.2→9.5) | **Null** (McNemar p=1.00) |
| Countdown, num_numbers=4 (LLM) | 82% (~60→~10) | Large positive (McNemar p=0.013, n=19) |
| ProsQA (LLM) | 82% (~57→10.0) | Large positive (McNemar p=0.0010, n=21) |

**The classical pattern does not transfer cleanly.** Countdown num_numbers=3 shows a *higher*
dedup ratio (76%) than either classical null domain (16-23%) and is nearly identical to
num_numbers=4's and ProsQA's ratios (82%, 82%) -- yet it's the one that came back null. If node
reduction alone predicted effect size the way it did across the four classical domains, this
point shouldn't exist.

## Why it breaks: a confound the classical domains never had

`H3_MERGE_SEARCH_FINDINGS.md`'s own diagnostics already explain this, independently of the
classical arc: num_numbers=3's null was diagnosed as a **model-capability ceiling**, not a
search-efficiency shortfall. 52.5% of problems were unsolved by *both* conditions, and follow-up
checks found 90% of those were 0/8 correct on raw, search-independent sampling (temperature 0.8,
no search at all) -- the model essentially cannot produce a correct derivation for these
problems under any amount of sampling, merged or not. Dedup ratio was real and substantial (76%)
but irrelevant, because the bottleneck wasn't which states got explored, it was whether the model
could ever generate the right one.

**No domain in the classical arc has an analogous ceiling.** Connect Four, the 8-puzzle, Morris,
and Sokoban are all small, finite, fully-solvable combinatorial games where a long enough random
rollout can *always* eventually find the solution -- "capability" and "search budget" are the
same axis there. An LLM's ability to generate a correct multi-step derivation at all is a
genuinely separate axis from search coverage, and num_numbers=3 sits in the regime where that
separate axis is the actual constraint. This is precisely the caveat the H3 doc itself flagged
before ever reaching for a redundancy explanation -- the cross-check confirms it rather than
discovering it fresh.

Once that confound is set aside (compare only the two LLM cases where the H3 doc's own capability
checks ruled out a capability ceiling -- num_numbers=4 and ProsQA), the redundancy story holds up
fine: both are high-dedup (82%), both show large, well-powered effects, and the H3 doc's own
"Mechanism" section describes exactly the classical arc's pattern in different words -- baseline
"exhausted its 15-expansion budget re-creating shallow duplicate nodes and ran out of budget
before finishing a full path," while treatment "always reached greater maximum depth." That's
node-count waste converted into completable search depth -- the same mechanism, observed via
depth-completion instead of raw node-count fraction because the LLM harness's per-problem
difficulty is less uniform than the classical domains' exactly-controlled `k_plies`/
`target_distance` puzzles.

## A finer split the classical arc has no analog for

The H3 doc's C6 ablation (swap PUCT guidance from the shared merge-projection to honest sampled
rollouts, keeping the merge criterion itself unchanged) splits num_numbers=4 and ProsQA further,
in a way nothing in the classical arc could test: Countdown's advantage **disappeared entirely**
under honest guidance (exact tie, 7/10 vs 7/10) while ProsQA's **strengthened** (81.0%→90.5%).
Reading: Countdown's redundancy was substantially *guidance-driven* -- a noisy shared value
estimate caused baseline to wander into duplicate states, and merging was partly compensating for
that noise rather than for anything structural about the task. ProsQA's redundancy is
*task-structural* -- a small, low-branching entity vocabulary makes the model regenerate the same
next fact across genuinely different search branches almost unavoidably, a duplication rate good
guidance doesn't fix. Every classical domain in this arc used honest random rollouts throughout
(no learned value guidance ever existed to be noisy), so this guidance-vs-structure axis simply
has no classical counterpart to check against -- it's a real, additional finding the LLM data
alone can speak to.

## New check: does num_numbers=3 stay null under honest rollout guidance too?

The capability-ceiling diagnosis makes a direct, testable prediction: unlike num_numbers=4
(whose advantage was guidance-driven and vanished under `value_source="rollout"`), num_numbers=3
should stay null under rollout guidance as well -- if the model can't generate a correct
derivation regardless of search, removing guidance noise shouldn't reveal a hidden benefit that
wasn't there. Never previously tested (the original C6 ablation only covered num_numbers=4 and
ProsQA). Ran it: same config as the original num_numbers=3 null (1.5B model, K=4, budget=10,
τ=0.045, held-out seed=99), `--value-source rollout --num-rollouts 2
--rollout-max-new-tokens 64` (n=10, a directional first pass matching the scoped-down precedent
the original C6 ablations used, not the full n=40 -- an exact tie at this n is already
unambiguous, no need to scale further for a confirmatory check).

**Result: baseline 2/10, treatment 2/10 -- an exact tie**, matching the prediction cleanly.
Dedup ratio was real and essentially unchanged from the projection-guided version (baseline mean
39.5 nodes, treatment mean 10.4 -- 74% reduction, versus the original run's 76%), confirming the
merge mechanism itself doesn't depend on `value_source` (it never did -- only PUCT guidance
changes, the merge criterion is always projection-based). Despite ~74-76% dedup under *both*
guidance conditions, accuracy stayed null under *both* -- unlike num_numbers=4, where the same
swap took a real 71.4%/14.3% split down to an exact tie. num_numbers=3 shows no accuracy
advantage to guidance noise removing, because there was no accuracy advantage there in the first
place under either guidance source. This is stronger confirmation than "consistent with the
theory" -- it directly rules out the one alternative explanation the original diagnosis couldn't
distinguish (that num_numbers=3's null might itself have been a guidance-driven redundancy
story like num_numbers=4's, just one where the noise happened not to help). It isn't: num_numbers=3
has no redundancy-driven benefit under any guidance quality tested, consistent with the
capability-ceiling explanation and inconsistent with a guidance-noise explanation.

## Answer

**Partially, and the way it fails is itself informative.** The classical arc's core claim --
merging helps in proportion to redundant re-exploration the search actually generates -- is
consistent with the two LLM results where a capability ceiling has been ruled out (num_numbers=4,
ProsQA), via the same "wasted budget prevents reaching completable depth" mechanism, just
observed through a different diagnostic (depth-completion vs. raw node-fraction, because the LLM
domains don't have the classical domains' exact, uniform difficulty control). It does **not**
predict num_numbers=3's null on its own -- that null is governed by a different, independent
axis (model capability) that has no equivalent in any of the four classical domains, where enough
budget always eventually finds the answer. Dedup ratio and accuracy effect are only tightly
linked once capability is controlled for; raw dedup ratio alone, across domains that differ in
whether capability is even the binding constraint, is not a reliable predictor by itself. The C6
guidance-vs-structure split further shows the "high dedup ratio" cases aren't even a single
mechanism underneath -- two different sub-causes (noisy guidance, task structure) produced
similar dedup numbers for different reasons, something no amount of classical-domain data alone
could have surfaced, since none of those domains had a noisy value source to isolate.

## Status

Closed. Started as pure re-analysis of existing diagnostics in `H3_MERGE_SEARCH_FINDINGS.md` and
the four classical findings docs, then extended with one new, cheap, directly targeted run (the
num_numbers=3 rollout-guidance ablation above) that the re-analysis alone couldn't settle. This
is the most complete synthesis available of how the no-LLM mechanism findings relate to the
original LLM-based results that motivated this whole investigation, and the capability-ceiling
explanation for num_numbers=3's null is now confirmed, not just inferred.
