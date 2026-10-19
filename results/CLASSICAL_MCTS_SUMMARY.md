# No-LLM Classical MCTS Merging: Consolidated Summary

## Why this arc exists

Every H3 result in this project before this arc (`H3_MERGE_SEARCH_FINDINGS.md`) entangled two
separate questions: does merging help search, and does the *learned* merge criterion work --
because the merge decision was always an LLM hidden-state projection and every value estimate
came from LLM sampling. This arc isolates the first question in its purest form: classical UCB1
MCTS, exact-state-equality merging (a real transposition table), random-rollout value estimates,
zero LLM/learned anything, across four structurally different domains chosen specifically to vary
one property at a time.

## The four domains

| Domain | Structure | Result | Node reduction |
|---|---|---|---|
| Connect Four | irreversible, adversarial | **Null** -- zero solve-rate difference at every budget (10-400) | up to 23% |
| 8-puzzle | reversible, single-agent | **Large positive** -- baseline stuck at 0.000-0.215 where treatment reaches 1.000 | up to 62% |
| Three Men's Morris | reversible, adversarial | **Null** -- zero difference at every budget (3-200), for a different reason than Connect Four's | up to ~16% |
| Sokoban | partially reversible, single-agent | **Large positive**, comparable in scale to the 8-puzzle | up to 81% |

Every result is a real, well-powered, budget-swept finding with node-count evidence
distinguishing "no merging happened" from "merging happened but didn't help" -- not a single one
is a raw ceiling/floor artifact; each was checked with a per-difficulty breakdown before being
trusted (see each domain's own findings doc for the full sweep tables).

## How the mechanism story evolved, domain by domain

1. **Connect Four (null)**: UCB1's own `untried_moves` bookkeeping already prevents redundant
   exploration when moves are irreversible -- every expansion is guaranteed genuinely new
   ground, so there's nothing wasteful for merging to reclaim.
2. **8-puzzle (large positive)**: reversible moves let a plain tree revisit the same state many
   times over as fresh, separately-tracked, low-visit nodes. Merging pools those fragments into
   one well-estimated node. First pass at the hypothesis: *reversibility drives the effect.*
3. **Morris (null)**: reversibility alone was not sufficient -- disproves the first-pass
   hypothesis. Adversarial, multi-piece interleaving makes reaching the identical joint state via
   a different move order a rare coincidence (both sides' moves must cancel out), even though
   every individual slide is fully reversible. Revised hypothesis: it's *redundant-exploration
   density under UCB1*, not reversibility per se, that predicts the effect -- and reversibility is
   necessary but not sufficient for that density.
4. **Sokoban (large positive, comparable to the 8-puzzle)**: refines the revised hypothesis
   further. Predicted a modest, intermediate effect (only walking is reversible, pushing mostly
   isn't). Got an effect as large as the fully-reversible 8-puzzle instead. Reason: what predicts
   the effect isn't the fraction of move *types* that are reversible, it's whether the search's
   *dominant activity* is reversible. Most legal moves at most Sokoban states are plain walks
   (pushes are comparatively rare opportunities), so the search spends most of its budget on
   fully-reversible exploration regardless of pushing's one-way nature.

**Final claim**: merging helps in proportion to how much of the search's actual exploration
activity revisits already-seen states under UCB1 -- which depends on what the search spends most
of its moves doing, not on a simple reversible/irreversible move-type ratio, and not on
reversibility alone. Two structurally different mechanisms can each independently suppress this
(Connect Four's built-in anti-redundancy vs. Morris's adversarial-interleaving-dilutes-revisits),
and two structurally different domains can each independently generate it (the 8-puzzle's single
continuous mobile element vs. Sokoban's walking-dominated branching).

## Connecting back to the original LLM-based H3 result

`H3_MERGE_SEARCH_FINDINGS.md`'s eventual significant result (Qwen2.5-7B-Instruct, num_numbers=4,
McNemar p=0.0215 at n=14, strengthening to a larger, still-significant effect at n=19) was driven
by a diagnostic that reads, in hindsight, exactly like this arc's mechanism: **baseline was
"pinned at 59-60 unique nodes on every problem"** -- independent K-sampling at each expansion kept
regenerating near-duplicate candidates that consumed real LLM calls while contributing almost
nothing beyond what their twin already contributed, and merging reclaimed that waste. That's
precisely the redundant-exploration-density pathology this arc has now reproduced, understood,
and shown to generalize (and *not* generalize) across four classical domains with zero LLM
involved. The LLM result was never really about "does graph search beat tree search" in general
-- it was a fix for independent-sampling's specific redundancy problem, and this arc shows that
same fix helps or doesn't help in classical search exactly to the degree the underlying search
process independently generates that same kind of redundancy.

## Status

This closes the user's two explicitly requested follow-up domains (Morris, then Sokoban) on top
of the original Connect Four / 8-puzzle pair. No domain is queued by default next. The four
results together are a complete, internally consistent, mechanistically-explained account of when
node-merging helps classical tree search -- a cleaner and more general finding than anything the
LLM-only experiments alone could have established, and a natural stopping point for the no-LLM
investigation pending further direction from the user.
