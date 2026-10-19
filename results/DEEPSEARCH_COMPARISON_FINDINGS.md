# Vanilla PUCT vs. DeepSearch-Style Search, Merge vs. Tree, on ProsQA

## Question

Two connected asks: compare this project's own MCTS against another algorithm (not just vanilla
UCB1/PUCT) on a shared benchmark, and review arXiv:2509.25454 ("DeepSearch," ICLR 2026) to decide
whether to replicate and test it. Read in full: DeepSearch embeds MCTS into LLM RL *training* for
math reasoning, contributing (a) **global frontier selection** -- scoring every unexpanded
frontier node across the whole tree at once and picking the global best, instead of root-to-leaf
UCT descent -- and (b) a depth-decayed, sign-constrained q-value backup. Like every other paper
reviewed in this project, it never merges or deduplicates states; its own stated motivation for
frontier selection is explicitly "eliminating redundant root-to-leaf traversals," addressing a
redundancy problem through *selection*, not *state identity*. That made it a sharp test of whether
merging still helps on top of a smarter selection strategy, not just on vanilla PUCT.

**Scoped to the inference-time search mechanism only** (frontier selection + backup rule), not
DeepSearch's actual RL training loop (Tree-GRPO, q-value soft clipping) -- that trains a
128xH100-cluster-scale policy update, wildly outside this project's scale. Ported into this
project's existing frozen-LLM search harness (`search.py`) instead, run against **ProsQA**, this
project's cleanest and most reliable H3 result, reusing the exact projection/tau/model already
deployed there (`Qwen2.5-7B-Instruct`, tau=0.01, layer `mid`) and the exact same 21 held-out,
calibration-filtered instances as the original H3 trust-bar run.

## What's confirmed, not assumed

- Global frontier score (Eq. 6-7): `F(s) = λ1·tanh(Q_parent(s)) + λ2·H(π(s|o_s)) + λ3·D(depth(s))`,
  `D(depth(s)) = sqrt(depth(s)/max_depth)`. `Q_parent(s)` uses the parent's own accumulated value,
  not the frontier node's own (confirmed necessary here too, not just an artifact of the paper's
  own tree: a frontier node in this harness never gets backed up until it's selected and expanded,
  so its own value is always uninformative). Generalized to the mean over multiple parents under
  `merge_enabled=True` (the paper's own tree has exactly one parent per node).
- Entropy term reuses `next_token_entropy_from_hidden` (already built for this project's own
  false-merge-rate analysis), extended to average across every token position in a step's own
  span -- a direct extension of an existing building block, not a new mechanism.
- Backup rule (Eq. 3/5): depth-decay `γ(i,l)=max(i/l, 0.1)`. The extracted equation has a genuine
  OCR artifact (two of its three branches read identically); implemented as a best-faith
  reconstruction preserving the paper's own stated invariant (q >= 0 for nodes on any path that
  reached a correct solution) -- flagged, not presented as a verified transcription.
- Outer loop rhythm ported faithfully, not flattened: batches of
  `iterations_per_backup_round` {select, expand} rounds before one backup phase targeting either a
  newly-found correct trajectory or (Eq. 2) the lowest-entropy incorrect one -- a real structural
  difference from this project's usual per-expansion backup loop.
- Verified: `tests/test_search_deepsearch.py` (11 new tests: hand-computed frontier scores
  including the multi-parent-mean case, the backup rule's non-negative invariant explicitly
  checked -- not just that some update happens -- and harness-neutrality mirroring `search.py`'s
  own established mocked-model test convention) + full suite (331 passed) green.
- Calibrated with a 2-problem real-model pilot before the real run (confirmed correctness, no
  crashes, real timing: ~283s/problem across all 4 arms) before committing to the full n=21 run.

## Results

21 real ProsQA problems (the exact held-out set from this project's original H3 trust-bar run),
budget=15, K=4, real `Qwen2.5-7B-Instruct` inference:

| algorithm | baseline (tree) | treatment (merge) | diff | 95% CI | wilcoxon p | nodes tree->merge |
|---|---|---|---|---|---|---|
| vanilla PUCT | 0.238 (5/21) | 0.810 (17/21) | +0.571 | [0.381, 0.762] | 0.0005 | 56.5 -> 9.5 (83% fewer) |
| DeepSearch-style | 0.476 (10/21) | 0.762 (16/21) | +0.286 | [0.048, 0.524] | 0.0339 | 52.4 -> 12.3 (77% fewer) |

**Merging helps both algorithms, significantly (both CIs exclude zero) -- but by very different
margins.** Discordant pairs: vanilla PUCT flips 12 problems from unsolved to solved with merging,
zero the other way (perfectly one-directional, matching this project's original ProsQA finding).
DeepSearch flips 7 unsolved->solved, but **1 solved->unsolved** -- not perfectly one-directional
here, a genuine small exception worth reporting rather than smoothing over.

**DeepSearch's own tree-mode baseline is much stronger than vanilla PUCT's** (0.476 vs. 0.238) --
its global frontier selection genuinely helps on its own, exactly as the paper claims. **But its
merge-enabled ceiling (0.762) is slightly below vanilla PUCT's (0.810)**, and it costs more:
average per-problem wall-clock was close for the tree condition (153.4s vs. 154.1s) but DeepSearch's
merge condition took noticeably longer (123.3s vs. PUCT's 85.6s) -- the extra per-token entropy
computation and whole-tree frontier scan (versus a single root-to-leaf walk) add real overhead
that shows up most once merging has already shrunk the tree.

## Mechanism

DeepSearch's global frontier selection and this project's merging address two different sources of
wasted search effort, and they only partially overlap. Selection quality (frontier scoring vs.
plain root-to-leaf UCT) determines *which* already-existing part of the tree gets explored next;
it never touches whether two different reasoning paths that happen to reach an equivalent
intermediate state get represented as one node or two. Smarter selection can *reduce* how often a
search wastes budget re-treading unproductive territory (raising DeepSearch's own baseline well
above vanilla PUCT's), but it cannot eliminate the specific redundancy merging targets: the same
semantic reasoning state, reached via different token-level phrasings, still gets created as
separate nodes under DeepSearch's tree mode exactly as under vanilla PUCT's, and merging still
reclaims a large, real fraction of that (statistically significant, though smaller than under
vanilla PUCT -- consistent with DeepSearch's own selection already having reclaimed *some*, not
all, of the same waste). This is the clearest demonstration yet in this project that merging and
guidance/selection quality are answers to different questions -- selection quality can shrink the
size of the problem merging solves, but doesn't make it disappear, even for an algorithm
specifically designed to make search more efficient.

The one solved->unsolved flip under DeepSearch's merge condition, and its slightly lower
merge-enabled ceiling than vanilla PUCT's, are also worth taking seriously rather than
explaining away: DeepSearch's own backup rule and entropy-driven selection were designed for a
different objective (training-data diversity and credit assignment for policy gradient updates),
not for maximizing solve rate at a fixed inference-time budget. It's plausible that the extra
machinery built for that different purpose doesn't transfer as cleanly to pure accuracy
maximization as a simpler mechanism combined with merging does.

## Scope and limitations

- **Entropy is computed from raw, untempered logits** (`next_token_entropy_from_hidden`'s existing
  behavior), not adjusted for the temperature (0.8) actually used during generation -- a further
  documented simplification, directionally the same signal as the paper's own per-token entropy
  but not verified to match it exactly.
- **`iterations_per_backup_round` (=4) is this project's own hyperparameter**, not the paper's --
  it never states a batch size for the search mechanism itself (only an unrelated Pass@K
  threshold), so no "correct" value exists to match; not tuned beyond this default.
- **Eq. 5's third branch is a best-faith reconstruction** of an OCR-corrupted equation, not a
  verified transcription -- see `search_deepsearch.py`'s own docstring.
- **N=21 is this project's own established trust-bar for ProsQA**, not independently re-validated
  at a larger N for this specific 4-arm comparison.
- **Not the training method** -- Tree-GRPO, q-value soft clipping, and the iterative
  hard-example-filtering training loop are entirely out of scope; only the inference-time search
  mechanism was reimplemented and tested.
- **A later direct test (`RANDOM_VS_MCTS_FINDINGS.md`) shows a pure random-restart control beats
  every arm reported below**, including merge-enabled PUCT (0.905 vs. 0.810) and merge-enabled
  DeepSearch (0.905 vs. 0.762), on this project's exact same n=21 held-out set. This doc's framing
  of the two baselines (0.238, 0.476) as "not collapsing to near-zero" is still accurate on its own
  terms, but should not be read as evidence that guidance prevents the tree-vs-random failure mode
  described there -- it does not (see that doc's "Real guidance doesn't prevent it either"
  section). The comparison in this doc (merging helps both algorithms, DeepSearch's selection
  raises its baseline) still holds; what's revised is only the cross-doc inference about guidance
  and the random-restart control, not any number reported here.

## Status

Closed. Merging helps both vanilla PUCT and a faithfully-ported DeepSearch-style global-frontier
search on the same real benchmark, significantly in both cases -- but DeepSearch's own smarter
selection raises its untreated baseline well above vanilla PUCT's, shrinking (not eliminating) the
gap merging closes, at a real per-problem compute cost. This is the clearest evidence yet that
selection/guidance quality and state-identity merging are separate, non-competing levers on search
efficiency. No further follow-up queued by default.
