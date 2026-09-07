# Before Trusting a Tree Search: Benchmark Against Random Restarts

A practical checklist distilled from `RANDOM_VS_MCTS_FINDINGS.md`: across every classical domain
in this project (Blocksworld, the 8-puzzle, Sokoban, Connect Four, Morris) plus ARC-AGI-1 program
synthesis, blind tree-guided search (UCB1-MCTS with no heuristic/no learned value) lost to
`budget` independent random rollouts at low budget, by up to 99+ percentage points -- and on
ProsQA, PUCT search *with a real learned value projection guiding it* lost the same way (28.6% vs.
90.5%, see item 2 below). This is not a claim that MCTS is broadly worse than random search. It's a
claim that **a specific, common failure mode is invisible unless you check for it, is not fixed by
guidance quality alone, and, remarkably, whether the gap persists or closes as budget grows turns
out to predict whether merging will help, before you've built any merge machinery at all.**

## The check

Before trusting a tree search's own reported solve rate as evidence that "search helps" on a new
domain, run the cheapest possible control: `budget` independent random rollouts from the start
state, no tree, no bandit, nothing carried between attempts. If your search already has a
`simulate()`/rollout function, this is a five-line ablation -- call it in a loop instead of
threading it through a tree (see `run_random_search` in `classical_mcts_blocksworld.py`,
`classical_mcts_arc_program.py`, and three more classical modules, or `search_deepsearch.py`'s
`run_search_deepsearch` for the LLM-harness version).

**If tree search loses to this control, don't conclude "tree search doesn't work here."** It means
one of two specific things is missing, and the fix follows directly from which one:

1. **Missing state-identity tracking (merging/deduplication).** If the domain has dense
   transpositions (many different action/token sequences reach the same or an equivalent state)
   and the tree has no way to recognize a revisit, UCB1 can get permanently trapped re-deriving an
   already-explored corner once one early branch looks even slightly better than its siblings --
   exactly what happened in Blocksworld and ARC. A memoryless random sampler never accumulates
   that commitment bias, so it doesn't get stuck the same way, even though it also doesn't
   dedupe. Fix: add a merge/transposition-table condition and re-run the same control -- in both
   domains tested, this closed most or all of the gap (Blocksworld's merge condition still trailed
   random at the lowest budget tested but overtook it and reached a strictly higher ceiling as
   budget grew). **This survives real (imperfect) heuristic guidance too, not just blind UCB1**:
   `REAL_HEURISTIC_MERGE_FINDINGS.md` gave Blocksworld/8-puzzle/Sokoban a real heuristic strong
   enough to beat random, then re-tested merging on top -- it still helped substantially for
   Sokoban and the 8-puzzle (Blocksworld's own result was inconclusive at small N, not a null). One
   real caveat worth calibrating for first: a deterministic heuristic value needs a much smaller
   UCB1 exploration constant than blind rollout's usual c~1.4, or the guidance gets swamped by an
   exploration bonus that was never calibrated for a noise-free reward.
2. **~~Missing real guidance~~ -- tested directly, and it did NOT fully fix this, even at its
   best possible quality.** An earlier version of this checklist inferred, from
   `DEEPSEARCH_COMPARISON_FINDINGS.md`'s baseline numbers alone (PUCT-with-a-real-projection and
   DeepSearch-style selection both scored 24-48% on ProsQA, well above blind classical baselines'
   near-zero collapse), that real guidance was a third lever preventing this failure mode. A direct
   test (`run_random_search` vs. vanilla PUCT with the real projection, same n=21 ProsQA set,
   budget=15) overturned that: tree solved 6/21 (28.6%), random solved 19/21 (90.5%), p=0.0003 -- a
   bigger gap than every classical domain except Blocksworld. A follow-up isolated *why*, swapping
   the noisy learned projection for a perfect, noise-free ground-truth oracle value
   (`search_oracle.py`): the oracle raises tree's solve rate to 47.6% (guidance quality does matter,
   p=0.0455 vs. the learned projection) but tree *still* loses to random by 47.6pp even with
   flawless guidance (p=0.0016). **Guidance quality is a real but secondary lever -- the dominant
   cause is that a small budget spent one-leaf-per-iteration across a tree is structurally worse
   than the same budget spent on fully-independent free attempts, regardless of how good the value
   estimate is.** Guidance quality is not a reliable substitute for the state-identity fix in (1)
   above, and improving it alone (however far) won't close a persistent gap caused by budget
   allocation -- check for the persistent gap even when a domain has real (even perfect) guidance,
   and if it's still there, look at how the budget is being spent, not just at the value estimator.

## The sharper version: check whether the gap persists or closes

Confirmed directly across six domains (`RANDOM_VS_MCTS_FINDINGS.md`): the random-vs-tree gap comes
in two distinct shapes, and which shape you see tells you which fix you need, before you've built
either one.

- **Persistent** (gap stays large or grows as budget grows): Blocksworld, 8-puzzle, Sokoban, ARC,
  and ProsQA (with real guidance) -- the classical four are the domains where merging shows this
  project's largest effects; ProsQA's own persistent gap under real guidance is the direct evidence
  behind item 2 above. The tree is stuck in the dense-transposition trap (or an analogous
  budget-allocation trap under real guidance, see `RANDOM_VS_MCTS_FINDINGS.md`); more budget alone
  doesn't fix it.
- **Transient** (gap is huge at very low budget, then closes rapidly): Connect Four, Morris -- this
  project's two established *null* merge-benefit domains. Here the tree isn't trapped by anything
  -- it's just ordinary bandit cold-start noise before enough samples accumulate to stabilize
  UCB1's Q-estimates, and it resolves on its own well before any interesting budget range. Merging
  correctly has nothing to fix here, because there's no redundant-transposition problem to fix.

Practically: run the random-restart control at a *few* budgets spanning small to the range you
actually care about. If the gap is closing fast, don't bother building a merge condition -- it
won't help, for the same reason it doesn't help in Connect Four or Morris. If the gap holds flat
or grows, that's a strong, cheap signal that merging is worth building.

## What this doesn't mean

- It doesn't mean random restarts are a good *final* answer -- in every domain tested, the best
  condition (merge, or merge combined with better guidance) beat pure random, sometimes by a lot.
  Random restarts are a diagnostic floor, not a target.
- It doesn't mean this generalizes to every possible search domain -- six domains, all blind
  (no heuristic), all previously characterized as large-merge-benefit or null-merge-benefit. The
  check is cheap enough to run on a new domain directly rather than assuming the pattern transfers.

## The one-line version

**A tree search with no way to recognize a state it has already found, and no real guidance
telling it which branch is actually better, is not obviously better than guessing randomly and
trying again -- benchmark against that guess before trusting the tree.**
