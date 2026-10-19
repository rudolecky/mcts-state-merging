# Final Summary: Does Merging Help Search?

## The one-paragraph answer

Merging semantically-equivalent search states helps tree search in proportion to how much
redundant re-exploration of the same state the search's own expansion mechanism generates under
its selection policy -- never as a generic "graph search beats tree search" fact. This was first
found, accidentally, in real LLM-guided search (independent K-candidate sampling wastes budget on
near-duplicate continuations); it was then reproduced, dissected, and directly manipulated in a
completely separate, classical, no-LLM setting, arriving at the same structural explanation from
the opposite direction. Two independent lines of evidence converging on one mechanism is the
strongest form of confirmation this investigation could have produced, and it's what happened.

## Phase 0: does a distance in hidden-state space predict value similarity? (`PHASE0_*`, `PHASE0_LEARNED_PROJECTION_FINDINGS.md`)

- **H1 (raw distance, no learning) failed everywhere it was tested** -- cosine/L2 distance
  between hidden states never predicted `|ΔV|` above noise (Qwen/Countdown final ρ=0.132
  [-0.059, 0.253]; Qwen/ProsQA mid ρ=0.163 [-0.071, 0.462]; R1-Distill/ProsQA ρ=0.068
  [-0.212, 0.320] -- every CI spans zero).
- **A learned linear probe (RidgeCV, leave-one-problem-out) found a real, robust within-problem
  signal on Qwen2.5-1.5B-Instruct**: ρ 0.44-0.55, p<0.001 in 5/6 layer×dataset configs (best:
  ProsQA/mid ρ=0.550, Countdown/final ρ=0.455), replicated as the out-of-fold projection used
  for the rest of the investigation (Countdown 3/4 ρ=0.457, CI [0.330, 0.578] at n=22 problems).
  Step-depth confound explicitly ruled out.
- **DeepSeek-R1-Distill-Qwen-1.5B never replicated this signal**, at any scale tested (up to
  n=11 problems/147 pairs) -- a real, checked negative, not an underpowered null. The signal is
  model-class-specific, not universal.
- **Connect Four's own ground-truth transposition check (exact board-state equality, not a value
  proxy) found the learned projection gave no detectable advantage over untrained raw distance**
  for merge *detection* specifically (both hit recall=1.000, near-identical precision) -- the
  opposite of the value-prediction finding, and a real nuance worth keeping: the projection helps
  predict value similarity, not necessarily state-equivalence detection.

## H3: does merging improve search accuracy at matched budget, with an LLM? (`H3_MERGE_SEARCH_FINDINGS.md`)

- **Countdown num_numbers=3: null** (13/40 vs 14/40, McNemar p=1.00) despite real, large
  deduplication (39.2→9.5 nodes). Diagnosed as a **model-capability ceiling**, not a
  search-efficiency shortfall -- 90% of jointly-unsolved problems were 0/8 on raw,
  search-independent sampling.
- **Countdown num_numbers=4 (bigger model, Qwen2.5-7B-Instruct): a real, strengthening positive
  effect** -- 20%→70% at n=10 (p=0.125, underpowered) to 14.3%→71.4% at n=14 (p=0.0215) to
  15.8%→68.4% at n=19 (p=0.0129), significance *strengthening* with every batch added, the
  opposite trajectory from a small-n fluke.
- **ProsQA (structurally unrelated task, same model): the cleanest result in the whole
  investigation** -- 28.6%→81.0% at n=21, McNemar p=0.0010, every single discordant problem
  favoring treatment.
- **Connect Four (LLM): a third confirmed H3 domain**, closing what was originally left an open
  n=3 data point -- scaled to n=18 across three batches (`CONNECT_FOUR_FINDINGS.md`'s Stage 4,
  scaled): 16.7%→61.1%, McNemar p=0.0215, CI [0.167, 0.722], 1 vs 9 discordant (heavily
  one-directional, ProsQA's own pattern), 78% node reduction -- matching the other two confirmed
  domains' magnitude closely. This is the predicted outcome from
  `CONNECT_FOUR_KSAMPLE_FINDINGS.md`'s classical-arc finding, not an independent coincidence: the
  LLM harness's batched-K-candidate expansion lacks exactly the structural anti-duplicate
  guarantee that made Connect Four's *classical* search null.
- **C6 ablation (swap PUCT's value source from the shared noisy projection to honest sampled
  rollouts) split Countdown-4 and ProsQA apart**: Countdown-4's advantage **vanished** (exact
  tie, 7/10 vs 7/10) -- its redundancy was guidance-driven. ProsQA's **survived and strengthened**
  (81.0%→90.5%) -- its redundancy was task-structural. Two different mechanisms behind two
  superficially similar "high dedup ratio, large effect" results.
- Entropy-stratified false-merge check closed the standing H5 gate: false merges concentrate in
  *low*-entropy states on both tasks, not high-entropy/ambiguous ones -- the opposite of the
  feared failure mode.

## GSM8K: a real benchmark, closed with a definitive null (`GSM8K_FINDINGS.md`)

The within-problem value probe never validated on GSM8K at any scale tried (n=15→26,
plain-instruct or native-boundary reasoning-RL harness variants), collapsing from a
promising-looking p=0.09 at n=22 to decisive noise at n=26. Closed as a real, well-evidenced
negative specific to this benchmark/model combination -- H3 search was never attempted here,
since the prerequisite probe validation never cleared.

## The classical (no-LLM) arc: isolating "does merging help search" from "does the learned metric work" (`CLASSICAL_MCTS_SUMMARY.md` and its four domain docs)

Every H3 result above entangled the merge criterion (a learned projection) with the search
mechanism. Four classical UCB1-MCTS domains -- exact-state-equality merging, honest random
rollouts, zero learned anything -- isolated the search question alone:

| Domain | Structure | Result |
|---|---|---|
| Connect Four (`CLASSICAL_MCTS_FINDINGS.md`) | irreversible, adversarial | Null |
| 8-puzzle (`SLIDING_PUZZLE_FINDINGS.md`) | reversible, single-agent | Large positive |
| Three Men's Morris (`MORRIS_FINDINGS.md`) | reversible, adversarial | Null (different reason) |
| Sokoban (`SOKOBAN_FINDINGS.md`) | partially reversible, single-agent | Large positive (≈8-puzzle scale) |

Mechanism, refined across all four: merging helps in proportion to how much of the search's
*actual* exploration activity revisits already-seen states under UCB1 -- not reversibility
itself (Morris disproves that), not move-type ratio (Sokoban disproves that too). Connect Four's
`untried_moves` bookkeeping structurally prevents redundant exploration; Morris's adversarial
interleaving makes independently-reaching the same joint state rare even with reversible moves;
the 8-puzzle's single mobile token and Sokoban's walk-dominated branching both generate dense
redundant re-exploration for merging to reclaim.

## Closing the loop: does the classical mechanism explain the original LLM results? (`CROSS_DOMAIN_MECHANISM_CHECK.md`)

Partially, and the failure mode is itself informative. Countdown num_numbers=3's high dedup
ratio (76%, comparable to num_numbers=4's and ProsQA's 82%) initially looks like it should
predict an effect the classical pattern doesn't -- but it doesn't, because of the model-capability
ceiling, an axis with no classical-domain equivalent (classical games are always eventually
solvable by enough budget; LLM generation capability is a genuinely separate constraint). **Ran
the C6 rollout-guidance ablation on num_numbers=3 directly** (never previously tested): stayed
null under honest guidance too (2/10 vs 2/10, dedup unchanged at 74%) -- confirming the
capability-ceiling diagnosis by direct manipulation, not just by elimination.

## The capstone: manipulating the mechanism directly, in the classical setting

Two further, purpose-built experiments tested the theory's central claim by intervention rather
than observation:

- **`CONNECT_FOUR_GUIDANCE_ABLATION_FINDINGS.md`**: degraded plain classical MCTS's value
  estimate (short-capped rollout, neutral-on-exhaustion) across a full budget×cap grid. Every
  single cell was an exact tie -- Connect Four's null is completely guidance-independent, because
  `untried_moves` gives baseline a *structural* guarantee against duplicate nodes that no amount
  of guidance degradation can touch.
- **`CONNECT_FOUR_KSAMPLE_FINDINGS.md`**: removed that structural guarantee on purpose (K
  independent draws with replacement per expansion, mirroring the LLM harness's real mechanism,
  baseline never deduplicating even within one batch). The same domain, same puzzles, now showed
  a real, monotonic, guidance-*dependent* merge benefit (0.117→0.142→0.198 as guidance degraded
  at budget=3) -- confirming by direct manipulation, not just inference, that guidance quality
  only matters when the expansion mechanism itself lacks a structural anti-duplicate guarantee.
- **`MORRIS_KSAMPLE_FINDINGS.md`**: replicated the K-sample manipulation on Morris. The core
  claim generalized (Morris's null flipped too, k=3/k=5 baseline stuck at 0.000 vs. treatment
  reaching 0.34-0.36). The guidance-dependence piece did **not** replicate -- effect size stayed
  flat across guidance quality, plausibly because Morris's small branching factor (2-4 legal
  moves per position) generates collisions by raw combinatorics rather than by where the search's
  attention wanders, leaving guidance quality nothing to modulate. Reported as a genuine partial
  replication, not smoothed into false symmetry with Connect Four's result.

## What's actually settled

1. A learned value-similarity probe works robustly on Qwen2.5-1.5B/7B-Instruct across two
   structurally different tasks; it does not replicate on DeepSeek-R1-Distill-Qwen-1.5B.
2. Merging improves LLM-guided search accuracy at matched budget, confirmed at trust-bar scale on
   two tasks (Countdown-4, ProsQA), through two independently-diagnosed mechanisms (guidance-noise
   compensation vs. task-structural redundancy).
3. A model-capability ceiling is a real, separate axis from search-efficiency, capable of
   producing a null regardless of how much redundancy exists to reclaim -- demonstrated by direct
   manipulation on Countdown num_numbers=3.
4. In classical, no-LLM search, merging's benefit tracks redundant-exploration density under
   UCB1's own selection policy, not reversibility or task type directly.
5. Guidance quality and redundancy density are not independent variables -- guidance is a
   modulator that only has something to modulate when the expansion mechanism itself lacks a
   structural anti-duplicate guarantee, demonstrated by direct manipulation in two domains, with
   the modulation itself confirmed to generalize only partially across domains.

## What's genuinely still open

- Why R1-Distill-class models don't show the value-probe signal at all, rather than showing a
  weaker version of it -- a deep interpretability question (hidden-state geometry under
  reasoning-RL training), not something a follow-up experiment of this investigation's scale
  could resolve.
## Points closed after this document's first draft

- **Connect Four's LLM-based H3 result**: scaled from the original uninformative n=3 to n=18
  across two new batches (`CONNECT_FOUR_FINDINGS.md`'s "Stage 4, scaled" section) -- McNemar
  p=0.0215, CI [0.167, 0.722], now a third confirmed H3 domain alongside Countdown num_numbers=4
  and ProsQA, and the predicted outcome from the classical-arc mechanism finding rather than an
  independently-surprising result.
- **Morris's branching-factor hypothesis** (why guidance-quality modulation didn't replicate
  there): tested directly by narrowing Connect Four's own board (width 5→4, no new code) and
  re-running the same guidance sweep. Found a genuine gradient, not a sharp threshold: branching
  factor 5-7 (original Connect Four) gives strong, clean guidance-dependence; ~4 (narrowed
  Connect Four) gives weak, budget-dependent guidance-sensitivity; 2-4 (Morris) gives none. A
  real, if imperfect, partial confirmation -- see `MORRIS_KSAMPLE_FINDINGS.md`'s follow-up
  section.
- **GSM8K's probe failure -- benchmark vs. model class**: turns out to already be closed, not
  open -- re-reading `GSM8K_FINDINGS.md` confirms the plain Step-N harness *was* tested on
  Qwen2.5-7B-Instruct specifically (the exact model that worked for Countdown num_numbers=4 and
  ProsQA), through this project's own n=21 trust bar, and it collapsed there too (the same
  significant-then-evaporating trajectory as every other false alarm in this investigation).
  DeepSeek-R1-Distill-Qwen-1.5B/7B were *also* tried (native-boundary harness, pushed to n=26).
  **No configuration on any of three models, across two harness variants, ever produced a
  surviving signal** -- this rules out "wrong model class" specifically, since the proven-working
  model failed too, and leaves GSM8K itself (or some shared factor this investigation hasn't
  identified) as the more likely explanation, not model choice. No new experiment needed; this
  was a documentation gap in this file's first draft, not a real open question.

## The final capstone: a true self-play AlphaZero clone (`ALPHAZERO_CONNECT_FOUR_FINDINGS.md`)

One genuinely new question emerged after this document's first draft: every PUCT-based result in
this project used either a degenerate uniform-1/K prior (the LLM harness) or classical UCB1 (no
prior at all). Does merging help once the prior comes from a *real* trained policy network, the
way actual AlphaZero works? Built one from scratch on Connect Four -- a small CNN, real self-play
(3,000 games, 120 iterations, ~40 minutes), real PUCT (no rollout, network evaluates leaves
directly), trained to a clean, stable convergence (loss 1.36→~0.91, win-rate-vs-random and
tight-budget puzzle-solving both plateauing alongside it, no instability anywhere in the run --
genuinely uncommon for a from-scratch RL build). Then ran the actual merge-vs-tree comparison on
the trained network, same k_plies puzzle set and statistical design as every classical Connect
Four result in this project.

**Exact tie at every single budget swept, floor to ceiling** (0/40 at budget 2 through 40/40 by
budget 25) -- with real, growing merging activity in the node counts throughout (up to 8% fewer
at budget 100), the identical "dedup happens, doesn't help" signature as classical UCB1's own
Connect Four null. This was the *predicted* outcome, not a fresh discovery: real PUCT's
eager-per-node child creation gives every legal move exactly one edge, the same "tried at most
once" structural guarantee classical UCB1 has, just enforced for a different reason (the prior
already ranks untried moves). The prior's *source* -- uniform, frozen LLM projection, or a
genuinely trained self-play network -- was never the load-bearing variable anywhere in this
investigation; the expansion mechanism's structural relationship to redundant exploration is, and
real AlphaZero-style PUCT shares Connect Four's classical structure on exactly that dimension.

## Extension: Blocksworld, testing against the planning literature (`BLOCKSWORLD_FINDINGS.md`)

Reviewed arXiv:2405.18248 ("Extreme Value Monte Carlo Tree Search for Classical Planning") at the
user's request. Its contribution -- a tighter bandit confidence bound (UCB1-Uniform, derived from
Extreme Value Theory rather than Hoeffding's generic bound) for classical STRIPS planning MCTS --
never touches merging or transposition tables; every variant it tests still expands one child per
action from a plain tree open list, the same structural anti-duplicate guarantee already found not
to be the load-bearing variable anywhere in this project. But the paper surfaced a genuinely new
domain family: built **Blocksworld** on top of **Pyperplan** (the exact planner the paper itself
uses), reusing its `Task`/`Operator`/`breadth_first_search` primitives directly rather than a
hand-rolled engine -- a real test against that paper's own literature context, not just an
analogous domain.

**The largest, cleanest effect of any domain in this project.** Full sweep (45 puzzles across
plan lengths 8/10/12 at 5 blocks, budgets 200-3000, 50 repeats/condition): baseline climbs only to
25.5% solved even at the top of the range, while treatment reaches 100%. At plan lengths 10 and
12 specifically, baseline essentially *never* solves across the entire swept range (0.000 through
budget=1500, still ≤0.019 at budget=3000) while treatment reaches 1.000 by budget=3000. Node
reduction grows from 43% at the lowest budget to 85% at the highest, never plateauing within the
range tested. Mechanism: at 5 blocks, several blocks are simultaneously relocatable at once, and
every independent pair of relocations commutes -- so the number of distinct move orders reaching
the same resulting state grows combinatorially with how many blocks are "in play," generating
denser redundant re-exploration than any single-token domain (8-puzzle) or single-box domain
(Sokoban) tested so far. Confirms and sharpens the standing mechanism rather than surprising it:
guidance-bound quality (this paper's whole contribution) and merging (this project's question)
are answers to two different, orthogonal questions -- which child to visit next, versus whether a
child gets created at all when its target state has already been seen.

## Extension: GUCT-Uniform, testing the paper's own algorithm (`BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`)

The Blocksworld extension above used this project's own blind UCB1-MCTS, not arXiv:2405.18248's
actual algorithm -- a fair objection, since a blind, rollout-based, Monte-Carlo-average-backup
search and a heuristic-guided, Full-Bellman-backed one are different planners independent of
merging entirely. Built **GUCT-Uniform** itself (the paper's best performer in Table 1) inside
this project's harness -- LCB1-Uniform's exact bandit formula (Theorem 6), Full Bellman backup
(Definition 3), and `hFFHeuristic` (pyperplan's own class, the exact heuristic the paper uses) as
the reward, no rollout -- and ran the merge-vs-tree A/B under it directly.

**The largest effect size and the flattest baseline of any domain in this project** -- two
separate claims, both checked directly against every other findings doc's own numbers, not
recalled from memory. Largest: peak diff_mean here is +0.954 (budget=4000), vs. +0.770 for blind
Blocksworld, +0.761 for Sokoban, +0.558 for the 8-puzzle. Flattest: across the entire 40x budget
range tested (100-4000) and every plan length, baseline never exceeds 5% solved and never climbs
with more budget -- every other domain's baseline visibly improves given more budget (e.g. blind
Blocksworld's 0.001->0.255, Sokoban's 0.023->0.517); this one doesn't move at all. Treatment
reached 98-99% by budget=4000. (Different puzzle instances from the blind-MCTS Blocksworld run
above -- 6 blocks/plan-lengths 10-14 here vs. 5 blocks/8-12 there, recalibrated because heuristic
guidance solves the smaller puzzles almost trivially; see `BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`'s
own "Scope and limitations" section for exactly what is and isn't verified about this comparison.) Mechanism: Full Bellman backup
means a merged node's value is the *minimum* ever found through any arriving path, so every
transposition directly sharpens one shared estimate; the untreated tree instead pays for an
isolated, from-scratch subtree per transposition, and because LCB1-Uniform has no self-limiting
exploration term (a deliberate design choice in the paper, for planning's "escape the plateau
quickly" use case), it doesn't even distribute budget broadly across those wasted subtrees the way
UCB1 would -- it tunnels down one lane and often never recovers. Two real bugs were found and
fixed while building this: naively reusing a frozen resample value on a cycle-block or a
confirmed-terminal (safe under every other module's self-limiting UCB1, catastrophic under
LCB1-Uniform's divergent one) had to be replaced with permanently closing that edge instead. This
directly answers the scoping question motivating the whole extension: guidance-bound quality and
merging are orthogonal questions, and the paper's own best, most theoretically-justified algorithm
shows the largest merging benefit found anywhere in this project, not the smallest.

## Extension: a real IPC instance where the paper's own algorithm just fails (`TABLE1_REPRODUCTION_FINDINGS.md`)

Asked to reproduce the paper's Table 1; scoped to a minimal slice -- GBFS, plain GUCT, and
GUCT-Uniform on real (not project-generated) IPC Blocksworld and Gripper instances, sourced from
the standard `downward-benchmarks` repository, under the paper's own protocol (10,000
node-evaluation cap, 5 seeds, `h^FF`). The ranking didn't reproduce, in an informative way: on
Blocksworld, tree-mode GUCT-Uniform (exactly the paper's own algorithm -- it never merges) solved
almost nothing (1/25 seed-runs) while GBFS and even plain GUCT clearly beat it -- the opposite of
the paper's own reported ranking. On Gripper, the paper's claimed advantage showed up cleanly
(GUCT-Uniform solved 35/35, including two instances plain GUCT failed completely). Same
implementation, same protocol, opposite outcome by domain.

**Enabling merging alone, no other change, took GUCT-Uniform from 4% to 100% solved on the exact
same Blocksworld instances** (confirmed down to a budget as low as 100 evaluations; confirmed
tree-mode still hadn't solved the smallest instance even at 50,000 evaluations, 5x the paper's own
cap, while GBFS solves it in 24). Mechanism: GBFS deduplicates visited states as a matter of
ordinary definition; tree-mode GUCT-Uniform, confirmed from the paper's own text, never does --
combined with LCB1-Uniform's already-confirmed lack of a self-limiting exploration term, it can
commit hard to one narrow branch and keep re-deriving the same small cluster of already-visited
states as "new" nodes indefinitely, on a domain -- Blocksworld -- whose reversible, densely-
transposing structure is exactly what this project's other Blocksworld findings already
identified as the mechanism merging reclaims. This is the sharpest version of the whole project's
thesis found anywhere: not a modest efficiency gain, but the difference between an algorithm that
essentially never solves a real, easy benchmark instance and one that solves it every time --
surfacing unprompted on real, externally-sourced data while attempting to reproduce someone else's
result, not from a domain this project constructed to make the point.

## Extension: program-synthesis search on real ARC-AGI-1 (`ARC_AGI_FINDINGS.md`)

Asked whether ARC-AGI is a domain where the merge-vs-tree question even applies. It does in
principle (DSL program synthesis over grids has the same "many orderings reach the same
intermediate value" structure as Blocksworld), but it's a genuinely different *kind* of search --
a program-synthesis dataflow graph, not physical state-transition planning -- built through three
scope revisions once real data ruled out easier framings: a from-scratch minimal DSL solved 0/60
real tasks (an expressiveness ceiling); the real, vendored `arc-dsl` (160 primitives, MIT-licensed)
restricted to simple chains reached only 18/400; full multi-variable dataflow over a curated,
frequency-selected 60-function subset -- including the higher-order combinators (`fork`,
`compose`, `lbind`, ...) that dominate most real solutions -- was what got built and run.

**Result: a real but small, non-significant effect** (diff_mean +0.000/+0.042/+0.067 across
budgets 500/2000/5000, all CIs crossing or touching zero), with modest, consistent node-count
reduction (11-13%) confirming genuine deduplication without a large solve-rate payoff at this N=12.
The aggregate hides real structure: 5 of 12 tasks never solve under either condition (too hard for
blind search), 2 solve at ceiling from the smallest budget (trivially easy), and among the
remaining 5 "informative middle" tasks the pattern is a real, sometimes large, growing advantage
for merging (two tasks reaching +40 percentage points by the highest budget). Mechanism: unlike a
physical world-state, a program-synthesis context's canonical merge key needs the *entire* set of
available values to coincide (a looser "match only the latest value" key would silently discard a
path's distinct reusable byproducts) -- a real, structural reason full transpositions are rarer
here than in Blocksworld's independent-block-relocations, on top of blind search's floor/ceiling
stratification diluting the aggregate signal.

**A second, more surprising finding emerged from asking how much MCTS itself contributes, separate
from merging**: an ablation firing `budget` independent random rollouts with no tree and no UCB1
at all (reusing the exact same rollout MCTS's own simulate step uses) beat tree-mode MCTS at every
budget tested (0.242 vs. 0.208 at budget 500, 0.442 vs. 0.250 at 2000, 0.508 vs. 0.358 at 5000) --
and beat the merge condition too (0.508 vs. 0.383 at 5000). In a search space where the reward is
binary and essentially uncorrelated between neighboring programs (changing one DSL primitive
produces an unrelated result, not a slightly-better one), UCB1's premise -- concentrate budget on
what looks promising -- backfires: one early lucky sample makes a prefix look artificially good,
and the tree commits further budget to refining that one branch instead of spending it on the
diversity of fully independent attempts pure random restarts get for free. Both MCTS variants here
are instances of a strategy that's already behind the simplest alternative, so the merge-vs-tree
result above should be read as "does merging make the weaker of two options a little less weak,"
not "does it improve on the best available approach."

Two substantive engineering bugs were found and fixed while building this, not glossed over: a
cycle-safety self-loop specific to set-based merge keys (a move whose result duplicates an
existing context value doesn't grow the merge key at all, making the "child" identical to its own
parent -- confirmed via a hung run traced with SIGINT), and a grid-size scaling blowup (one
30x30-grid task drove an early run to 9.5GB/97 CPU-minutes before being killed and the task
selection filtered to <=15x15 grids). This is the first domain in this project whose result
doesn't cleanly resemble any prior one -- reported as a real, modest, structurally-explained
finding, not forced into a larger claim.

## Extension: merging vs. a real competing algorithm on a shared benchmark (`DEEPSEARCH_COMPARISON_FINDINGS.md`)

Asked to compare this project's own MCTS against another algorithm (not just vanilla) on a shared
benchmark, and to review arXiv:2509.25454 ("DeepSearch," ICLR 2026). It embeds MCTS into LLM RL
*training*, contributing global frontier selection (score every unexpanded node across the whole
tree at once, not root-to-leaf UCT) and a depth-decayed, sign-constrained backup rule -- and, like
every other paper reviewed here, never merges states; its own stated motivation for frontier
selection is "eliminating redundant root-to-leaf traversals," addressing redundancy through
selection, not state identity. Ported the search mechanism alone (not the RL training loop, which
needs a 128xH100 cluster) into this project's existing frozen-LLM harness and ran it against
vanilla PUCT, merge vs. tree, on the exact 21 real ProsQA instances from this project's original
H3 trust-bar result.

**Merging helps both algorithms, significantly** (vanilla PUCT: 0.238->0.810, p=0.0005; DeepSearch-
style: 0.476->0.762, p=0.0339) **but by very different margins.** DeepSearch's own global frontier
selection genuinely raises its untreated baseline well above vanilla PUCT's (0.476 vs. 0.238) --
its central claim holds up under independent reimplementation -- but its merge-enabled ceiling
(0.762) is slightly below vanilla PUCT's (0.810), it costs more wall-clock per problem in the
merge condition (123.3s vs. 85.6s), and it produced this project's only non-monotonic result on
ProsQA (one problem flips solved->unsolved under merging, versus a perfectly one-directional
12-flips-the-other-way for vanilla PUCT). Mechanism: selection quality and state-identity merging
address different sources of wasted search effort with only partial overlap -- smarter selection
can reduce how much a search wastes on unproductive territory, but doesn't stop two different
reasoning paths that reach an equivalent state from being represented as separate nodes, which is
exactly what merging reclaims regardless of how good the selection policy already is. The clearest
demonstration yet that guidance/selection quality and merging are separate, non-competing levers.

## Extension: does tree-guided search even help? Random restarts vs. tree-mode MCTS, seven domains (`RANDOM_VS_MCTS_FINDINGS.md`, `RANDOM_BASELINE_CHECKLIST.md`)

The ARC pilot's most surprising result -- pure random restarts (no tree, no UCB1) beating
tree-mode MCTS at every budget -- was tested for generalization across every classical domain in
this project: Blocksworld, the 8-puzzle, and Sokoban (established large-merge-benefit domains),
plus Connect Four and Morris (established *null*-merge-benefit domains). **The pattern splits
exactly along the existing merge-benefit line.** Blocksworld/8-puzzle/Sokoban/ARC all show a
**persistent** tree-vs-random gap that holds or grows across the whole budget range tested
(Blocksworld: 0.115->0.255 vs. 0.749->0.926, a 63-67pp gap even at budget 3000; 8-puzzle and
Sokoban similarly 54-63pp, barely closing by budget 1200). Connect Four and Morris instead show a
**transient** gap -- enormous at very low budget (up to 99+ points) but closing rapidly, with
Connect Four's gap no longer statistically distinguishable from zero by budget 50.

This is the cleanest unifying result in the whole project: **whether the random-vs-tree gap
persists or closes is an independent, cheap diagnostic for whether merging will help, discoverable
without building any merge machinery at all.** A persistent gap means blind tree-mode UCB1 is
stuck in the dense-transposition trap merging exists to fix (once one early branch looks slightly
better, UCB1 commits further budget to it, and in these domains that mostly means re-deriving
already-explored states via combinatorially many alternate paths -- exactly what a memoryless
random sampler never gets stuck in). A transient gap is just ordinary bandit cold-start noise
before Q-estimates stabilize, with no redundant-transposition problem underneath it for merging to
fix -- consistent with Connect Four's and Morris's own established null results.
`RANDOM_BASELINE_CHECKLIST.md` turns this into a practical recommendation: benchmark a new domain
against random restarts at a few budgets before building anything else -- persistent gap, build
merge; closing gap, don't bother.

A follow-up closed the one inference this doc originally left unverified: does real guidance
prevent the persistent gap? `DEEPSEARCH_COMPARISON_FINDINGS.md`'s baseline numbers alone (vanilla
PUCT and DeepSearch-style selection both scoring 24-48% on ProsQA, not near-zero) had suggested
real guidance was an independent third lever. A direct test -- `run_random_search` added to
`search.py`, matched against vanilla PUCT with the project's real learned projection, same n=21
ProsQA set, budget=15 -- overturned that: tree solved 6/21 (28.6%), pure random solved 19/21
(90.5%), p=0.0003, a bigger gap than every classical domain except Blocksworld, and random beat
tree on every problem tree also solved (0 tree-only wins). **Real guidance does not, by itself,
prevent this failure mode.**

A second follow-up (`search_oracle.py`) isolated why, by replacing the noisy learned projection
with a perfect, noise-free ground-truth value computed directly from a ProsQA instance's own known
fact graph -- no LLM, no training data, no possible noise. Guidance quality turned out to be a real
but secondary lever: the oracle raised tree's solve rate from 28.6% to 47.6% (p=0.0455, oracle never
lost to the learned projection on any problem), but tree still lost to random by 47.6 points even
with flawless guidance (p=0.0016). **The dominant cause is budget allocation, not guidance
quality**: a small budget spent one-leaf-per-iteration across a multi-step tree is structurally
worse than the same budget spent on fully-independent free attempts, however accurately each node's
value is estimated. The practical upshot: check for the persistent gap even when a domain has real
(even perfect) guidance -- and if it's still there, look at how the budget gets spent, not just at
the value estimator.

## Extension: does merging still help once tree search has real guidance strong enough to beat random? (`REAL_HEURISTIC_MERGE_FINDINGS.md`)

The Connect Four AlphaZero result showed merging can stay null even with a real trained network,
for a structural reason specific to that domain (real PUCT's expansion never creates duplicate
nodes). That doesn't transfer to Blocksworld, the 8-puzzle, or Sokoban -- the domains where merging
matters most -- whose blind-UCB1 expansion has no such guarantee. This gave all three a real
heuristic (Manhattan distance, a box-goal-plus-player-distance heuristic, and `pyperplan`'s
`hFFHeuristic` respectively) strong enough to make tree search actually beat random restarts, then
re-tested whether merging still helps on top.

**Two calibration findings shaped every result**: a deterministic heuristic value needs a far
smaller UCB1 exploration constant than blind rollout's usual c~1.4 (its exploration bonus is
calibrated for noisy rewards; at c=1.4 the 8-puzzle's heuristic-guided tree solved 0/30 on an
8-move puzzle, budget=40, vs. 28-35/50 at c=0.02-0.1, same budget); and Sokoban's initial box-goal-
only heuristic gave zero discrimination among purely-walking moves, solving 0/50 across a full
calibration sweep until a player-to-box distance term was added.

**Results split by domain.** Sokoban is the cleanest: heuristic-guided tree decisively beats random
from budget~=80 on (p=0.008 repeatedly), and merging then adds a further significant gain on top
(p=0.008-0.031 at several budgets). The 8-puzzle shows the same pattern in direction and magnitude
(merging adds +0.16 to +0.51 solve-rate) but stays short of significance at n=8 puzzles/tier.
Blocksworld's Stage 1 (does guidance beat random) works exactly as well as the other two domains,
but its Stage 2 (does merging still help) is genuinely inconclusive at n=7-8 -- not a confirmed
null, an open question pending more data. A bonus comparison benchmarked `guct_uniform_blocksworld.py`'s
own real heuristic-guided algorithm (LCB1-Uniform, Full-Bellman backup, a genuine relaxed-plan
heuristic) against random restart for the first time: its solve rate stays flat at ~0.03 across a
40x budget range, and **random restart eventually overtakes it outright** (budget=4000, p=0.036) --
the sharpest demonstration yet that no amount of algorithmic sophistication guarantees beating
random. **Merging's benefit on top of real guidance is domain-dependent, not automatic and not
automatically absent** -- check per domain, the same way every other lesson in this project says to.

## Extension: merging catastrophically harms a dense domain, then a backup-rule redesign fixes it (`RUBIKS_CUBE_FINDINGS.md`)

Asked what other multistep benchmark to try, and built a full-fidelity 2x2x2 Rubik's Cube: exact
3,674,160-reachable-state rotation-group representation (every corner's orientation tracked as a
full element of the 24-element cube rotation group, not a hand-derived twist-count, so correctness
reduces to checkable group axioms), verified via a full BFS matching the known state count exactly
and, as a bonus independent check, a diameter of exactly 14 -- the published quarter-turn-metric
god's number for this puzzle. A diagnosed dead end along the way: simple per-piece heuristics
(misplaced-corner count, even a richer per-corner-alone-distance sum) saturate almost immediately
(mean value flat from true distance 4 through 14) because a single quarter turn moves 4 of 8
corners at once -- a real structural fact about this puzzle family, not a calibration failure, and
exactly why real cube solvers need multi-piece pattern databases. All results use the exact
BFS-distance oracle as guidance instead.

Oracle-guided tree beats random as decisively as anywhere in this project (reaching a perfect 1.000
at every difficulty tier tested, given enough budget, while random caps near 0.05). But merging
didn't just fail to help on top of that -- it was **exactly 0.000 at every single budget and
difficulty tested, including every point where tree-mode reached 1.000** (p=0.008 repeatedly, the
floor p-value at n=8). Tracing a single run showed why: merge-mode created far fewer nodes than
tree-mode for the same budget (159 vs. 401) but reached worse progress (best distance-to-goal 8 vs.
4) -- confirming heavy merging was genuinely happening, and that it hurt. **This is a third
mechanism, distinct from both "merging helps" (the classical domains) and "merging is structurally
null" (Connect Four's real PUCT)**: path-keyed tree search, because it can never collide on a
node, guarantees every expansion pushes the search frontier outward even when much of that growth
is semantically redundant -- and in a domain transposition-dense enough, that guaranteed forward
motion is what keeps the search moving at all. Merging collapses redundant growth into a small,
heavily-revisited hub of nodes near the start, and removing "redundant" node creation removes the
very thing driving the search forward, rather than removing something holding it back.

Two independent threshold-based mitigations were tried next -- capping how many parents, and
separately how many visits, a merged node could absorb -- and both hit an identical all-or-nothing
cliff: a cap of exactly 1 "recovered" performance only by preventing any real pooling from ever
happening, while any cap permitting genuine pooling reproduced the full 0.000 failure at
undiminished severity, even a 25x-more-generous visit cap. This ruled out "too much pooling" as the
mechanism and pointed at the backup rule itself: MC-average pooling was diluting a signal, not
merely firing too often. Porting `guct_uniform_blocksworld.py`'s exact algorithm (Full Bellman
min-backup + LCB1-Uniform selection, already proven to help merging on Blocksworld) to the cube
confirmed this directly: **merge and tree became statistically indistinguishable in solve rate at
every single budget and difficulty tested** (`merge_diff` exactly 0.000, CI=[0.000,0.000],
everywhere), while merge still delivered a consistent ~57% node-count reduction over tree mode at
matched (perfect) solve rate -- the classic dedup efficiency benefit, now showing up cleanly with
zero cost. With an exact oracle, a node's value is already correct on its first visit, so Full
Bellman's min-aggregation converges identically whether one parent visits a state once or five
parents visit it collectively -- there's no noisy signal left for merging to either dilute (as
MC-averaging did) or improve, only search-cost efficiency to reclaim, which it does. Pushed to the
hardest instances this domain has (td=12 and td=14, the exact diameter) at the sharpest
floor-to-ceiling budget transition: still exactly identical.

**Correction**: this was initially written up as an algorithm-level guarantee -- "the one function
that decides where the search goes next never reads a shared child's pooled value, so merging's
efficiency gain can never become a solve-rate gain, on any domain." That overreached. It's true that
`_lcb1_uniform_score` is edge-local, so at any single node merging can't change which child gets
picked -- but this project's own `guct_uniform_blocksworld.py` (the very module this redesign was
ported *from*, referenced two paragraphs up) already has an established result contradicting the
"never, on any domain" version of the claim: tree-mode GUCT-Uniform on Blocksworld stays flat at
0.030-0.050 across a 40x budget range while merge climbs to 98.7% by budget=4000
(`BLOCKSWORLD_GUCT_UNIFORM_FINDINGS.md`, n=30, p<0.0001 throughout) -- the identical algorithm,
a massive solve-rate gap. Porting the same algorithm to the 8-puzzle and Sokoban confirmed parity
there too (`REAL_HEURISTIC_MERGE_FINDINGS.md`: diff exactly 0.000 at all 16 budget points, merge
using 20-35% fewer nodes), so cube/8-puzzle/Sokoban really do tie -- but Blocksworld proves that
outcome isn't a law the algorithm's code enforces everywhere.

**The corrected, complete picture**: merging is never *harmful* under GUCT-Uniform, on any of the
four domains tested. Whether it adds solve-rate *benefit* on top of that is a separate, genuinely
domain-dependent question -- it does exactly when tree mode's own unmerged, redundant growth gets
trapped and can't reach the goal within budget on its own (Blocksworld, whose dense transpositions
combined with GUCT-Uniform's deliberately depth-first selection commit the search hard to one bad
lane), and it doesn't when tree mode was never actually failing (cube, 8-puzzle, Sokoban at the
settings tested, where tree reaches the same ceiling merge does, just less efficiently). This is
the same persistent-vs-transient gap distinction `RANDOM_VS_MCTS_FINDINGS.md` already established
for blind UCB1 search -- it turns out to predict GUCT-Uniform's own tree-vs-merge answer too.

A fifth domain, real ARC-AGI tasks (`REAL_HEURISTIC_MERGE_FINDINGS.md`), needed a genuinely new
heuristic first (no exact oracle is feasible over an unbounded DSL program space, and no heuristic
existed anywhere in this project for this domain until now: fraction of output grid cells matching
the target, restricted to exactly the value `is_goal` checks). A first, smaller run (n=12) looked
like it leaned Blocksworld's direction, but that didn't survive retesting at n=30 with wider move
sampling: diff is essentially zero at higher budget and mildly negative, not positive, at lower
budget -- parity, matching the cube/8-puzzle/Sokoban, not a persistent gap. The smaller run's
apparent lean turned out to be two informative tasks agreeing by chance, not a real trend --
reported honestly as a corrected reading, not left standing as a suggestive result.

**Why Blocksworld alone shows the persistent gap, resolved**: a final mechanism trace instrumented
identical `new_nodes`/`merge_hits` budget accounting across Blocksworld, the cube, and ARC-AGI
(`REAL_HEURISTIC_MERGE_FINDINGS.md`). The persistent gap needs two factors together, not one: dense
transpositions AND a tree mode structurally unable to solve within budget on its own. The cube has
dense transpositions (even denser than Blocksworld's, ~57% vs. ~35-41% of budget spent on re-links)
but not the stuck tree (it reaches the same ceiling merge does). ARC has a mildly stuck tree but not
dense transpositions (DSL program composition is mostly order-sensitive, unlike Blocksworld's
block-stacking or the cube's move commutativity, so merge hits stay rare there). Blocksworld alone
combines both, which is exactly why it alone gets a dramatic rescue instead of parity.

## Status

This is the stopping point for this investigation. No further experiments are planned as of this
document. Every claim above is backed by a specific findings doc with full sweep tables,
statistical tests, and mechanism diagnostics -- this file is a map to that evidence, not a
replacement for it.
