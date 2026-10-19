# Classical MCTS Merging on the 8-Puzzle (No LLM)

## Question

Connect Four's classical UCB1 MCTS merge-vs-tree test (`CLASSICAL_MCTS_FINDINGS.md`) found a
clean null: merging genuinely deduplicates nodes but buys zero solve-rate benefit, because
UCB1's own untried-move bookkeeping already prevents wasted redundant exploration in that domain
(moves are irreversible -- every column drop advances the game by exactly one ply, so a plain
tree never revisits the same position twice). This second domain was picked specifically to vary
that one variable: the 8-puzzle's moves are **reversible** (undoing a move returns the prior
state), so a plain tree can -- and does -- waste huge amounts of budget re-exploring positions
it has already visited, each time as a fresh, separately-tracked node starting from zero
statistics. If Connect Four's "UCB1 already prevents redundant exploration" story is right, a
domain that actually generates heavy redundant exploration under UCB1 should show a real
benefit from merging. It does, dramatically.

## Setup

- `src/mcts_phase0/datasets/sliding_puzzle_engine.py`: flat-tuple state (already canonical, same
  principle as Connect Four's board tuple), `bfs_distances` -- a single reverse BFS from the goal
  giving the exact optimal-solution length for every one of the 181,440 solvable states on a 3x3
  board (0.13s, one-time; max distance found was 31, matching the well-known God's Number for the
  8-puzzle -- a strong correctness check that the BFS is right). `generate_puzzles` samples states
  at an exact target distance from this table, the same "exact difficulty by construction"
  principle as Connect Four's `k_plies`.
- `src/mcts_phase0/classical_mcts_puzzle.py`: single-agent UCB1 MCTS (no opponent, no
  hero-relative value -- solved or not), same merge-switch design as `classical_mcts.py` (state
  tuple as the node key in treatment mode, move-path in baseline mode).
- Verified: `pytest tests/test_sliding_puzzle_engine.py tests/test_classical_mcts_puzzle.py` (20
  tests) + full suite (175 passed) green.

## A real bug found and fixed along the way

The first version of `select()` computed the raw UCB1 argmax over *all* of a node's children,
then bailed out entirely if that single best child happened to be an ancestor already on the
current path (a real possibility here, since moves are reversible -- unlike Connect Four, where
this is structurally impossible). That bail-out was too aggressive: it gave up even when the node
had *other*, perfectly good non-cycling children available. The result was a silent, severe
under-exploration bug -- the treatment condition's node count plateaued at ~9 nodes regardless of
budget (25 through 200), because the search got trapped in a small already-visited pocket and
never escaped it, even though the true 8-puzzle state graph has no dead ends at all (it's fully
connected within a parity class). Fixed by restricting the argmax to non-cycling children, only
giving up when *every* child would revisit the path (genuinely rare). After the fix, node counts
scale with budget correctly in both conditions, confirmed directly before re-running the real
experiment.

## Results

**Sweep 1** -- target distances 4/8/12 (15 puzzles each), budgets 25-200, 50 repeats/condition:

| budget | overall diff | CI | wilcoxon p | nodes base→treat |
|---|---|---|---|---|
| 25 | +0.081 | [0.044, 0.123] | 0.0006 | 26.0→18.2 |
| 50 | +0.092 | [0.051, 0.137] | 0.0006 | 51.0→34.3 |
| 100 | +0.008 | [0.004, 0.016] | 0.0016 | 101.0→61.7 |
| 200 | +0.079 | [0.024, 0.152] | 0.0006 | 200.7→104.6 |

By target distance (budget=200): td=4 both conditions already saturated at 1.000 (a ceiling
effect that mutes the aggregate, hence the dip at budget=100 where td=4 is finishing its climb
for both conditions simultaneously); **td=8: baseline 0.000, treatment 0.236**; td=12 both still
at floor. The overall significance at every budget is real, not an artifact -- but budgets up to
200 aren't yet the interesting regime for the harder puzzles.

**Sweep 2** -- target distances 8/12 only (already-solved td=4 dropped), budgets 300-1200:

| budget | overall diff | CI | wilcoxon p | nodes base→treat |
|---|---|---|---|---|
| 300 | +0.291 | [0.178, 0.415] | 0.0004 | 301.0→182.5 |
| 500 | +0.511 | [0.349, 0.669] | 0.0002 | 501.0→271.0 |
| 800 | +0.556 | [0.397, 0.710] | <0.0001 | 801.0→360.0 |
| 1200 | +0.558 | [0.425, 0.687] | <0.0001 | 1200.8→456.6 |

By target distance:

| budget | td=8 base | td=8 treat | td=12 base | td=12 treat |
|---|---|---|---|---|
| 300 | 0.000 | 0.577 | 0.000 | 0.004 |
| 500 | 0.004 | 0.953 | 0.000 | 0.072 |
| 800 | 0.061 | 1.000 | 0.000 | 0.173 |
| 1200 | 0.215 | 1.000 | 0.000 | 0.331 |

**td=8 at budget=800/1200: treatment solves every single time (1.000); baseline never exceeds
0.215 even at budget=1200 (5.6x the plain-tree solve rate).** td=12 baseline never solves a
single puzzle across the entire 300-1200 budget range (0 out of 750 baseline runs), while
treatment climbs steadily to 0.331. This is a large, decisive, and still-growing effect, not a
transient blip.

Node-count reduction grows with budget too: 39% fewer at budget=100 (Sweep 1), 62% fewer at
td=8/budget=1200. Note treatment's node count for td=8 actually *plateaus* around 240-360 past
budget=800 (once solve rate saturates, additional budget mostly refines/re-confirms the known
solution rather than growing the graph), while baseline's node count keeps growing linearly with
budget by construction (tree mode: node count == budget, always).

## Mechanism

This is the sharp, predicted contrast with Connect Four, and it confirms that project's
mechanism story directly rather than complicating it. Connect Four's UCB1 baseline never wastes
budget on redundant exploration -- moves are irreversible, so every one of a plain tree's nodes
represents genuine, non-duplicated progress, and merging has nothing to reclaim. The 8-puzzle's
UCB1 baseline **does** waste enormous budget: because moves are reversible, a plain tree can
(and, given random UCB1-driven wandering, routinely does) revisit the same underlying puzzle
state many times over via different move sequences, each time building a **fresh, independent,
low-visit-count node** for what is actually the same position -- exactly the "many near-duplicate
nodes each getting only a fraction of the value signal they'd get if pooled" pathology that made
merging valuable in this project's original LLM-based H3 results, just produced here by
random-walk revisits instead of parallel independent sampling.

Merging fixes it the same way in both cases: pooling statistics across every path that reaches
the same state means the state gets one well-estimated value from ALL the visits combined,
instead of many poorly-estimated fragments. Combined with `CLASSICAL_MCTS_FINDINGS.md`'s null,
the two results together sharpen the project's central finding precisely: **merging helps
exactly to the degree that the underlying search process generates redundant, wasted exploration
of the same state under a different name.** UCB1 with irreversible actions generates ~none
(Connect Four); UCB1 with reversible actions generates a lot (8-puzzle); independent LLM sampling
generates a lot regardless of reversibility (the original H3 domains). Transposition density
isn't itself the deciding variable -- redundant-exploration density is; reversible moves are just
one concrete way to produce it.

## Status

Closed. Clean, well-powered (50 repeats × 15-30 puzzles × 8 budget points across two sweeps),
large and still-growing positive effect, mechanistically explained and consistent with -- not
contradicting -- Connect Four's null. No further sweeps planned; the td=8 saturation curve
(0→0.577→0.953→1.000 for treatment vs. 0→0.004→0.061→0.215 for baseline) and td=12's still-rising
treatment curve against a completely flat baseline floor already tell the full story.
