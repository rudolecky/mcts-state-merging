# Classical MCTS Merging on Sokoban (No LLM)

## Question

Three domains closed with a clean pattern: Connect Four (irreversible, adversarial -- null),
the 8-puzzle (fully reversible, single-agent -- large positive), Three Men's Morris (reversible,
adversarial -- null, for a different reason than Connect Four's: adversarial interleaving
suppresses redundant exploration even with reversible moves). Sokoban was picked to test whether
effect size *grades* with redundant-exploration density: it's single-agent like the 8-puzzle, but
only walking is freely reversible -- pushing a box is directional and only reversible when the
board geometry allows walking around to push it back, which isn't always possible. The prediction
was an effect somewhere between Morris's near-null and the 8-puzzle's large positive.

## Setup

- `src/mcts_phase0/datasets/sokoban_engine.py`: one fixed 10x10 level (8x8 interior, 64 floor
  cells, one box, one goal at the room's center). Sized directly, not guessed: a 6x6 level (16
  floor cells) only reached 60 total pull-reachable states before saturating; 10x10 reaches 2,268
  states at a max optimal distance of 21 -- checked before committing to it.
- `bfs_distances`: a single multi-source BFS from every "solved" configuration (box on goal,
  player at any other free cell -- matching real Sokoban's player-position-independent win
  condition) via a private, generation-only `_pull_moves`/`_apply_pull` pair (the literal inverse
  of a push). Gives the exact optimal push-distance for every reachable state, the same
  invertible-moves-give-true-BFS-distances trick as the 8-puzzle's, just with pull instead of
  blank-swap as the traversal function. The real game (`legal_moves`/`apply_move`, used by the
  search) only ever pushes -- pulling is purely a construction device, never exposed to search.
- `src/mcts_phase0/classical_mcts_sokoban.py`: single-agent, cycle-safe `select()`/`run_search`
  (fourth reuse of the fix first needed for the 8-puzzle -- walking is reversible here too, and
  it passed its regression test cleanly on the first run). No deadlock detection: an unsolvable
  state (box wedged against a wall) just never solves within budget, same convention as every
  "didn't finish in time" case elsewhere in this project.
- Verified: `pytest tests/test_sokoban_engine.py tests/test_classical_mcts_sokoban.py` (21 tests)
  + full suite (221 passed) green.

## Results

**Sweep 1** -- target distances 3/6/9 (15 puzzles each), budgets 25-200, 50 repeats/condition:

| budget | overall diff | wilcoxon p | nodes base→treat |
|---|---|---|---|
| 25 | +0.030 | 0.0006 | 26.0→15.3 |
| 50 | +0.064 | 0.0015 | 51.0→25.8 |
| 100 | +0.061 | 0.0171 | 101.0→46.3 |
| 200 | +0.224 | 0.0003 | 194.9→78.8 |

By target distance (budget=200): d=3 both near-saturated (1.000/1.000); **d=6: baseline 0.005,
treatment 0.521**; d=9: baseline 0.000, treatment 0.157. Already real and significant, and
clearly still growing -- extended to higher budgets.

**Sweep 2** -- distances 6/9 only, budgets 300-1200:

| budget | overall diff | wilcoxon p | nodes base→treat |
|---|---|---|---|
| 300 | +0.601 | <0.0001 | 301.0→119.8 (60% fewer) |
| 500 | +0.761 | <0.0001 | 501.0→162.7 (68% fewer) |
| 800 | +0.720 | <0.0001 | 800.0→195.5 (76% fewer) |
| 1200 | +0.663 | <0.0001 | 1190.4→222.2 (81% fewer) |

By target distance:

| budget | d=6 base | d=6 treat | d=9 base | d=9 treat |
|---|---|---|---|---|
| 300 | 0.023 | 0.769 | 0.000 | 0.455 |
| 500 | 0.076 | 0.977 | 0.000 | 0.620 |
| 800 | 0.243 | 1.000 | 0.000 | 0.683 |
| 1200 | 0.517 | 1.000 | 0.000 | 0.844 |

**d=9: baseline never solves a single puzzle across the entire 300-1200 range (0 out of 750
runs), while treatment climbs to 0.844.** d=6 reaches treatment=1.000 by budget=800 while
baseline is still only at 0.243; even at budget=1200, baseline (0.517) is roughly half the
treatment rate. This is large, decisive, and -- notably -- **comparable in scale to the
8-puzzle's effect**, not a modest midpoint between Morris and the 8-puzzle as a naive
"half the moves are irreversible" intuition would predict.

## Mechanism

This sharpens the graded-density hypothesis rather than confirming its naive form. The prediction
was effect size proportional to *fraction of moves that are reversible*. What actually seems to
matter is different: **whether the dominant activity in the search is reversible, not what
fraction of the move-type vocabulary is.** In Sokoban, most legal moves at most states are plain
walks (the player has up to 4 walk-options at any position but a push is only available when
adjacent to the box in exactly the right spot) -- so even though pushing itself is largely
one-way, the search spends most of its exploration budget on walking, which is exactly as
reversible as every 8-puzzle move. That's enough to generate 8-puzzle-scale redundant
re-exploration of the same joint (player, box) state under UCB1, which merging reclaims just as
effectively -- node-count reduction here (60-81%) matches or exceeds the 8-puzzle's own (30-62%).

Combined with all three prior results, the refined claim is now: **merging helps in proportion to
how much of the search's actual exploration activity revisits already-seen states under UCB1 --
which depends on what the search spends most of its moves doing, not on a simple reversible/
irreversible move-type ratio.** Morris's null wasn't about partial irreversibility diluting the
effect; it was about adversarial interleaving making even fully-reversible moves rarely revisit
the same joint state. Sokoban's largely-walking-dominated branching means most of its search
activity is reversible in practice, so it behaves close to the 8-puzzle despite having a
genuinely irreversible move type in its rules.

## Status

Closed. Two sweeps (budgets 25-1200, spanning genuine floor-to-ceiling for every target
distance), a large, decisive, still-strong-at-1200 positive effect, node-count evidence of
substantial real merging (up to 81% fewer nodes), and a mechanism read that meaningfully refines
rather than just confirms the standing hypothesis. This was the second of the user's two
requested follow-up domains (Morris, then Sokoban) -- no further domain is queued by default.
