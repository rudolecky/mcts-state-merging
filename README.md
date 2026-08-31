# MCTS State Merging

This project tests a simple idea: in Monte Carlo Tree Search (MCTS), a search node is normally
identified by the sequence of moves that reached it. If two different move sequences lead to the
exact same underlying state, the tree treats them as two separate nodes and explores each one from
scratch. "Merging" instead identifies a node by the state itself, so any later path that reaches
an already seen state reuses the existing node and pools its statistics (visit counts, accumulated
value) with every other path that reached it.

The question this project asks, over and over, across many domains and algorithms: does merging
improve search accuracy at a matched compute budget, and when does it not?

## How merging works, concretely

Every search domain in this repo can run in two modes, controlled by a single config flag
(`merge_enabled`):

* Tree mode (baseline): nodes are keyed by the path taken to reach them. Every expansion creates a
  brand new node, even if the resulting state already exists elsewhere in the tree.
* Merge mode (treatment): nodes are keyed by the state itself. If an expansion lands on a state
  already present in the tree, the search adds a new edge to the existing node instead of creating
  a duplicate.

Classical domains key on exact state equality. The LLM domain (ProsQA) keys on a near match within
a small tolerance in a learned value space, since free text reasoning traces rarely match exactly.

## Domains tested

* Classical puzzles: Blocksworld, the 8 puzzle, Sokoban, Connect Four, Three Men's Morris.
* ARC AGI program synthesis (DSL based search).
* A full 2x2x2 Rubik's Cube, built from scratch with a group theoretic state representation,
  verified against the exact known count of 3,674,160 reachable states.
* An LLM guided harness on ProsQA (a synthetic entailment reasoning task), using a small linear
  projection trained on hidden states as the value signal.
* A real self play AlphaZero clone on Connect Four (trained network, not a heuristic stand in).

## What was found

Under blind UCB1 search with Monte Carlo average backup, merging helps a lot in domains with dense
redundant transpositions (Blocksworld, the 8 puzzle, Sokoban, ARC AGI), is null in domains without
much redundancy (Connect Four, Morris), and is structurally null under a real AlphaZero style PUCT
search, since that algorithm never creates duplicate nodes in the first place.

A separate, independent finding: pure random restarts (no tree at all) often beat blind tree search
at low to moderate budgets. Whether that gap persists as budget grows or closes quickly turns out to
predict, cheaply and without building any merge machinery, whether merging will help on a given
domain. Giving the tree real learned guidance (a trained value projection on the ProsQA LLM harness)
does not by itself fix a tree that is stuck this way, confirming that budget allocation, not
guidance quality, is often the actual bottleneck.

The sharpest result came from the Rubik's Cube. Under UCB1 with Monte Carlo average backup, merging
does not just fail to help there, it is catastrophic: solve rate drops to exactly zero at every
budget and difficulty where plain tree search reaches 100 percent. Two different capping based
mitigations (limiting how many parents or how many visits a merged node can absorb) both failed in
the same all or nothing way, which pointed at the backup rule itself rather than at how often
merging fires.

Swapping the backup rule fixed it completely. Porting GUCT Uniform (arXiv:2405.18248), which uses
Full Bellman minimum backup and LCB1 Uniform selection instead of Monte Carlo averaging and UCB1,
made merge and tree solve rates statistically identical on the cube, the 8 puzzle, and Sokoban,
while merge still used 20 to 57 percent fewer nodes for the same result. On Blocksworld, the same
backup rule swap went the other way: merging became even more beneficial than under blind UCB1
(tree stayed flat near 3 to 5 percent solved across a 40x budget range, merge reached 99 percent).

Put together, the conclusion is that whether merging helps, hurts, or does nothing to solve rate is
mostly a property of the search algorithm's backup rule and of whether plain tree search on that
domain gets permanently trapped without it, not a fixed property of how much redundant structure a
domain has.

## Repository layout

* `src/mcts_phase0/`: all search algorithms and domain engines.
* `src/mcts_phase0/datasets/`: puzzle and game engines (state representation, legal moves, exact
  distance tables where feasible).
* `tests/`: unit tests for every module, including hand computed checks and small scale real runs.
* `data/`: external benchmark files (ARC AGI tasks, PDDL benchmark instances), not code.

## Running it

```
uv sync
uv run pytest -q
```

Individual experiments are runnable as modules, for example:

```
uv run python -m mcts_phase0.run_heuristic_merge_experiment_rubiks
```

Most classical domain experiments run in seconds to minutes on a laptop. The ProsQA LLM harness
requires downloading a real model and is much slower.
