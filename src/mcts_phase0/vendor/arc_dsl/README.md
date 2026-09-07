# Vendored: `michaelhodel/arc-dsl`

- Source: `https://github.com/michaelhodel/arc-dsl`
- Commit: `635de4902a5fb4e376f27333feaa396d3f5dfdcb` (branch `main`, fetched 2026-08-28)
- License: MIT (see `LICENSE` in this directory, copied unmodified from the source repo)
- Files vendored unmodified: `arc_types.py`, `constants.py`, `dsl.py` (160 primitive functions),
  `solvers.py` (one hand-written reference solution per real ARC-AGI-1 training task, validated
  399/400 correct against real data by the source repo's own `main.py` test harness -- rerun and
  confirmed by this project directly before use).

These files are imported directly (via a `sys.path` addition in
`src/mcts_phase0/datasets/arc_types_registry.py`, not by editing their import statements) rather
than reimplemented, so their own tested behavior is reused as-is. See
`src/mcts_phase0/datasets/arc_types_registry.py` for how this project layers a typed-signature
registry on top of `dsl.py`'s functions (including its 19 higher-order/`Callable`-taking
functions, whose signatures are not statically parameterized in the source and needed to be
hand-derived from their implementations) -- see `results/ARC_AGI_FINDINGS.md` for what this
project built and found using this DSL.
