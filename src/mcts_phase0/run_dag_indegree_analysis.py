"""CLI: does a merged node's IN-DEGREE carry signal about whether it lies
on a solution path?

Merging turns the search tree into a DAG, and the only structure the DAG
has that a tree does not is that some nodes have several parents. That
structure is currently produced and thrown away. If in-degree predicts
solution-path membership, it is a free exploration prior -- free because it
is already computed, and a prior because it needs no extrinsic reward,
which is exactly the regime interactive/no-reward search settings sit in.

Two controls are reported alongside it, because the interesting claim is
narrow. `depth` tests whether in-degree is just re-reading distance from
the root. `visits` tests whether it is just re-reading "UCB1 spent budget
here", which would make it a restatement of the search's own preference
rather than independent information. An in-degree AUC that is not clearly
better than both is not a finding.

Since one control points the other way (on-path nodes are SHALLOWER), every
cross-feature comparison uses `_discrimination` = max(AUC, 1-AUC) rather
than the raw AUC; a depth AUC of 0.28 discriminates as well as one of 0.72,
and reading it raw would make a strong control look dead.

The searches run with `stop_on_solve=True`, which removes one artifact
without pretending to remove the confound. Without it, once a terminal node
exists the loop keeps re-selecting it and backing up 1.0 along the solution
path without consuming any expansion budget, so on-path visit counts inflate
purely as a function of how long the loop ran afterwards (measured 589
on-path vs 1.3 off-path). With it, on-path nodes still carry far more visits
than off-path ones (~16 vs ~1.2), because they sit on the UCB1-preferred
descent and get backed up every iteration. That is a real property of the
search, not an artifact -- so visits remains a strong control and the only
claim here is that its strength is no longer exaggerated.

"On a solution path" means: reachable from the root AND able to reach a
terminal node, computed over the real edge set (so cycles, which merging
can create and trees cannot, are handled by plain reachability rather than
assumed away).

Zero new model compute: this reuses the classical UCB1 merge-mode searches
that already run in seconds per instance.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from . import classical_mcts_blocksworld as bw
from . import classical_mcts_sokoban as sk
from .datasets import blocksworld_engine, sokoban_engine


_EMPTY_STRATIFIED = {"auc": float("nan"), "n_strata_used": 0, "n_pairs": 0,
                     "pair_coverage": 0.0, "degenerate": True}


def in_degrees(graph) -> dict:
    """Incoming edge count per node key. Counts EDGES, not distinct
    parents: two different moves from the same parent landing on the same
    merged child are two routes into it, which is what "how many ways does
    the search reach this state" means here."""
    counts = {key: 0 for key in graph.nodes}
    for node in graph.nodes.values():
        for edge in node.children.values():
            counts[edge.child_key] = counts.get(edge.child_key, 0) + 1
    return counts


def depths_from_root(graph) -> dict:
    """BFS distance from the root over the edge set (shortest number of
    edges, not the path a particular selection took)."""
    depth = {graph.root_key: 0}
    queue = deque([graph.root_key])
    while queue:
        key = queue.popleft()
        for edge in graph.nodes[key].children.values():
            if edge.child_key not in depth:
                depth[edge.child_key] = depth[key] + 1
                queue.append(edge.child_key)
    return depth


def _forward_reachable(graph, sources) -> set:
    seen = set(sources)
    queue = deque(sources)
    while queue:
        key = queue.popleft()
        for edge in graph.nodes[key].children.values():
            if edge.child_key not in seen:
                seen.add(edge.child_key)
                queue.append(edge.child_key)
    return seen


def _backward_reachable(graph, targets) -> set:
    parents: dict = {key: set() for key in graph.nodes}
    for key, node in graph.nodes.items():
        for edge in node.children.values():
            parents.setdefault(edge.child_key, set()).add(key)
    seen = set(targets)
    queue = deque(targets)
    while queue:
        key = queue.popleft()
        for parent in parents.get(key, ()):
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)
    return seen


def solution_path_nodes(graph, is_terminal_fn) -> set:
    """Nodes lying on at least one root -> terminal path."""
    terminals = [key for key, node in graph.nodes.items() if is_terminal_fn(node)]
    if not terminals:
        return set()
    return _forward_reachable(graph, [graph.root_key]) & _backward_reachable(graph, terminals)


def _auc(on_values, off_values) -> float:
    """P(a random on-path node scores above a random off-path node), ties
    counted as half -- which is exactly Mann-Whitney U normalized."""
    if not on_values or not off_values:
        return float("nan")
    u, _p = mannwhitneyu(on_values, off_values, alternative="two-sided")
    return float(u / (len(on_values) * len(off_values)))


def _discrimination(auc: float) -> float:
    """`max(auc, 1-auc)`: how well the feature separates the two classes,
    regardless of which way it points.

    Raw AUCs are not comparable across features that point opposite ways,
    and here one of them does -- on-path nodes are SHALLOWER, so depth
    lands near 0.28 while in-degree lands near 0.85. Read raw, depth looks
    like a dead control at 0.28 when it is in fact discriminating at 0.72.
    Every comparison across features has to use this, matching
    `_permutation_p`, which already tests |AUC - 0.5|."""
    if auc != auc:  # NaN
        return auc
    return max(auc, 1.0 - auc)


def _permutation_p(values, labels, graph_ids, n_perm: int, rng: np.random.Generator) -> float:
    """Label-shuffling null for |AUC - 0.5|, permuting labels WITHIN each
    search graph.

    Permuting globally would be the wrong null for the stated reason: nodes
    inside one search graph are not independent draws, and a global shuffle
    destroys that clustering instead of preserving it, so it smuggles back
    exactly the independence assumption the permutation was meant to avoid
    (and pins the p-value at the 1/(n_perm+1) floor). Shuffling within
    graph keeps each graph's on/off-path counts fixed and only randomizes
    which of its own nodes carry which label."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    graph_ids = np.asarray(graph_ids)
    observed = abs(_auc(list(values[labels]), list(values[~labels])) - 0.5)

    blocks = [np.flatnonzero(graph_ids == g) for g in np.unique(graph_ids)]
    shuffled_labels = labels.copy()
    hits = 0
    for _ in range(n_perm):
        for idx in blocks:
            block = labels[idx].copy()
            rng.shuffle(block)
            shuffled_labels[idx] = block
        on = list(values[shuffled_labels])
        off = list(values[~shuffled_labels])
        if not on or not off:
            continue
        if abs(_auc(on, off) - 0.5) >= observed:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def _stratified_auc(values, labels, strata, n_strata: int) -> dict:
    """AUC of `values` within bins of `strata`, pooled as a size-weighted
    mean over bins that contain both classes.

    This is the test that decides whether in-degree is worth anything as a
    prior. The unconditional in-degree AUC competes against visit count,
    but visit count is not a competitor a search can actually use -- it is
    the search's own output, so "the nodes UCB1 visited most are the ones
    the solution ran through" is close to circular. The question is whether
    in-degree separates on- from off-path nodes among nodes the search
    treated the SAME, which is what conditioning on visits asks."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    strata = np.asarray(strata, dtype=float)

    # Bin by the most COMMON stratum values, pooling the rest into one bin.
    #
    # A plain "exact if few distinct values else quantile" switch fails on
    # exactly the distribution this is applied to: visit counts put most of
    # their mass on a handful of small integers and then trail off into a
    # long tail of unique large values. The distinct count is therefore
    # large, the quantile branch is taken, and its edges collapse onto the
    # dense low values -- measured 2 usable strata out of 10 requested,
    # covering 10% of the available pairs, with a spread of stratum values
    # left inside each bin. That residual within-bin variation is the
    # confound leaking back in as apparent conditional signal, which is the
    # one thing this statistic exists to exclude. Quantile capping does not
    # fix it either, since the right cutoff depends on how heavy the tail
    # happens to be.
    #
    # Taking the top `n_strata - 1` values by frequency keeps every dense
    # value exact regardless of tail shape, and the pooled remainder bin
    # holds only sparse values, which almost always carry a single class and
    # get dropped below anyway.
    distinct, counts = np.unique(strata, return_counts=True)
    if len(distinct) < 2:
        return {"auc": float("nan"), "n_strata_used": 0, "n_pairs": 0,
                "pair_coverage": 0.0, "degenerate": True}
    if len(distinct) <= n_strata:
        keep = distinct
    else:
        keep = np.sort(distinct[np.argsort(-counts)[: max(n_strata - 1, 1)]])
    bin_of = {value: i for i, value in enumerate(keep)}
    other = len(keep)
    bins = np.fromiter((bin_of.get(v, other) for v in strata), dtype=int, count=len(strata))
    # The pooled remainder bin is deliberately NOT scored: it is the one bin
    # whose stratum values differ inside it, so it is the one bin that could
    # leak the confound back in. Only exact-value bins contribute.
    n_bins = other

    weighted, total, used = 0.0, 0, 0
    for b in range(n_bins):
        mask = bins == b
        on = list(values[mask & labels])
        off = list(values[mask & ~labels])
        if not on or not off:
            continue
        auc = _auc(on, off)
        weight = len(on) * len(off)
        weighted += auc * weight
        total += weight
        used += 1
    # Coverage is reported, not used as the trigger: conditioning always
    # discards most pairs (any pair whose confounder values differ), so low
    # coverage is expected rather than suspicious. What does invalidate the
    # statistic is having almost no strata or almost no pairs left, which is
    # what `degenerate` flags -- a conditional AUC over one or two bins is
    # not evidence in either direction.
    available = int(labels.sum()) * int((~labels).sum())
    coverage = (total / available) if available else 0.0
    return {
        "auc": float(weighted / total) if total else float("nan"),
        "n_strata_used": used, "n_pairs": int(total),
        "pair_coverage": float(coverage),
        "degenerate": bool(used < 3 or total < 1000),
    }


def collect_domain(name: str, searches, n_perm: int, seed: int, n_strata: int = 10) -> dict:
    """`searches` yields (graph, is_terminal_fn) for solved merge-mode runs.

    Two node classes are excluded, both because their feature values are
    fixed by construction rather than earned, and both on the on-path side
    (so the exclusions cut against the hypothesis, not for it):

    * the root -- always on-path, and its depth is 0 by definition. Note
      its in-degree is NOT 0 in merge mode: undoing a move re-reaches the
      start state, so the root has parents in every solved merge graph
      checked. Dropping it therefore discards a high-in-degree on-path node.
    * terminal nodes -- under `stop_on_solve` the terminal is created by the
      very last expansion, so its in-degree is exactly 1, the minimum
      possible, which would drag the on-path in-degree mean down.
    """
    records = {"indegree": [], "depth": [], "visits": [], "on_path": [], "graph_id": []}
    n_graphs = 0
    for graph, is_terminal_fn in searches:
        on_path = solution_path_nodes(graph, is_terminal_fn)
        if not on_path or len(on_path) == len(graph.nodes):
            continue  # nothing to discriminate between
        indeg = in_degrees(graph)
        depth = depths_from_root(graph)
        for key, node in graph.nodes.items():
            if key == graph.root_key or is_terminal_fn(node):
                continue  # see docstring -- fixed by construction
            records["indegree"].append(indeg[key])
            records["depth"].append(depth[key])
            records["visits"].append(node.n_visits)
            records["on_path"].append(key in on_path)
            records["graph_id"].append(n_graphs)
        n_graphs += 1

    labels = records["on_path"]
    rng = np.random.default_rng(seed)
    out = {"domain": name, "n_graphs": n_graphs,
           "n_on": int(sum(labels)), "n_off": int(len(labels) - sum(labels))}
    both_classes = bool(out["n_on"] and out["n_off"])
    for feature in ("indegree", "depth", "visits"):
        on_values = [v for v, on in zip(records[feature], labels) if on]
        off_values = [v for v, on in zip(records[feature], labels) if not on]
        auc = _auc(on_values, off_values)
        out[feature] = {
            "mean_on": float(np.mean(on_values)) if on_values else float("nan"),
            "mean_off": float(np.mean(off_values)) if off_values else float("nan"),
            "auc": auc,
            "discrimination": _discrimination(auc),
            "perm_p": (_permutation_p(records[feature], labels, records["graph_id"], n_perm, rng)
                       if both_classes else float("nan")),
        }
    # Always present, even with no usable data -- `main` reads them
    # unconditionally, and nothing is written to disk until after it prints,
    # so a KeyError here would throw away the whole run's search spend.
    out["indegree_given_visits"] = _stratified_auc(
        records["indegree"], labels, records["visits"], n_strata) if both_classes else _EMPTY_STRATIFIED
    out["indegree_given_depth"] = _stratified_auc(
        records["indegree"], labels, records["depth"], n_strata) if both_classes else _EMPTY_STRATIFIED
    return out


def _blocksworld_searches(args):
    task = blocksworld_engine.make_task(args.num_blocks)
    goal = blocksworld_engine.goal_state(args.num_blocks)
    puzzles = []
    for plan_length in args.plan_lengths:
        puzzles.extend(blocksworld_engine.generate_puzzles(
            n=args.n_puzzles, seed=args.seed, num_blocks=args.num_blocks, target_plan_length=plan_length))
    print(f"blocksworld: {len(puzzles)} puzzles")
    for i, inst in enumerate(puzzles):
        for r in range(args.repeats):
            config = bw.ClassicalMCTSConfig(merge_enabled=True)
            graph = bw.run_search(inst.start_state, goal, task, config, args.budget,
                                  args.rollout_depth, random.Random(i * 100_003 + r),
                                  stop_on_solve=True)
            if bw.is_solved(graph):
                yield graph, lambda n: n.is_terminal and n.terminal_value == 1.0


def _sokoban_searches(args):
    distances = sokoban_engine.bfs_distances()
    puzzles = []
    for target_distance in args.target_distances:
        puzzles.extend(sokoban_engine.generate_puzzles(
            n=args.n_puzzles, seed=args.seed, target_distance=target_distance, distances=distances))
    print(f"sokoban: {len(puzzles)} puzzles")
    for i, inst in enumerate(puzzles):
        for r in range(args.repeats):
            config = sk.ClassicalMCTSConfig(merge_enabled=True)
            graph = sk.run_search(inst.start_state, config, args.budget,
                                  args.rollout_depth, random.Random(i * 100_003 + r),
                                  stop_on_solve=True)
            if sk.is_solved(graph):
                yield graph, lambda n: n.is_terminal and n.terminal_value == 1.0


def main():
    ap = argparse.ArgumentParser(description="Does merge in-degree predict solution-path membership?")
    ap.add_argument("--domains", nargs="+", choices=["blocksworld", "sokoban"],
                    default=["blocksworld", "sokoban"])
    ap.add_argument("--num-blocks", type=int, default=5)
    ap.add_argument("--plan-lengths", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--target-distances", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--n-puzzles", type=int, default=10, help="puzzles per difficulty tier")
    ap.add_argument("--repeats", type=int, default=5, help="searches per puzzle")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--rollout-depth", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-strata", type=int, default=10,
                    help="quantile bins for the conditional (in-degree given visits) AUC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/dag_indegree_analysis.json")
    args = ap.parse_args()

    start = time.time()
    rows = []
    for domain in args.domains:
        searches = _blocksworld_searches(args) if domain == "blocksworld" else _sokoban_searches(args)
        row = collect_domain(domain, searches, args.n_perm, args.seed, args.n_strata)
        rows.append(row)
        print(f"{domain}: {row['n_graphs']} solved graphs with a discriminable split "
              f"(n_on={row['n_on']}, n_off={row['n_off']})")
        for feature in ("indegree", "depth", "visits"):
            stats = row[feature]
            # `discrimination` is max(AUC, 1-AUC) -- the only figure that is
            # comparable across features here, since depth points the other way.
            print(f"  {feature:9s} on={stats['mean_on']:.2f} off={stats['mean_off']:.2f} "
                  f"AUC={stats['auc']:.3f} discrim={stats['discrimination']:.3f} "
                  f"perm_p={stats['perm_p']:.4f}")
        for key in ("indegree_given_visits", "indegree_given_depth"):
            stats = row[key]
            flag = "  DEGENERATE (too few strata or too few pairs -- not evidence)" if stats["degenerate"] else ""
            print(f"  {key:23s} AUC={stats['auc']:.3f} discrim={_discrimination(stats['auc']):.3f} "
                  f"({stats['n_strata_used']} strata, {stats['n_pairs']} pairs, "
                  f"{stats['pair_coverage']:.0%} of pairs){flag}")

    elapsed = time.time() - start
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "elapsed_sec": elapsed, "rows": rows}, f, indent=2)
    print(f"done in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
