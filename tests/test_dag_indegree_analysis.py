"""Verification for run_dag_indegree_analysis.py on hand-built graphs:
in-degree counting over a diamond (the minimal merge), BFS depth, the
root->terminal reachability that defines "on a solution path" (including a
cycle, which merging can create and a tree cannot), and the stratified AUC
-- tested on a constructed confound where the unconditional AUC is perfect
and the conditional AUC is exactly chance, since detecting that difference
is the whole reason the conditional statistic exists.
"""

from dataclasses import dataclass, field

from mcts_phase0.run_dag_indegree_analysis import (
    _auc,
    collect_domain,
    _stratified_auc,
    depths_from_root,
    in_degrees,
    solution_path_nodes,
)


@dataclass
class _Edge:
    child_key: object


@dataclass
class _Node:
    n_visits: int = 0
    is_terminal: bool = False
    terminal_value: float | None = None
    children: dict = field(default_factory=dict)


@dataclass
class _Graph:
    nodes: dict
    root_key: object


def _terminal(node):
    return node.is_terminal and node.terminal_value == 1.0


def _diamond():
    """root -> a, root -> b, a -> d, b -> d. `d` is the merged node with
    two routes into it; `c` hangs off `a` as a dead end."""
    return _Graph(
        nodes={
            "root": _Node(children={0: _Edge("a"), 1: _Edge("b")}),
            "a": _Node(children={0: _Edge("d"), 1: _Edge("c")}),
            "b": _Node(children={0: _Edge("d")}),
            "c": _Node(),
            "d": _Node(is_terminal=True, terminal_value=1.0),
        },
        root_key="root",
    )


def test_in_degrees_counts_every_route_into_a_merged_node():
    assert in_degrees(_diamond()) == {"root": 0, "a": 1, "b": 1, "c": 1, "d": 2}


def test_in_degrees_counts_edges_not_distinct_parents():
    """Two different moves from one parent onto the same merged child are
    two routes into it -- which is what "how many ways does the search
    reach this state" has to mean for the statistic to be about the DAG."""
    graph = _Graph(
        nodes={"root": _Node(children={0: _Edge("x"), 1: _Edge("x")}), "x": _Node()},
        root_key="root",
    )
    assert in_degrees(graph)["x"] == 2


def test_depths_from_root_is_the_shortest_edge_distance():
    assert depths_from_root(_diamond()) == {"root": 0, "a": 1, "b": 1, "c": 2, "d": 2}


def test_solution_path_nodes_excludes_dead_branches():
    assert solution_path_nodes(_diamond(), _terminal) == {"root", "a", "b", "d"}


def test_solution_path_nodes_is_empty_without_a_terminal():
    graph = _diamond()
    graph.nodes["d"] = _Node()  # no terminal anywhere
    assert solution_path_nodes(graph, _terminal) == set()


def test_solution_path_nodes_survives_a_cycle():
    """Aggregation can produce cycles a tree cannot have. Plain
    reachability handles them; anything path-enumerating would not."""
    graph = _Graph(
        nodes={
            "root": _Node(children={0: _Edge("a")}),
            "a": _Node(children={0: _Edge("b")}),
            "b": _Node(children={0: _Edge("a"), 1: _Edge("goal")}),  # b -> a closes the loop
            "goal": _Node(is_terminal=True, terminal_value=1.0),
            "dead": _Node(),
        },
        root_key="root",
    )
    assert solution_path_nodes(graph, _terminal) == {"root", "a", "b", "goal"}


def test_auc_is_one_for_perfectly_separated_values():
    assert _auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert _auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert _auc([1, 1, 1], [1, 1, 1]) == 0.5  # all ties -> no information


def test_stratified_auc_reveals_a_pure_confound():
    """Constructed so that in-degree separates the classes perfectly
    overall, but only because it tracks the stratifying variable: inside
    each stratum the two classes have identical in-degree, so the
    conditional AUC must be 0.5 while the unconditional one is 1.0."""
    values, labels, strata = [], [], []
    # In-degree is a function of the stratum ALONE -- never of the class.
    # The classes differ only in how they are distributed across strata.
    for stratum, value, n_on, n_off in [(0, 10, 2, 8), (1, 20, 5, 5), (2, 30, 8, 2)]:
        for _ in range(n_on):
            values.append(value)
            labels.append(True)
            strata.append(stratum)
        for _ in range(n_off):
            values.append(value)
            labels.append(False)
            strata.append(stratum)

    on_values = [v for v, on in zip(values, labels) if on]
    off_values = [v for v, on in zip(values, labels) if not on]
    assert _auc(on_values, off_values) > 0.6  # looks informative pooled

    conditional = _stratified_auc(values, labels, strata, n_strata=10)
    assert conditional["auc"] == 0.5  # and carries nothing once conditioned
    assert conditional["n_strata_used"] > 0


def test_stratified_auc_keeps_real_signal():
    """Same shape, but in-degree genuinely separates the classes within
    every stratum -- the conditional AUC has to stay high."""
    values, labels, strata = [], [], []
    for stratum in range(3):
        for _ in range(5):
            values.append(9)
            labels.append(True)
            strata.append(stratum)
        for _ in range(5):
            values.append(1)
            labels.append(False)
            strata.append(stratum)
    conditional = _stratified_auc(values, labels, strata, n_strata=3)
    assert conditional["auc"] == 1.0


def test_stratified_auc_handles_a_single_stratum_value():
    conditional = _stratified_auc([1, 2], [True, False], [7, 7], n_strata=10)
    assert conditional["n_strata_used"] == 0
    assert conditional["auc"] != conditional["auc"]  # NaN -- nothing to condition on
    assert conditional["degenerate"] is True


def test_stratified_auc_flags_degenerate_coverage():
    """A conditional AUC computed over one usable bin covering a sliver of
    the available pairs is not evidence, and has to say so."""
    values = [1, 2] + [5] * 20
    labels = [True, False] + [True] * 10 + [False] * 10
    strata = [1, 1] + list(range(100, 120))  # dense pair, then all-unique tail
    conditional = _stratified_auc(values, labels, strata, n_strata=3)
    assert conditional["degenerate"] is True
    assert conditional["pair_coverage"] < 0.5


def test_stratified_auc_keeps_dense_values_exact_despite_a_long_tail():
    """The distribution this is applied to -- most mass on a few small
    integers, a long tail of unique large ones -- must not push every dense
    value into one collapsed quantile bin."""
    values, labels, strata = [], [], []
    for stratum in (1, 2, 3):
        for _ in range(150):
            values.append(9); labels.append(True); strata.append(stratum)
            values.append(1); labels.append(False); strata.append(stratum)
    for tail in range(500, 560):  # unique large values, one class each
        values.append(5); labels.append(True); strata.append(tail)
    conditional = _stratified_auc(values, labels, strata, n_strata=10)
    assert conditional["n_strata_used"] == 3
    assert conditional["auc"] == 1.0
    assert conditional["degenerate"] is False


def test_discrimination_is_orientation_free():
    from mcts_phase0.run_dag_indegree_analysis import _discrimination
    assert _discrimination(0.85) == 0.85
    assert _discrimination(0.28) == 0.72  # a "low" AUC discriminates just as well
    assert _discrimination(0.5) == 0.5


def test_permutation_p_shuffles_within_graph_and_leaves_inputs_alone():
    """Global shuffling would smuggle back the independence assumption the
    permutation exists to avoid. With one on-path and one off-path node per
    graph, a within-graph null can only ever produce two labelings, so the
    p-value must be bounded well away from the 1/(n_perm+1) floor."""
    import numpy as np
    from mcts_phase0.run_dag_indegree_analysis import _permutation_p
    values = [9, 1] * 8
    labels = [True, False] * 8
    graph_ids = [i // 2 for i in range(16)]
    values_before, labels_before = list(values), list(labels)
    p = _permutation_p(values, labels, graph_ids, 200, np.random.default_rng(0))
    assert 0.0 < p <= 1.0
    assert p > 1 / 201  # not pinned at the floor
    assert values == values_before and labels == labels_before


def test_collect_domain_returns_conditional_keys_even_with_no_data():
    """`main` reads these unconditionally and nothing is written to disk
    until after it prints, so a missing key would discard the whole run."""
    from mcts_phase0.run_dag_indegree_analysis import collect_domain
    row = collect_domain("empty", iter([]), n_perm=5, seed=0)
    assert row["indegree_given_visits"]["degenerate"] is True
    assert row["indegree_given_depth"]["degenerate"] is True
    assert row["n_graphs"] == 0


def test_collect_domain_excludes_root_and_terminal_nodes():
    graph = _diamond()
    graph.nodes["a"].n_visits = 5
    graph.nodes["b"].n_visits = 4
    graph.nodes["c"].n_visits = 1
    row = collect_domain("diamond", iter([(graph, _terminal)]), n_perm=5, seed=0)
    # root and the terminal `d` are dropped; a, b (on-path) and c (off) remain.
    assert row["n_on"] == 2 and row["n_off"] == 1
