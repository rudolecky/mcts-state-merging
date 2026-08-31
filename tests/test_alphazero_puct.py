"""Verification for alphazero/puct.py, independent of any trained network
-- a fixed, hand-set fake policy/value stands in for the network
throughout, the same way test_classical_mcts.py validated selection and
backup without needing a real model. The sign-flip convention in `backup`
is the single highest-risk piece of code in this module (a well-known
AlphaZero-clone bug class if gotten wrong), so it gets a dedicated,
hand-computed test.
"""

from mcts_phase0.alphazero.puct import (
    MCTSEdge,
    MCTSNode,
    PUCTConfig,
    _make_node,
    _puct_score,
    backup,
    create_root,
    expand_and_evaluate,
    select,
)
from mcts_phase0.datasets.connect_four_engine import X, O, make_empty_board


def _fake_evaluator(policy_by_col: dict[int, float], value: float):
    """A stand-in for network.evaluate: ignores the board, always returns
    the same fixed policy/value -- deterministic, no torch involved."""
    def _fn(board, to_move):
        return policy_by_col, value
    return _fn


# ---------- _puct_score: finite even for unvisited edges (unlike UCB1) ----------

def test_puct_score_is_finite_for_an_unvisited_edge():
    parent = MCTSNode(board=(), to_move=X, n_visits=5)
    child = MCTSNode(board=(), to_move=O, n_visits=0)
    edge = MCTSEdge(child_key="c", prior=0.4, n_edge=0)
    score = _puct_score(parent, edge, child, c_puct=1.5)
    assert score == 1.5 * 0.4 * (5 ** 0.5) / 1  # Q=0 (unvisited) + full exploration term
    assert score < float("inf")


def test_puct_score_uses_complement_for_parent_perspective():
    parent = MCTSNode(board=(), to_move=X, n_visits=10)
    # child has been visited once with a high value (0.9) from the CHILD's own to_move's
    # perspective -- that's actually bad news for the parent, so Q should be low (0.1), not 0.9.
    child = MCTSNode(board=(), to_move=O, n_visits=1, w_value=0.9)
    edge = MCTSEdge(child_key="c", prior=0.5, n_edge=1)
    score = _puct_score(parent, edge, child, c_puct=0.0)  # zero exploration term isolates Q
    assert abs(score - 0.1) < 1e-9


# ---------- select: PUCT argmax on a hand-built graph ----------

def test_select_picks_the_expected_puct_argmax():
    root = MCTSNode(board=(), to_move=X, n_visits=4, expanded=True)
    child_a = MCTSNode(board=(), to_move=O, n_visits=2, w_value=0.2)  # child's own value 0.1 -> parent Q=0.9
    child_b = MCTSNode(board=(), to_move=O, n_visits=2, w_value=1.8)  # child's own value 0.9 -> parent Q=0.1
    root.children = {
        0: MCTSEdge(child_key="a", prior=0.5, n_edge=2),
        1: MCTSEdge(child_key="b", prior=0.5, n_edge=2),
    }
    graph_nodes = {(): root, "a": child_a, "b": child_b}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    config = PUCTConfig(merge_enabled=True, c_puct=0.1)  # small c_puct: Q should dominate
    path, edge_draws, path_moves = select(_Graph(), config)
    assert path == [(), "a"]  # higher parent-perspective Q (0.9) wins
    assert edge_draws == [0]


# ---------- backup: the sign-flip convention, hand-computed ----------

def test_backup_flips_perspective_by_complement_each_ply():
    # root -> child -> leaf: root.to_move and leaf.to_move are the SAME player
    # (two plies = one full round trip), child.to_move is the opponent.
    root = MCTSNode(board=(), to_move=X)
    child = MCTSNode(board=(), to_move=O)
    leaf = MCTSNode(board=(), to_move=X)
    root.children = {0: MCTSEdge(child_key="child", prior=1.0)}
    child.children = {0: MCTSEdge(child_key="leaf", prior=1.0)}
    graph_nodes = {(): root, "child": child, "leaf": leaf}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    leaf_value = 0.8  # from leaf's (= root's) own to_move's perspective
    backup(_Graph(), [(), "child", "leaf"], [0, 0], leaf_value)

    assert leaf.w_value == 0.8  # leaf's own perspective, unflipped
    assert child.w_value == 1.0 - 0.8  # one ply removed: complement
    assert root.w_value == 0.8  # two plies removed: complement of complement = original value
    assert root.n_visits == 1 and child.n_visits == 1 and leaf.n_visits == 1
    assert root.children[0].n_edge == 1
    assert child.children[0].n_edge == 1


def test_backup_pools_at_a_shared_merged_node():
    # Mirrors every other classical module's two-parent pooling regression test.
    root = MCTSNode(board=(), to_move=X)
    a = MCTSNode(board=(), to_move=O)
    b = MCTSNode(board=(), to_move=O)
    m = MCTSNode(board=(), to_move=X)
    root.children = {0: MCTSEdge(child_key="a", prior=0.5), 1: MCTSEdge(child_key="b", prior=0.5)}
    a.children = {0: MCTSEdge(child_key="m", prior=1.0)}
    b.children = {0: MCTSEdge(child_key="m", prior=1.0)}
    graph_nodes = {(): root, "a": a, "b": b, "m": m}

    class _Graph:
        nodes = graph_nodes
        root_key = ()

    graph = _Graph()
    backup(graph, [(), "a", "m"], [0, 0], 0.8)
    backup(graph, [(), "b", "m"], [1, 0], 0.2)

    assert m.n_visits == 2
    assert m.w_value == 1.0  # pooled: 0.8 + 0.2, both m's own-perspective values, unflipped
    assert a.children[0].n_edge == 1
    assert b.children[0].n_edge == 1


# ---------- expand_and_evaluate: merge-vs-tree structural neutrality ----------

def test_expand_merges_transposed_paths_when_enabled_not_when_disabled():
    # (0,4,1) vs (1,4,0): X's own two moves (cols 0 and 1) reordered around
    # O's single sandwiched move (col 4) -- a genuine transposition, unlike
    # naively swapping the first two plies (which would swap X and O's
    # columns instead of reordering one player's own moves).
    width = 5

    def _play(config, moves):
        graph = create_root((), X, width, config.merge_enabled)
        key = graph.root_key
        path_moves = ()
        for col in moves:
            node = graph.nodes[key]
            if not node.expanded:
                expand_and_evaluate(graph, key, path_moves, width, config, _fake_evaluator({c: 0.2 for c in range(width)}, 0.5))
                node = graph.nodes[key]
            key = node.children[col].child_key
            path_moves = path_moves + (col,)
        return graph, key

    treatment_cfg = PUCTConfig(merge_enabled=True)
    baseline_cfg = PUCTConfig(merge_enabled=False)

    t_graph_a, t_key_a = _play(treatment_cfg, (0, 4, 1))
    t_graph_b, t_key_b = _play(treatment_cfg, (1, 4, 0))
    assert t_key_a == t_key_b  # merged: identical resulting board

    b_graph_a, b_key_a = _play(baseline_cfg, (0, 4, 1))
    b_graph_b, b_key_b = _play(baseline_cfg, (1, 4, 0))
    assert b_key_a != b_key_b  # never merges: distinct move paths -> distinct keys
    assert b_graph_a.nodes[b_key_a].board == b_graph_b.nodes[b_key_b].board  # same underlying board though


# ---------- expand_and_evaluate: eager children, priors, terminal detection ----------

def test_expand_and_evaluate_creates_all_legal_children_with_priors():
    width = 5
    graph = create_root((), X, width, merge_enabled=True)
    policy = {c: 1.0 / width for c in range(width)}
    evaluate_fn = _fake_evaluator(policy, value=0.5)
    value = expand_and_evaluate(graph, graph.root_key, (), width, PUCTConfig(merge_enabled=True), evaluate_fn)
    root = graph.nodes[graph.root_key]
    assert root.expanded is True
    assert value == 0.5
    assert len(root.children) == width  # every column legal on an empty board
    for col, edge in root.children.items():
        assert abs(edge.prior - 1.0 / width) < 1e-9


def test_make_node_terminal_value_is_zero_for_the_side_that_just_lost():
    from mcts_phase0.datasets.connect_four_engine import apply_move as am

    # build a simple vertical win for X, then check the resulting node (O to move) is a loss for O
    b = make_empty_board(5)
    for _ in range(4):
        b = am(b, 0, X)
    node = _make_node(b, to_move=O, just_won=True)
    assert node.is_terminal is True
    assert node.terminal_value == 0.0  # O is to_move and just lost


def test_make_node_draw_is_terminal_half_value():
    # _make_node's HEIGHT is fixed at 4 (this module's board is always 5x4),
    # so a "full" board here means every column has exactly 4 pieces -- the
    # actual X/O pattern doesn't matter, since just_won is supplied by the
    # caller (via check_win), never independently re-derived here.
    board = tuple((X, O, X, O) for _ in range(5))
    node = _make_node(board, to_move=X, just_won=False)
    assert node.is_terminal is True
    assert node.terminal_value == 0.5
