"""Toy PUCT-style search over Countdown reasoning states, testing H3: does
merging nodes by the learned projection improve accuracy at a matched
node-expansion budget, versus an identical search that never merges.

Key correctness properties, established once here rather than re-derived
per function (see the approved plan for the reasoning):
- Cycles are structurally impossible by construction: every edge goes from
  depth d to depth d+1 (merge targets are only ever looked up in the exact
  (d+1, kind) bucket), so no directed path can revisit a node. Guarded by an
  assertion at edge creation, not a runtime cycle check.
- Multi-parent pooling needs no explicit ancestor sweep: n_visits/w_value
  live on the shared Node object, so the next PUCT evaluation through ANY
  parent already reads the fully pooled Q from every prior visit via every
  other parent -- backup only ever walks the single path just traversed.
- Root never gets a proj_value (the training data has zero depth-0
  snapshots) and is designed to never need one: PUCT only reads a node's own
  stats when it's somebody's child, and root is never anyone's child.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .datasets.common import split_steps
from .model import find_answer_boundary, find_step_boundaries, generate_traces, hidden_states_for_sequence, score_rollouts
from .projection import FrozenProjection, apply_projection


@dataclass
class Edge:
    child_id: int
    prior: float
    n_edge: int = 0


@dataclass
class Node:
    node_id: int
    depth: int
    kind: str  # "root" | "step" | "answer" | "exhausted"
    full_ids: torch.Tensor
    hidden_vecs: dict[str, "object"] | None = None
    proj_value: float | None = None
    guide_value: float | None = None  # what PUCT actually reads; defaults to proj_value (see __post_init__)
    entropy: float | None = None  # average per-token generation entropy; unused by PUCT, read by search_deepsearch.py
    n_visits: int = 0
    w_value: float = 0.0
    is_terminal_correct: bool | None = None
    has_been_expanded: bool = False
    no_admissible_children: bool = False
    """Set when this node has been expanded but `select_leaf` cannot descend
    past it -- either every candidate its expansion produced was filtered as
    a duplicate, or every child it does have is itself in this state.

    Only reachable in a filtering arm. Without it, such a node keeps being
    returned as the search leaf (`select_leaf`'s loop needs `node.children`
    to descend) and `run_search` re-expands it: K fresh LLM generations that
    will be filtered again for the same reason. Measured at up to 58% of the
    arm's expansion budget spent producing nothing, invisible in every
    recorded statistic. Excluded from selection, and propagated lazily up
    the tree in `run_search`."""
    parents: list[int] = field(default_factory=list)
    children: list[Edge] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.guide_value is None and self.proj_value is not None:
            self.guide_value = self.proj_value


@dataclass
class SearchGraph:
    problem_prompt_len: int
    nodes: dict[int, Node]
    root_id: int
    depth_index: dict[tuple[int, str], list[int]] = field(default_factory=dict)
    new_nodes: int = 0
    merge_hits: int = 0
    rejected: int = 0
    expansions_used: int = 0
    """Expansions `run_search` actually spent. Recorded because it is not
    always equal to the budget: the loop can exit on its iteration cap
    instead, whenever a terminal leaf dominates PUCT in a narrow tree."""
    _next_id: int = 1

    def new_node_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def register(self, node: Node) -> None:
        self.nodes[node.node_id] = node
        self.depth_index.setdefault((node.depth, node.kind), []).append(node.node_id)


@dataclass
class SearchConfig:
    K: int
    max_new_tokens_step: int
    temperature: float
    c_puct: float
    projection: FrozenProjection
    merge_enabled: bool
    tau: float = 0.0
    reject_duplicates: bool = False
    """Admission filter: an alternative to merging, not a modifier of it.

    A candidate whose projected value lands within `tau` of an existing
    node in the same (depth, kind) bucket is DISCARDED instead of pooled
    into that node -- so the tree stays a tree, no node ever holds
    statistics from more than one path, and the backup rule is untouched.

    The motivation is the classical-domain result (see
    `results/ARTICLE_DERIVED_MERGING_FINDINGS.md`): merging does two things at once,
    deduplicating expansions and pooling their statistics, and the second
    is where the harm lives. Under an approximate equivalence relation --
    which is all a learned projection can offer -- the asymmetry is stark:
    a wrong rejection costs one wasted candidate, while a wrong merge
    corrupts a value estimate that then steers all later selection, and
    that error compounds with depth.

    A rejected candidate is treated exactly like the harness already
    treats a candidate that yields no usable node (no step/answer boundary
    within budget, or past `max_depth`): its `prior_share` is simply never
    assigned, so total prior mass under a parent is <= 1 in every arm,
    including the two that already existed."""
    reject_answer_nodes: bool = False
    """Whether the admission filter also applies to `answer` candidates.

    Default False, and the default is the point. An answer candidate can be
    VERIFIED -- ground truth is available at a terminal, and checking it
    costs one verifier call against generation the harness has already paid
    for. Filtering it on an approximate projected-value match instead would
    throw away a possibly-correct solution on the strength of the one signal
    this project has repeatedly found unreliable. Merge mode cannot avoid
    this (the node is shared, so the newcomer's own correctness is
    discarded either way); rejection can, and should.

    Set True for the arm that is symmetric with merge mode. Both variants
    are worth running: the difference between them measures exactly how
    much the equivalence relation costs when it is allowed to overrule a
    verifier."""
    max_depth: int = 6
    value_source: str = "projection"  # "projection" | "rollout" -- see C6 ablation, guide_value is what this controls
    num_rollouts: int = 4
    rollout_max_new_tokens: int = 80

    def __post_init__(self) -> None:
        if self.merge_enabled and self.reject_duplicates:
            raise ValueError("reject_duplicates is only meaningful with merge_enabled=False")


def create_root(prompt_ids: torch.Tensor, prompt_len: int) -> SearchGraph:
    root = Node(node_id=0, depth=0, kind="root", full_ids=prompt_ids)
    graph = SearchGraph(problem_prompt_len=prompt_len, nodes={}, root_id=0, _next_id=1)
    graph.nodes[0] = root  # root is never registered in depth_index -- never a merge target
    return graph


def _puct_score(parent: Node, edge: Edge, child: Node, c_puct: float) -> float:
    q = child.w_value / child.n_visits if child.n_visits > 0 else child.guide_value
    exploration = c_puct * edge.prior * math.sqrt(parent.n_visits) / (1 + edge.n_edge)
    return q + exploration


def select_leaf(graph: SearchGraph, config: SearchConfig) -> list[int]:
    path = [graph.root_id]
    node = graph.nodes[graph.root_id]
    while node.kind not in ("answer", "exhausted") and node.has_been_expanded and node.children:
        admissible = [e for e in node.children if not graph.nodes[e.child_id].no_admissible_children]
        if not admissible:
            break  # every continuation from here is blocked -- run_search marks this node
        best_edge = max(
            admissible,
            key=lambda e: _puct_score(node, e, graph.nodes[e.child_id], config.c_puct),
        )
        node = graph.nodes[best_edge.child_id]
        path.append(node.node_id)
    return path


def _find_or_add_edge(parent: Node, child_id: int, prior_share: float) -> None:
    for edge in parent.children:
        if edge.child_id == child_id:
            edge.prior += prior_share
            return
    parent.children.append(Edge(child_id=child_id, prior=prior_share))


def find_merge_target(graph: SearchGraph, new_depth: int, new_kind: str, proj_value: float, tau: float) -> int | None:
    """Among existing nodes at the exact (new_depth, new_kind) bucket, return
    the id of the closest one by |Δproj_value| if that distance is < tau,
    else None. Never looks outside this bucket -- the depth_index's bucketing
    by (depth, kind) is what makes cross-depth/cross-kind merges structurally
    impossible, not a runtime check inside this function.
    """
    bucket = graph.depth_index.get((new_depth, new_kind), [])
    best_id, best_dist = None, None
    for cand_id in bucket:
        cand = graph.nodes[cand_id]
        if cand.kind == "exhausted" or cand.no_admissible_children:
            # `register` indexes by kind at registration time and
            # `expand_node` mutates `kind` to "exhausted" without reindexing,
            # so the bucket still holds nodes already proven to go nowhere.
            # Matching one is bad in both arms -- the merge arm at least
            # keeps an edge to the dead end, while a filtering arm destroys
            # the candidate and leaves a known dead end as the sole
            # representative of its equivalence class.
            continue
        dist = abs(cand.proj_value - proj_value)
        if best_dist is None or dist < best_dist:
            best_id, best_dist = cand_id, dist
    if best_id is not None and best_dist < tau:
        return best_id
    return None


def expand_node(lm, graph: SearchGraph, node_id: int, instance, verifier_fn, config: SearchConfig) -> None:
    parent = graph.nodes[node_id]
    prior_share = 1.0 / config.K
    candidates = generate_traces(lm, parent.full_ids, config.K, config.max_new_tokens_step, config.temperature)
    layer_idx_map = _resolve_layer_indices(lm.num_hidden_layers)

    any_alive = False
    for full_ids in candidates:
        step_boundaries = find_step_boundaries(lm.tokenizer, graph.problem_prompt_len, full_ids)
        answer_idx = find_answer_boundary(lm.tokenizer, graph.problem_prompt_len, full_ids)
        new_step_idx = step_boundaries[parent.depth] if len(step_boundaries) > parent.depth else None

        if new_step_idx is None and answer_idx is None:
            continue  # dead candidate: no new boundary within budget
        if new_step_idx is not None and (answer_idx is None or new_step_idx <= answer_idx):
            boundary_idx, new_kind = new_step_idx, "step"
        else:
            boundary_idx, new_kind = answer_idx, "answer"

        new_depth = parent.depth + 1
        if new_depth > config.max_depth:
            continue

        child_full_ids = full_ids[: boundary_idx + 1]
        hidden = hidden_states_for_sequence(lm, child_full_ids)
        hidden_vecs = {
            name: hidden[idx][0, boundary_idx, :].float().cpu().numpy()
            for name, idx in layer_idx_map.items()
        }
        proj_value = apply_projection(hidden_vecs[config.projection.layer], config.projection)

        target_id = None
        if config.merge_enabled or config.reject_duplicates:
            target_id = find_merge_target(graph, new_depth, new_kind, proj_value, config.tau)

        if target_id is not None and config.reject_duplicates:
            if new_kind != "answer" or config.reject_answer_nodes:
                # Admission filter: an equivalent state is already in the
                # tree, so drop this candidate rather than pooling into it.
                #
                # Its prior share is still credited to the surviving sibling
                # when that sibling is already a child of THIS parent. That
                # is deliberate and it is what keeps `treatment` - `reject`
                # a measurement of statistic pooling alone: `_find_or_add_edge`
                # accumulates prior on a repeat target, so the merge arm turns
                # duplicate draws into prior mass ("3 of K samples landed
                # here" -- which is exactly what PUCT's P term is supposed to
                # be). Dropping it would have handed the merge arm a 1.2
                # absolute advantage in the exploration term at c_puct=1.5,
                # against a Q component that only spans [0, 1] -- a
                # first-order confound, not a rounding effect. Crediting the
                # prior shares the *policy* signal without sharing
                # `n_visits`/`w_value`, which is the split this arm is about.
                #
                # A cross-branch duplicate (the target hangs under a different
                # parent) still drops its share, exactly as the pre-existing
                # dead-candidate `continue` paths do -- there is no sibling
                # edge to credit it to.
                if any(e.child_id == target_id for e in parent.children):
                    _find_or_add_edge(parent, target_id, prior_share)
                graph.rejected += 1
                any_alive = True  # the expansion did real work; the parent is not exhausted
                continue
            # Spared answer candidate: it gets its OWN node. Clearing
            # target_id matters -- leaving it set would fall through into the
            # merge branch below and pool this candidate into the existing
            # node, which is the one thing this arm must never do.
            target_id = None

        if target_id is not None:
            target = graph.nodes[target_id]
            assert target.depth == parent.depth + 1  # regression tripwire, see module docstring
            if parent.node_id not in target.parents:
                target.parents.append(parent.node_id)
            _find_or_add_edge(parent, target_id, prior_share)
            graph.merge_hits += 1
        else:
            new_id = graph.new_node_id()
            new_node = Node(
                node_id=new_id, depth=new_depth, kind=new_kind, full_ids=child_full_ids,
                hidden_vecs=hidden_vecs, proj_value=proj_value, parents=[parent.node_id],
            )
            if config.value_source == "rollout":
                # C6 ablation: guide search with an actual rollout estimate instead of the
                # projection, while the merge decision above still used the projection --
                # isolates "fewer wasted nodes" from "value-guidance quality" (see
                # results/PLAN_GATE_REVIEW.md). Only paid for newly-created nodes, never merge
                # targets, so merge_enabled=True arms make fewer of these calls by construction.
                new_node.guide_value = score_rollouts(
                    lm, instance, verifier_fn, child_full_ids, graph.problem_prompt_len,
                    config.num_rollouts, config.rollout_max_new_tokens, config.temperature,
                )
            assert new_node.depth == parent.depth + 1
            graph.register(new_node)
            graph.new_nodes += 1
            _find_or_add_edge(parent, new_id, prior_share)
            target_id = new_id

        if new_kind == "answer" and graph.nodes[target_id].is_terminal_correct is None:
            text = lm.tokenizer.decode(
                graph.nodes[target_id].full_ids[graph.problem_prompt_len :].tolist(), skip_special_tokens=True
            )
            step_bodies, answer_body = split_steps(text)
            ok, _info = verifier_fn(instance, step_bodies, answer_body)
            graph.nodes[target_id].is_terminal_correct = ok

        any_alive = True

    parent.has_been_expanded = True
    if not any_alive:
        parent.kind = "exhausted"


def backup(graph: SearchGraph, path: list[int], value: float) -> None:
    for i, node_id in enumerate(path):
        node = graph.nodes[node_id]
        node.n_visits += 1
        node.w_value += value
        if i > 0:
            parent = graph.nodes[path[i - 1]]
            edge = next(e for e in parent.children if e.child_id == node_id)
            edge.n_edge += 1


def run_search(lm, prompt_ids: torch.Tensor, prompt_len: int, instance, verifier_fn, config: SearchConfig, budget: int) -> SearchGraph:
    graph = create_root(prompt_ids, prompt_len)
    expansions_used = 0
    iterations = 0
    max_iterations = 10 * budget
    while expansions_used < budget and iterations < max_iterations:
        iterations += 1
        path = select_leaf(graph, config)
        leaf = graph.nodes[path[-1]]
        if leaf.kind in ("answer", "exhausted"):
            value = 1.0 if leaf.is_terminal_correct else 0.0
            backup(graph, path, value)
            continue
        if leaf.has_been_expanded:
            # `select_leaf` returned an already-expanded, non-terminal node,
            # which means it could not descend past it. Pre-existing arms
            # never produce this (every non-continue path in `expand_node`
            # adds an edge, and an all-continue expansion sets kind
            # "exhausted"); a filtering arm does, whenever an expansion's
            # every candidate was a duplicate. Re-expanding regenerates the
            # same duplicates, so mark the node and refresh its own estimate
            # instead -- NOT with 0.0, which would tell the parent that a
            # merely-duplicate continuation is worthless and reintroduce the
            # value corruption this arm exists to avoid.
            leaf.no_admissible_children = True
            backup(graph, path, leaf.guide_value if leaf.node_id != graph.root_id else 0.5)
            if leaf.node_id == graph.root_id:
                break  # nothing anywhere is admissible; the search is done
            continue
        expand_node(lm, graph, leaf.node_id, instance, verifier_fn, config)
        expansions_used += 1
        graph.expansions_used = expansions_used
        value = leaf.guide_value if leaf.node_id != graph.root_id else 0.5
        backup(graph, path, value)
    return graph


def is_solved(graph: SearchGraph) -> bool:
    return any(n.kind == "answer" and n.is_terminal_correct for n in graph.nodes.values())


def _resolve_layer_indices(num_hidden_layers: int) -> dict[str, int]:
    from .model import resolve_layers

    return resolve_layers(num_hidden_layers)


def _random_walk_once(
    lm, prompt_ids: torch.Tensor, prompt_len: int, instance, verifier_fn,
    max_new_tokens_step: int, temperature: float, max_depth: int,
) -> bool:
    """One independent random walk from the root: at each step, draw a single
    LLM candidate continuation (K=1) and take it -- no tree, no PUCT, no value
    memory carried between steps or across walks. Reuses expand_node's own
    boundary-finding (find_step_boundaries / find_answer_boundary) so dead-end
    and depth-cap handling matches the tree-mode harness exactly; the only
    difference is there's nothing to select among since K=1.
    """
    current_ids = prompt_ids
    for depth in range(max_depth):
        full_ids = generate_traces(lm, current_ids, 1, max_new_tokens_step, temperature)[0]
        step_boundaries = find_step_boundaries(lm.tokenizer, prompt_len, full_ids)
        answer_idx = find_answer_boundary(lm.tokenizer, prompt_len, full_ids)
        new_step_idx = step_boundaries[depth] if len(step_boundaries) > depth else None

        if new_step_idx is None and answer_idx is None:
            return False  # dead candidate: no new boundary within budget
        if new_step_idx is not None and (answer_idx is None or new_step_idx <= answer_idx):
            current_ids = full_ids[: new_step_idx + 1]
            continue

        text = lm.tokenizer.decode(full_ids[prompt_len : answer_idx + 1].tolist(), skip_special_tokens=True)
        step_bodies, answer_body = split_steps(text)
        ok, _info = verifier_fn(instance, step_bodies, answer_body)
        return ok
    return False  # exhausted max_depth steps without reaching an answer


def run_random_search(
    lm, prompt_ids: torch.Tensor, prompt_len: int, instance, verifier_fn,
    budget: int, max_new_tokens_step: int, temperature: float, max_depth: int = 6,
) -> bool:
    """The "no MCTS at all" ablation (see RANDOM_VS_MCTS_FINDINGS.md and every
    classical module's own run_random_search): `budget` independent random
    walks from the root prompt, no tree, no PUCT, nothing carried between
    attempts. search.py has no simulate() of its own to reuse (evaluation here
    IS the search, via expand_node's batched K-candidate generation) --
    _random_walk_once is a fresh K=1 analogue of expand_node's boundary logic,
    built the same way guct_uniform_blocksworld.py's _random_rollout was when
    that module also lacked a simulate(). `budget` counts independent full
    walks here, matching run_search's own `budget` counting expansions -- the
    same cross-condition unit asymmetry already accepted for every other
    domain in this project (see RANDOM_VS_MCTS_FINDINGS.md's scope note).
    """
    for _ in range(budget):
        if _random_walk_once(lm, prompt_ids, prompt_len, instance, verifier_fn,
                              max_new_tokens_step, temperature, max_depth):
            return True
    return False
