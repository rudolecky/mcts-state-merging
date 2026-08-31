"""A faithful port of DeepSearch's (arXiv:2509.25454) inference-time search
mechanism -- global frontier selection and its depth-decayed, sign-
constrained q-value backup -- into this project's existing frozen-LLM
search harness (`search.py`). Reuses `search.py`'s `Node`/`Edge`/
`SearchGraph`/`create_root`/`find_merge_target`/`_find_or_add_edge`
directly; only the selection and backup mechanics differ, so only those are
duplicated (`classical_mcts_*` modules follow the same "reuse the shared
scaffolding, write new algorithm-specific functions" pattern).

Not ported: DeepSearch's actual RL training loop (Tree-GRPO, q-value soft
clipping) -- that trains a 128xH100-cluster-scale policy update, wildly
outside this project's scope. What's here is the search mechanism alone,
run at inference time against a frozen model, exactly how every other
paper-comparison module in this project reimplements an algorithm's
formulas without its surrounding infrastructure.

Two genuine fidelity notes, not glossed over:
- The extracted paper text has an OCR artifact in its constrained backup
  equation (two branches read identically). This module's `backup_constrained`
  is a best-faith reconstruction preserving the paper's own stated invariant
  (q(s_i) >= 0 for nodes on any path that reached a correct solution), not a
  verified transcription -- see that function's docstring.
- The paper doesn't state a batch size for "repeat K iterations" before a
  backup phase (its only stated K is an unrelated Pass@K filtering
  threshold). Exposed here as `iterations_per_backup_round`, a new,
  clearly-named hyperparameter this project introduces, not a guess dressed
  up as the paper's own number.

IMPORTANT semantic note on the shared `Node.w_value` field: `search.py`'s
own PUCT reads it as a running SUM (divided by `n_visits` for the mean).
This module's `backup_constrained` instead writes the CURRENT q-value
directly into `w_value` (Eq. 3/5 are an iterative update rule, not an
average) -- so anywhere this module reads `w_value`, it is read directly,
never divided by `n_visits`. The two algorithms never read each other's
graphs, so this dual meaning is safe, but it's a real semantic split worth
knowing about the shared field, not an oversight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .model import find_answer_boundary, find_step_boundaries, generate_traces, hidden_states_for_sequence, next_token_entropy_from_hidden
from .projection import FrozenProjection, apply_projection
from .search import Edge, Node, SearchGraph, _find_or_add_edge, create_root, find_merge_target


@dataclass
class DeepSearchConfig:
    K: int
    max_new_tokens_step: int
    temperature: float
    projection: FrozenProjection
    merge_enabled: bool
    tau: float = 0.0
    max_depth: int = 6
    lambda1: float = 0.4  # quality potential weight, tanh(Q_parent)
    lambda2: float = 0.4  # uncertainty bonus weight, step entropy
    lambda3: float = 0.01  # depth bonus weight
    gamma_min: float = 0.1
    iterations_per_backup_round: int = 4


def _resolve_layer_indices(num_hidden_layers: int) -> dict[str, int]:
    from .model import resolve_layers

    return resolve_layers(num_hidden_layers)


def _q_parent(graph: SearchGraph, node: Node) -> float:
    """Mean of each parent's own current q (`w_value`, stored directly by
    `backup_constrained` -- not divided by n_visits, see module docstring)
    over parents that have been visited at least once. 0.0 (neutral) if no
    parent has any accumulated evidence yet -- a node can have several
    parents under merge_enabled=True; this degenerates exactly to the
    paper's own single-parent Q_parent(s) when there's only one."""
    visited_parents = [graph.nodes[pid] for pid in node.parents if graph.nodes[pid].n_visits > 0]
    if not visited_parents:
        return 0.0
    return sum(p.w_value for p in visited_parents) / len(visited_parents)


def _frontier_score(graph: SearchGraph, node: Node, config: DeepSearchConfig) -> float:
    """F(s) = lambda1*tanh(Q_parent(s)) + lambda2*H(pi(s|o_s)) + lambda3*D(depth(s)),
    D(depth(s)) = sqrt(depth(s)/max_depth) -- the paper's own best-performing
    depth-bonus variant (Table 3)."""
    quality = math.tanh(_q_parent(graph, node))
    uncertainty = node.entropy if node.entropy is not None else 0.0
    depth_bonus = math.sqrt(node.depth / config.max_depth) if config.max_depth > 0 else 0.0
    return config.lambda1 * quality + config.lambda2 * uncertainty + config.lambda3 * depth_bonus


def select_frontier_global(graph: SearchGraph, config: DeepSearchConfig) -> int | None:
    """Scores every current frontier node (not yet expanded, not terminal)
    across the WHOLE tree and returns the global argmax -- replaces
    root-to-leaf UCT descent entirely. None if nothing is left to expand."""
    frontier = [n for n in graph.nodes.values() if not n.has_been_expanded and n.kind not in ("answer", "exhausted")]
    if not frontier:
        return None
    best = max(frontier, key=lambda n: _frontier_score(graph, n, config))
    return best.node_id


def _reconstruct_path_to_root(graph: SearchGraph, node_id: int) -> list[int]:
    """Global selection has no natural root-to-leaf path the way local UCT
    descent does, so one is reconstructed after the fact by walking up via
    each node's own first recorded parent -- any valid ancestor chain is
    sufficient for propagating a backup value to the root; the tree is
    structurally acyclic (search.py's own edges always go depth d -> d+1),
    so this walk always terminates."""
    path = [node_id]
    current = graph.nodes[node_id]
    while current.node_id != graph.root_id:
        parent_id = current.parents[0]
        path.append(parent_id)
        current = graph.nodes[parent_id]
    path.reverse()
    return path


def expand_node_entropy(lm, graph: SearchGraph, node_id: int, instance, verifier_fn, config: DeepSearchConfig) -> list[int]:
    """Near-duplicate of search.py's expand_node: identical candidate
    generation / merge-target lookup, plus the per-step average-token
    entropy DeepSearch's frontier score needs, and returns the list of
    child node ids touched (new or merged-into) this call, since the caller
    needs to know which of them are newly-reached "answer" nodes."""
    parent = graph.nodes[node_id]
    prior_share = 1.0 / config.K  # unused by this algorithm's own selection; kept only for _find_or_add_edge's shape
    candidates = generate_traces(lm, parent.full_ids, config.K, config.max_new_tokens_step, config.temperature)
    layer_idx_map = _resolve_layer_indices(lm.num_hidden_layers)
    final_layer_idx = layer_idx_map["final"]
    parent_len_tokens = parent.full_ids.shape[0]

    touched_ids: list[int] = []
    any_alive = False
    for full_ids in candidates:
        step_boundaries = find_step_boundaries(lm.tokenizer, graph.problem_prompt_len, full_ids)
        answer_idx = find_answer_boundary(lm.tokenizer, graph.problem_prompt_len, full_ids)
        new_step_idx = step_boundaries[parent.depth] if len(step_boundaries) > parent.depth else None

        if new_step_idx is None and answer_idx is None:
            continue
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

        final_hidden = hidden[final_layer_idx][0]
        token_entropies = [
            next_token_entropy_from_hidden(lm, final_hidden[pos].float().cpu().numpy())
            for pos in range(parent_len_tokens - 1, boundary_idx)
        ]
        step_entropy = float(np.mean(token_entropies)) if token_entropies else 0.0

        target_id = None
        if config.merge_enabled:
            target_id = find_merge_target(graph, new_depth, new_kind, proj_value, config.tau)

        if target_id is not None:
            target = graph.nodes[target_id]
            assert target.depth == parent.depth + 1
            if parent.node_id not in target.parents:
                target.parents.append(parent.node_id)
            _find_or_add_edge(parent, target_id, prior_share)
        else:
            new_id = graph.new_node_id()
            new_node = Node(
                node_id=new_id, depth=new_depth, kind=new_kind, full_ids=child_full_ids,
                hidden_vecs=hidden_vecs, proj_value=proj_value, parents=[parent.node_id],
                entropy=step_entropy,
            )
            assert new_node.depth == parent.depth + 1
            graph.register(new_node)
            _find_or_add_edge(parent, new_id, prior_share)
            target_id = new_id

        if new_kind == "answer" and graph.nodes[target_id].is_terminal_correct is None:
            from .datasets.common import split_steps

            text = lm.tokenizer.decode(
                graph.nodes[target_id].full_ids[graph.problem_prompt_len :].tolist(), skip_special_tokens=True
            )
            step_bodies, answer_body = split_steps(text)
            ok, _info = verifier_fn(instance, step_bodies, answer_body)
            graph.nodes[target_id].is_terminal_correct = ok

        touched_ids.append(target_id)
        any_alive = True

    parent.has_been_expanded = True
    if not any_alive:
        parent.kind = "exhausted"
    return touched_ids


def backup_constrained(graph: SearchGraph, path: list[int], value: float, gamma_min: float) -> None:
    """Eq. 3/5: depth-decayed update gamma(i,l) = max(i/l, gamma_min), i the
    node's 1-based position in the trajectory, l the terminal's own
    position. The terminal node's own q is `value` directly (+1 correct,
    -1 incorrect, Eq. 4). For intermediate nodes: add the decayed value
    normally; if that would flip an already-positive q negative, keep the
    existing q unchanged instead of letting one bad trajectory erase
    accumulated positive evidence -- this preserves the paper's own stated
    invariant (q(s_i) >= 0 for nodes on any path that reached a correct
    solution). The extracted Eq. 5 has a real OCR artifact (two of its three
    branches read identically); this is a best-faith reconstruction of the
    missing third branch, not a verified transcription -- flagged here and
    in the findings doc, not presented as certain.
    """
    l = len(path)
    terminal = graph.nodes[path[-1]]
    terminal.n_visits += 1
    terminal.w_value = value

    for i in range(1, l):  # 1-based index over intermediate nodes s_1..s_{l-1}; terminal (index l) handled above
        node = graph.nodes[path[i - 1]]
        node.n_visits += 1
        gamma = max(i / l, gamma_min)
        delta = gamma * value
        prev_q = node.w_value
        if prev_q * delta >= 0:
            node.w_value = prev_q + delta
        elif prev_q > 0:
            pass  # would flip a positive q negative -- keep it, per the invariant above
        else:
            node.w_value = delta


def run_search_deepsearch(lm, prompt_ids, prompt_len: int, instance, verifier_fn, config: DeepSearchConfig, budget: int) -> SearchGraph:
    """Figure 1's own rhythm: repeat `iterations_per_backup_round` rounds of
    {global frontier select, expand} before a single backup phase that
    targets either a newly-reached correct trajectory, or (Eq. 2) the
    lowest-average-entropy incorrect one among everything reached since the
    last backup -- a real structural difference from search.py's
    per-expansion backup loop, ported faithfully rather than flattened."""
    graph = create_root(prompt_ids, prompt_len)
    expansions_used = 0
    iterations = 0
    max_iterations = 10 * budget

    while expansions_used < budget and iterations < max_iterations:
        newly_reached_terminals: list[int] = []
        frontier_exhausted = False
        for _ in range(config.iterations_per_backup_round):
            if expansions_used >= budget or iterations >= max_iterations:
                break
            iterations += 1
            node_id = select_frontier_global(graph, config)
            if node_id is None:
                frontier_exhausted = True
                break
            touched = expand_node_entropy(lm, graph, node_id, instance, verifier_fn, config)
            expansions_used += 1
            newly_reached_terminals.extend(cid for cid in touched if graph.nodes[cid].kind == "answer")

        if not newly_reached_terminals:
            if frontier_exhausted:
                break
            continue  # this round made progress but reached no terminal yet -- keep going

        correct = [cid for cid in newly_reached_terminals if graph.nodes[cid].is_terminal_correct]
        if correct:
            target_id, value = correct[0], 1.0
        else:
            target_id = min(
                newly_reached_terminals,
                key=lambda cid: graph.nodes[cid].entropy if graph.nodes[cid].entropy is not None else float("inf"),
            )
            value = -1.0

        path = _reconstruct_path_to_root(graph, target_id)
        backup_constrained(graph, path, value, config.gamma_min)

    return graph


def is_solved(graph: SearchGraph) -> bool:
    return any(n.kind == "answer" and n.is_terminal_correct for n in graph.nodes.values())
