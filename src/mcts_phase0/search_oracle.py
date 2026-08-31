"""Distinguishes the two candidate mechanisms left open in
RANDOM_VS_MCTS_FINDINGS.md's "Why doesn't real guidance prevent the
persistent gap on ProsQA?": (1) the learned projection's signal is too
weak/noisy to give PUCT a meaningfully better exploitation gradient than
blind UCB1, vs. (2) budget=15 is structurally too thin for how PUCT
allocates it (one leaf expanded by one depth level per iteration),
regardless of guidance quality.

`oracle_value` replaces the learned projection with a perfect, noise-free
ground-truth value computed directly from a ProsQA instance's own known
fact graph (no LLM, no training data, no possible noise): 1.0 if the
partial reasoning trace so far is well-formed and its current end can still
reach the target, 0.0 the moment it goes wrong. Running PUCT with this
oracle at the same budget=15 isolates the question: if tree-with-oracle
still loses badly to random restarts, the budget-allocation mechanism (2)
is the real culprit, independent of guidance quality; if it closes the gap,
the original learned projection (mechanism 1) was simply too weak.

Reuses search.py's Node/Edge/SearchGraph/SearchConfig/create_root/
select_leaf/backup/find_merge_target/_find_or_add_edge/is_solved unchanged
-- only expand_node needed a fork, to swap the projection-based guide_value
for the oracle (mirroring search_deepsearch.py's own expand_node_entropy
precedent for the same reason: one differing step doesn't justify
threading a new parameter through the shared, already-closed expand_node).
"""

from __future__ import annotations

import re
from collections import deque

import torch

from .datasets.common import split_steps
from .model import find_answer_boundary, find_step_boundaries, generate_traces, hidden_states_for_sequence
from .projection import apply_projection
from .search import Node, SearchConfig, SearchGraph, _find_or_add_edge, _resolve_layer_indices, backup, create_root, find_merge_target, select_leaf

_LINE_RE = re.compile(r"^\s*(.+?)\s+is a\s+(.+?)\s*\.?\s*$")


def _can_reach(fact_set: set[tuple[str, str]], start: str, target: str) -> bool:
    successors: dict[str, list[str]] = {}
    for u, v in fact_set:
        successors.setdefault(u, []).append(v)
    seen = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        for nxt in successors.get(node, []):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def oracle_value(instance, tokenizer, full_ids: torch.Tensor, prompt_len: int) -> float:
    """1.0 if the reasoning trace generated so far is well-formed (every
    step a real fact, chained, starting at instance.start) and its current
    end can still reach instance.target; 0.0 the moment it's already wrong.
    Ground truth from the instance's own fact graph -- no LLM involved.
    """
    text = tokenizer.decode(full_ids[prompt_len:].tolist(), skip_special_tokens=True)
    step_bodies, _answer_body = split_steps(text)
    fact_set = set(instance.facts)

    chain: list[tuple[str, str]] = []
    for body in step_bodies:
        m = _LINE_RE.match(body)
        if not m:
            return 0.0
        x, y = m.group(1).strip().rstrip("."), m.group(2).strip().rstrip(".")
        if (x, y) not in fact_set:
            return 0.0
        chain.append((x, y))

    if not chain:
        return 1.0  # no steps yet -- nothing to be wrong about
    if chain[0][0] != instance.start:
        return 0.0
    for i in range(len(chain) - 1):
        if chain[i][1] != chain[i + 1][0]:
            return 0.0

    current_end = chain[-1][1]
    if current_end == instance.target:
        return 1.0
    return 1.0 if _can_reach(fact_set, current_end, instance.target) else 0.0


def expand_node_oracle(lm, graph: SearchGraph, node_id: int, instance, verifier_fn, config: SearchConfig) -> None:
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

        target_id = None
        if config.merge_enabled:
            target_id = find_merge_target(graph, new_depth, new_kind, proj_value, config.tau)

        if target_id is not None:
            target = graph.nodes[target_id]
            if parent.node_id not in target.parents:
                target.parents.append(parent.node_id)
            _find_or_add_edge(parent, target_id, prior_share)
        else:
            new_id = graph.new_node_id()
            new_node = Node(
                node_id=new_id, depth=new_depth, kind=new_kind, full_ids=child_full_ids,
                hidden_vecs=hidden_vecs, proj_value=proj_value, parents=[parent.node_id],
            )
            new_node.guide_value = oracle_value(instance, lm.tokenizer, child_full_ids, graph.problem_prompt_len)
            graph.register(new_node)
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


def run_search_oracle(lm, prompt_ids: torch.Tensor, prompt_len: int, instance, verifier_fn, config: SearchConfig, budget: int) -> SearchGraph:
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
        expand_node_oracle(lm, graph, leaf.node_id, instance, verifier_fn, config)
        expansions_used += 1
        value = leaf.guide_value if leaf.node_id != graph.root_id else 0.5
        backup(graph, path, value)
    return graph
