"""A small greedy best-first search capped at a fixed number of node
evaluations -- pyperplan's own `greedy_best_first_search` has no such cap
and would run until success or open-list exhaustion, which could take an
unbounded amount of wall-clock time on a real IPC instance. The paper's own
protocol caps every algorithm at 10,000 node evaluations; this gives exact,
direct control over that same metric rather than reinventing the search
algorithm itself (heapq-by-h-value greedy best-first is the standard,
textbook shape -- the only addition over the library version is the cap).
"""

from __future__ import annotations

import heapq
import itertools

from pyperplan.search.searchspace import make_child_node, make_root_node


def capped_gbfs(task, heuristic, node_eval_limit: int) -> tuple[bool, int]:
    """Returns (solved, node_evaluations_used)."""
    counter = itertools.count()
    root = make_root_node(task.initial_state)
    if task.goal_reached(root.state):
        return True, 1
    evaluations = 1
    heap = [(heuristic(root), next(counter), root)]
    visited = {root.state}
    while heap and evaluations < node_eval_limit:
        _, _, node = heapq.heappop(heap)
        for op, succ_state in task.get_successor_states(node.state):
            if succ_state in visited:
                continue
            visited.add(succ_state)
            child = make_child_node(node, op, succ_state)
            if task.goal_reached(succ_state):
                return True, evaluations + 1
            evaluations += 1
            heapq.heappush(heap, (heuristic(child), next(counter), child))
            if evaluations >= node_eval_limit:
                break
    return False, evaluations
