"""Loads a real PDDL domain/problem file pair into a grounded pyperplan
`Task` -- the exact three-step pipeline pyperplan's own CLI (`planner.py`)
uses internally: parse domain, parse problem, ground. Confirmed directly
from that file's source before writing this, not assumed.

The resulting `Task` is the same class `blocksworld_engine.py` builds by
hand -- this project's existing generic engine wrappers (`legal_moves`,
`apply_move`, `is_goal`) and every function in `guct_uniform_blocksworld.py`
work on it unchanged, since none of that code is Blocksworld-specific.
"""

from __future__ import annotations

from pyperplan import grounding
from pyperplan.pddl.parser import Parser
from pyperplan.task import Task


def load_task(domain_file: str, problem_file: str) -> Task:
    parser = Parser(domain_file, problem_file)
    domain = parser.parse_domain()
    problem = parser.parse_problem(domain)
    return grounding.ground(problem)
