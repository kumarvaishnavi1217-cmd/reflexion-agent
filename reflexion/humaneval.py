"""HumanEval (Chen et al., 2021): 164 Python functions, the benchmark used in the Reflexion paper."""

import ast
import json
from dataclasses import dataclass
from pathlib import Path

from reflexion.prompts import code_blocks
from reflexion.sandbox import Executor, TestResult, check_code

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "HumanEval.jsonl"


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str  # signature + docstring, the only thing the agent sees
    entry_point: str
    test: str  # hidden `check(candidate)` tests, used only for final scoring


def load_tasks(path: Path = DEFAULT_PATH, limit: int | None = None) -> list[Task]:
    tasks = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            tasks.append(Task(row["task_id"], row["prompt"], row["entry_point"], row["test"]))
    return tasks[:limit] if limit else tasks


class HumanEval:
    name = "humaneval"
    implement_system = """\
You are an expert Python programmer. Write a complete, correct implementation of the function you are given.
Reply with a single ```python code block containing the full function and any imports it needs. No other text."""
    tests_system = """\
You write unit tests for Python functions from their specification alone.
Reply with a single ```python code block of one-line `assert` statements that call the function. \
Derive expected values strictly from the docstring and the examples in it; cover edge cases. No other text."""

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = {t.task_id: t for t in tasks}

    def task_ids(self) -> list[str]:
        return list(self._tasks)

    def problem(self, task_id: str) -> str:
        return f"Implement this function:\n```python\n{self._tasks[task_id].prompt}\n```"

    def tests_prompt(self, task_id: str, n: int) -> str:
        return f"Write {n + 2} unit tests for this function:\n```python\n{self._tasks[task_id].prompt}\n```"

    def parse_tests(self, task_id: str, text: str, n: int) -> list[str]:
        """Keep syntactically valid one-line asserts that call the function (the paper filters via AST)."""
        entry_point = self._tasks[task_id].entry_point
        source = "\n".join(code_blocks(text)) or text
        tests: list[str] = []
        for line in source.splitlines():
            line = line.strip()
            if not line.startswith("assert") or entry_point not in line or line in tests:
                continue
            try:
                ast.parse(line)
            except SyntaxError:
                continue
            tests.append(line)
        return tests[:n]

    def prepare(self, task_id: str, code: str) -> str:
        # keep imports from the prompt (e.g. `from typing import List`) in case the model drops them
        spec = self._tasks[task_id].prompt
        header = "\n".join(line for line in spec.splitlines() if line.startswith(("import ", "from ")))
        return f"{header}\n\n{code}" if header else code

    def check(self, task_id: str, program: str) -> None:
        check_code(program, self._tasks[task_id].entry_point)

    def run_internal(self, executor: Executor, task_id: str, program: str, tests: list[str]) -> list[TestResult]:
        return executor.run_tests(program, tests)

    def passes_hidden(self, executor: Executor, task_id: str, program: str) -> bool:
        task = self._tasks[task_id]
        [result] = executor.run_tests(f"{program}\n\n{task.test}", [f"check({task.entry_point})"], per_test_timeout=15)
        return result.passed
