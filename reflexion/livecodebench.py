"""LiveCodeBench (Jain et al., 2024), release v6: contest problems from LeetCode and AtCoder published
January-April 2025, collected after release to limit training-data contamination.

Hidden tests ship as base64(zlib(pickle(json))). Unpickling untrusted data can execute code, so they are
loaded with an unpickler that refuses every class: only plain strings, numbers and containers come back.
"""

import base64
import io
import json
import pickle
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from reflexion.prompts import code_blocks
from reflexion.sandbox import Executor, TestResult, check_code

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "lcb_test6.jsonl"

# Imports the official LiveCodeBench harness makes available to solutions
HEADER = """\
from typing import *
from collections import *
from itertools import *
from functools import *
from heapq import *
from bisect import *
import math
import string
import re
"""


@dataclass(frozen=True)
class Problem:
    task_id: str
    title: str
    content: str
    platform: str
    difficulty: str
    contest_date: str
    starter_code: str
    testtype: Literal["stdin", "functional"]
    fn: str
    public_tests: list[dict[str, str]]
    private_blob: str


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def decode_private_tests(blob: str) -> list[dict[str, str]]:
    try:
        return json.loads(blob)  # type: ignore[no-any-return]
    except json.JSONDecodeError:
        raw = _DataOnlyUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(blob)))).load()
        return json.loads(raw)  # type: ignore[no-any-return]


def load_problems(
    path: Path = DEFAULT_PATH, difficulties: set[str] | None = None, limit: int | None = None
) -> list[Problem]:
    problems = []
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            if difficulties and row["difficulty"] not in difficulties:
                continue
            public = json.loads(row["public_test_cases"])
            fn_match = re.search(r"def (\w+)\(self", row["starter_code"])
            problems.append(
                Problem(
                    task_id=f"{row['platform']}/{row['question_id']}",
                    title=row["question_title"],
                    content=row["question_content"],
                    platform=row["platform"],
                    difficulty=row["difficulty"],
                    contest_date=row["contest_date"][:10],
                    starter_code=row["starter_code"],
                    testtype=public[0]["testtype"],
                    fn=fn_match.group(1) if fn_match else "",
                    public_tests=[{"input": t["input"], "output": t["output"]} for t in public],
                    private_blob=row["private_test_cases"],
                )
            )
    problems.sort(key=lambda p: (p.contest_date, p.task_id))
    return problems[:limit] if limit else problems


class LiveCodeBench:
    name = "livecodebench"
    implement_system = """\
You are an expert competitive programmer. Write a correct and efficient Python 3 solution: it must finish \
within a few seconds on the largest inputs the constraints allow.
Reply with a single ```python code block and no other text."""
    tests_system = """\
You write test cases for programming problems from the problem statement alone.
Reply with a single ```json code block holding a JSON list of objects with string fields "input" and "output". \
Prefer small inputs whose expected output you can work out by hand with certainty, and cover edge cases. \
No other text."""

    def __init__(self, problems: list[Problem]) -> None:
        self._problems = {p.task_id: p for p in problems}

    def task_ids(self) -> list[str]:
        return list(self._problems)

    def get(self, task_id: str) -> Problem:
        return self._problems[task_id]

    def problem(self, task_id: str) -> str:
        p = self._problems[task_id]
        if p.testtype == "functional":
            how = (
                f"Complete this starter code, keeping the class and method signature:\n```python\n{p.starter_code}\n```"
            )
        else:
            how = "Write a complete Python 3 program that reads the input from standard input and prints the answer."
        return f"Solve this problem.\n\n# {p.title}\n\n{p.content}\n\n{how}"

    def tests_prompt(self, task_id: str, n: int) -> str:
        p = self._problems[task_id]
        fmt = (
            "`input` holds the method's arguments, one JSON value per line; `output` is the JSON-encoded return value."
            if p.testtype == "functional"
            else "`input` is the exact text on standard input; `output` is the exact expected standard output."
        )
        return f"Write {n + 2} test cases for this problem. {fmt}\n\n# {p.title}\n\n{p.content}"

    def parse_tests(self, task_id: str, text: str, n: int) -> list[str]:
        p = self._problems[task_id]
        cases: list[str] = []
        for block in code_blocks(text) or [text]:
            try:
                items = json.loads(block)
            except json.JSONDecodeError:
                continue
            for item in items if isinstance(items, list) else []:
                if not (isinstance(item, dict) and isinstance(item.get("input"), str)):
                    continue
                if not isinstance(item.get("output"), str):
                    continue
                if p.testtype == "functional" and not _valid_functional(item["input"], item["output"]):
                    continue
                encoded = json.dumps({"input": item["input"], "output": item["output"]})
                if encoded not in cases:
                    cases.append(encoded)
        return cases[:n]

    def prepare(self, task_id: str, code: str) -> str:
        extra = "import sys\n" if self._problems[task_id].testtype == "stdin" else ""
        return f"{HEADER}{extra}\n{code}"

    def check(self, task_id: str, program: str) -> None:
        p = self._problems[task_id]
        if p.testtype == "stdin":
            check_code(program, None, allow=frozenset({"sys", "input"}))
        else:
            check_code(program, p.fn, allow=frozenset({"sys"}))

    def run_internal(self, executor: Executor, task_id: str, program: str, tests: list[str]) -> list[TestResult]:
        p = self._problems[task_id]
        return executor.run_io(program, [json.loads(t) for t in tests], p.testtype, p.fn)

    def passes_hidden(self, executor: Executor, task_id: str, program: str) -> bool:
        p = self._problems[task_id]
        cases = p.public_tests + decode_private_tests(p.private_blob)
        results = executor.run_io(program, cases, p.testtype, p.fn, stop_on_fail=True)
        return len(results) == len(cases) and all(r.passed for r in results)


def _valid_functional(inp: str, out: str) -> bool:
    try:
        for line in inp.split("\n"):
            if line.strip():
                json.loads(line)
        json.loads(out)
    except json.JSONDecodeError:
        return False
    return True
