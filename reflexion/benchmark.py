from typing import Protocol

from reflexion.sandbox import Executor, TestResult


class Benchmark(Protocol):
    """What the agent needs from a benchmark. The agent sees problems and its own tests only;
    ``passes_hidden`` is for scoring after the agent has committed to an answer."""

    name: str
    implement_system: str
    tests_system: str

    def task_ids(self) -> list[str]: ...

    def problem(self, task_id: str) -> str: ...

    def tests_prompt(self, task_id: str, n: int) -> str: ...

    def parse_tests(self, task_id: str, text: str, n: int) -> list[str]: ...

    def prepare(self, task_id: str, code: str) -> str:
        """Final program from the model's code, e.g. with imports the problem implies."""
        ...

    def check(self, task_id: str, program: str) -> None:
        """Guardrail; raises ``GuardrailViolation``."""
        ...

    def run_internal(self, executor: Executor, task_id: str, program: str, tests: list[str]) -> list[TestResult]: ...

    def passes_hidden(self, executor: Executor, task_id: str, program: str) -> bool: ...
