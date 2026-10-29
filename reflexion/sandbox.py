"""Guardrails and isolated execution for model-written code.

Two layers: a static AST check rejects code that imports or calls anything that could touch the host,
and execution happens in a throwaway Docker container with no network, a read-only filesystem and memory,
CPU and process limits. ``LocalExecutor`` (subprocess + rlimits) exists for tests and CI only.
"""

import ast
import json
import resource
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes", "multiprocessing", "threading",
    "signal", "importlib", "pickle", "marshal", "urllib", "http", "requests", "ftplib", "asyncio", "builtins",
}  # fmt: skip
BLOCKED_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "globals", "vars"}
SAFE_DUNDERS = {"__name__", "__doc__", "__class__", "__init__", "__len__", "__main__"}


class GuardrailViolation(ValueError):
    pass


def check_code(code: str, required_function: str | None, allow: frozenset[str] = frozenset()) -> None:
    """Raise ``GuardrailViolation`` unless ``code`` parses, defines ``required_function`` (if given) and avoids
    blocked modules/calls. ``allow`` re-enables specific names, e.g. ``sys`` and ``input`` for stdin programs."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise GuardrailViolation(f"SyntaxError: {e.msg} (line {e.lineno})") from e
    if required_function:
        defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
        if required_function not in defined:
            raise GuardrailViolation(f"function `{required_function}` is not defined")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root in BLOCKED_MODULES and root not in allow:
                    raise GuardrailViolation(f"import of `{name}` is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS and node.func.id not in allow:
                raise GuardrailViolation(f"call to `{node.func.id}()` is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in SAFE_DUNDERS:
                raise GuardrailViolation(f"dunder attribute `{node.attr}` is not allowed")


@dataclass(frozen=True)
class TestResult:
    test: str
    passed: bool
    error: str = ""


# Runs inside the sandbox: defines the program, then each assertion with its own time limit.
_ASSERT_RUNNER = """
import json, signal, sys
def _timeout(*_): raise TimeoutError("timed out")
signal.signal(signal.SIGALRM, _timeout)
payload = json.loads(sys.stdin.read())
ns = {}
results = []
try:
    signal.alarm(5)
    exec(payload["program"], ns)
    signal.alarm(0)
except BaseException as e:
    print(json.dumps([{"test": t, "passed": False, "error": f"program failed to load: {type(e).__name__}: {e}"}
                      for t in payload["tests"]]))
    sys.exit(0)
for t in payload["tests"]:
    try:
        signal.alarm(payload["per_test_timeout"])
        exec(t, ns)
        signal.alarm(0)
        results.append({"test": t, "passed": True, "error": ""})
    except BaseException as e:
        signal.alarm(0)
        msg = f"{type(e).__name__}: {e}"[:300]
        results.append({"test": t, "passed": False, "error": msg})
print(json.dumps(results))
"""

# Input/output test cases, LiveCodeBench style: "stdin" programs read input and print the answer;
# "functional" programs define class Solution whose method gets one JSON value per input line.
_IO_RUNNER = """
import io, json, math, signal, sys
def _timeout(*_): raise TimeoutError("time limit exceeded")
signal.signal(signal.SIGALRM, _timeout)
p = json.loads(sys.stdin.read())
real_stdout = sys.stdout
def same(expected, got):
    a, b = expected.split(), got.split()
    if a == b: return True
    if len(a) != len(b): return False
    for x, y in zip(a, b):
        if x == y: continue
        try:
            if not math.isclose(float(x), float(y), rel_tol=1e-6, abs_tol=1e-6): return False
        except ValueError:
            return False
    return True
def show(s): return s if len(s) <= 200 else s[:200] + "..."
results = []
try:
    code = compile(p["program"], "<solution>", "exec")
except SyntaxError as e:
    print(json.dumps([{"test": "compile", "passed": False, "error": f"SyntaxError: {e}"}])); sys.exit(0)
for case in p["cases"]:
    label = f"input={show(case['input'])!r} expected={show(case['output'])!r}"
    out = io.StringIO()
    try:
        signal.alarm(p["per_test_timeout"])
        ns = {"__name__": "__main__"}
        sys.stdout = out
        if p["testtype"] == "stdin":
            sys.stdin = io.TextIOWrapper(io.BytesIO(case["input"].encode()))
            try:
                exec(code, ns)
            except SystemExit as e:
                if e.code not in (None, 0): raise
            sys.stdout = real_stdout
            signal.alarm(0)
            ok = same(case["output"], out.getvalue())
            err = "" if ok else f"wrong answer, got {show(out.getvalue())!r}"
        else:
            exec(code, ns)
            args = [json.loads(line) for line in case["input"].split("\\n") if line.strip()]
            got = getattr(ns["Solution"](), p["fn"])(*args)
            sys.stdout = real_stdout
            signal.alarm(0)
            ok = json.loads(json.dumps(got)) == json.loads(case["output"])
            err = "" if ok else f"wrong answer, got {show(json.dumps(got))!r}"
    except BaseException as e:
        signal.alarm(0)
        sys.stdout = real_stdout
        ok, err = False, f"{type(e).__name__}: {e}"[:300]
    results.append({"test": label, "passed": ok, "error": err})
    if not ok and p["stop_on_fail"]:
        break
print(json.dumps(results))
"""


class Executor(Protocol):
    def run_tests(self, program: str, tests: list[str], per_test_timeout: int = 3) -> list[TestResult]: ...

    def run_io(
        self,
        program: str,
        cases: list[dict[str, str]],
        testtype: Literal["stdin", "functional"],
        fn: str = "",
        per_test_timeout: int = 6,
        stop_on_fail: bool = False,
    ) -> list[TestResult]: ...


class _BaseExecutor:
    total_timeout = 180

    def _command(self, runner: str) -> list[str]:
        raise NotImplementedError

    def _preexec(self) -> Callable[[], None] | None:
        return None

    def _execute(self, runner: str, payload: dict[str, Any], labels: list[str]) -> list[TestResult]:
        try:
            proc = subprocess.run(  # noqa: S603 - fixed argv, code goes in via stdin
                self._command(runner),
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.total_timeout,
                check=False,
                preexec_fn=self._preexec(),
            )
            rows = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            return [TestResult(t, False, "TimeoutError: sandbox time limit") for t in labels]
        except (json.JSONDecodeError, IndexError):
            return [TestResult(t, False, f"sandbox error: {proc.stderr[-300:]}") for t in labels]
        return [TestResult(r["test"], r["passed"], r["error"]) for r in rows]

    def run_tests(self, program: str, tests: list[str], per_test_timeout: int = 3) -> list[TestResult]:
        payload = {"program": program, "tests": tests, "per_test_timeout": per_test_timeout}
        return self._execute(_ASSERT_RUNNER, payload, tests)

    def run_io(
        self,
        program: str,
        cases: list[dict[str, str]],
        testtype: Literal["stdin", "functional"],
        fn: str = "",
        per_test_timeout: int = 6,
        stop_on_fail: bool = False,
    ) -> list[TestResult]:
        payload = {
            "program": program,
            "cases": cases,
            "testtype": testtype,
            "fn": fn,
            "per_test_timeout": per_test_timeout,
            "stop_on_fail": stop_on_fail,
        }
        return self._execute(_IO_RUNNER, payload, ["io"])


class DockerExecutor(_BaseExecutor):
    def __init__(self, image: str = "python:3.12-slim", memory: str = "1g") -> None:
        self.image = image
        self.memory = memory

    def _command(self, runner: str) -> list[str]:
        return [
            "docker", "run", "--rm", "-i",
            "--network", "none", "--read-only", "--tmpfs", "/tmp:size=16m",  # noqa: S108 - tmpfs inside the container
            "--memory", self.memory, "--cpus", "1", "--pids-limit", "64",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
            self.image, "python", "-I", "-c", runner,
        ]  # fmt: skip


class LocalExecutor(_BaseExecutor):
    """Subprocess with resource limits. Not an isolation boundary: use only for trusted code in tests."""

    def _command(self, runner: str) -> list[str]:
        return [sys.executable, "-I", "-c", runner]

    def _preexec(self) -> Callable[[], None] | None:
        def limits() -> None:
            resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))

        return limits
