import re
import shutil
import subprocess

import pytest

from reflexion.humaneval import HumanEval, load_tasks
from reflexion.sandbox import DockerExecutor, GuardrailViolation, LocalExecutor, check_code


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("import os\ndef f(): pass", "import of `os`"),
        ("from subprocess import run\ndef f(): pass", "import of `subprocess`"),
        ("def f():\n    return open('/etc/passwd').read()", "call to `open()`"),
        ("def f():\n    return eval('1+1')", "call to `eval()`"),
        ("def f():\n    return ().__class__.__bases__[0].__subclasses__()", "dunder attribute `__subclasses__`"),
        ("def g(): pass", "function `f` is not defined"),
        ("def f(:\n  pass", "SyntaxError"),
    ],
)
def test_guardrail_rejects(code: str, message: str) -> None:
    with pytest.raises(GuardrailViolation, match=re.escape(message)):
        check_code(code, "f")


def test_guardrail_allow_list_reenables_names_for_stdin_programs() -> None:
    check_code("import sys\nn = int(input())\nprint(sys.maxsize > n)", None, allow=frozenset({"sys", "input"}))
    with pytest.raises(GuardrailViolation, match="import of `os`"):
        check_code("import os", None, allow=frozenset({"sys", "input"}))


def test_guardrail_allows_ordinary_code() -> None:
    check_code("import math\nfrom typing import List\n\ndef f(xs: List[int]) -> float:\n    return math.fsum(xs)", "f")


def test_executor_reports_each_test() -> None:
    results = LocalExecutor().run_tests("def f(x):\n    return x * 2", ["assert f(2) == 4", "assert f(2) == 5"])
    assert [r.passed for r in results] == [True, False]
    assert results[1].error.startswith("AssertionError")


def test_executor_times_out_infinite_loops() -> None:
    [result] = LocalExecutor().run_tests("def f():\n    while True: pass", ["f()"], per_test_timeout=1)
    assert not result.passed and "timed out" in result.error


def test_program_that_fails_to_load_fails_every_test() -> None:
    results = LocalExecutor().run_tests("raise ValueError('boom')", ["assert True", "assert True"])
    assert all("program failed to load" in r.error for r in results)


def test_canonical_solutions_pass_hidden_tests() -> None:
    import json

    with open("data/HumanEval.jsonl") as fh:
        rows = [json.loads(line) for line in fh][:5]
    tasks = load_tasks(limit=5)
    bench = HumanEval(tasks)
    for task, row in zip(tasks, rows, strict=True):
        assert bench.passes_hidden(LocalExecutor(), task.task_id, task.prompt + row["canonical_solution"])
        assert not bench.passes_hidden(LocalExecutor(), task.task_id, task.prompt + "    return None\n")


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0  # noqa: S607


@pytest.mark.skipif(not _docker_available(), reason="Docker not running")
def test_docker_sandbox_has_no_network_and_readonly_fs() -> None:
    probe = (
        "def net():\n    import socket\n    socket.create_connection(('1.1.1.1', 53), timeout=2)\n"
        "def write():\n    open('/etc/x', 'w').write('x')\n"
    )
    results = DockerExecutor().run_tests(probe, ["net()", "write()", "assert 1 + 1 == 2"])
    assert [r.passed for r in results] == [False, False, True]
