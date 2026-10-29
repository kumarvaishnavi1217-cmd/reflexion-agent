import base64
import json
import pickle
import zlib

import pytest

from reflexion.livecodebench import DEFAULT_PATH, LiveCodeBench, Problem, decode_private_tests, load_problems
from reflexion.sandbox import LocalExecutor


def problem(testtype: str, **overrides: object) -> Problem:
    fields: dict[str, object] = {
        "task_id": f"demo/{testtype}",
        "title": "Sum",
        "content": "Given a and b, output a + b.",
        "platform": "atcoder" if testtype == "stdin" else "leetcode",
        "difficulty": "easy",
        "contest_date": "2025-01-01",
        "starter_code": "" if testtype == "stdin" else "class Solution:\n    def add(self, a: int, b: int) -> int:\n",
        "testtype": testtype,
        "fn": "" if testtype == "stdin" else "add",
        "public_tests": [{"input": "2 3\n", "output": "5\n"}]
        if testtype == "stdin"
        else [{"input": "2\n3", "output": "5"}],
        "private_blob": json.dumps([]),
    }
    fields.update(overrides)
    return Problem(**fields)  # type: ignore[arg-type]


def encode(obj: object) -> str:
    return base64.b64encode(zlib.compress(pickle.dumps(obj))).decode()


def test_private_tests_decode_plain_data() -> None:
    cases = [{"input": "1 1\n", "output": "2\n", "testtype": "stdin"}]
    assert decode_private_tests(encode(json.dumps(cases))) == cases


def test_private_test_decoder_refuses_code_in_pickles() -> None:
    class Exploit:
        def __reduce__(self) -> tuple[object, tuple[str]]:
            return (print, ("pwned",))

    with pytest.raises(pickle.UnpicklingError, match="refusing to load"):
        decode_private_tests(encode(Exploit()))


STDIN_OK = "a, b = map(int, sys.stdin.readline().split())\nprint(a + b)"
FUNC_OK = "class Solution:\n    def add(self, a, b):\n        return a + b"


def test_stdin_program_scored_on_hidden_tests() -> None:
    p = problem("stdin", private_blob=encode(json.dumps([{"input": "10 -4\n", "output": "6\n"}])))
    bench = LiveCodeBench([p])
    assert bench.passes_hidden(LocalExecutor(), p.task_id, bench.prepare(p.task_id, STDIN_OK))
    wrong = bench.prepare(p.task_id, "print(5)")
    assert not bench.passes_hidden(LocalExecutor(), p.task_id, wrong)  # passes public, fails private


def test_functional_program_scored_on_hidden_tests() -> None:
    p = problem("functional")
    bench = LiveCodeBench([p])
    assert bench.passes_hidden(LocalExecutor(), p.task_id, bench.prepare(p.task_id, FUNC_OK))


def test_parse_tests_validates_format() -> None:
    p = problem("functional")
    bench = LiveCodeBench([p])
    text = (
        '```json\n[{"input": "1\\n2", "output": "3"}, {"input": "1\\n2", "output": "3"},'
        ' {"input": "not json", "output": "1"}, {"input": "4\\n5", "output": 9},'
        ' {"input": "4\\n5", "output": "9"}]\n```'
    )
    assert [json.loads(t) for t in bench.parse_tests(p.task_id, text, n=6)] == [
        {"input": "1\n2", "output": "3"},
        {"input": "4\n5", "output": "9"},
    ]


def test_internal_tests_report_wrong_answers_and_timeouts() -> None:
    p = problem("stdin")
    bench = LiveCodeBench([p])
    tests = [json.dumps({"input": "2 3\n", "output": "5\n"}), json.dumps({"input": "1 1\n", "output": "3\n"})]
    results = bench.run_internal(LocalExecutor(), p.task_id, bench.prepare(p.task_id, STDIN_OK), tests)
    assert [r.passed for r in results] == [True, False]
    assert "wrong answer" in results[1].error

    slow = bench.prepare(p.task_id, "while True: pass")
    [result] = LocalExecutor().run_io(slow, [{"input": "", "output": ""}], "stdin", per_test_timeout=1)
    assert "time limit" in result.error


def test_floats_compared_with_tolerance_and_exit_is_fine() -> None:
    program = "print(1/3)\nraise SystemExit(0)"
    [result] = LocalExecutor().run_io(program, [{"input": "", "output": "0.333333333"}], "stdin")
    assert result.passed


def test_guardrail_allows_sys_for_stdin_but_requires_method_for_functional() -> None:
    from reflexion.sandbox import GuardrailViolation

    stdin_bench = LiveCodeBench([problem("stdin")])
    stdin_bench.check("demo/stdin", stdin_bench.prepare("demo/stdin", STDIN_OK))
    func_bench = LiveCodeBench([problem("functional")])
    with pytest.raises(GuardrailViolation, match="`add` is not defined"):
        func_bench.check("demo/functional", "class Solution:\n    def other(self): pass")


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="run scripts/download_data.sh first")
def test_real_dataset_loads_and_reference_style_answer_scores() -> None:
    problems = load_problems(difficulties={"hard"})
    assert len(problems) == 80
    assert {p.testtype for p in problems} == {"stdin", "functional"}
    assert all(p.contest_date >= "2025-01-01" for p in problems)
    # a trivially wrong program must fail the hidden tests
    bench = LiveCodeBench(problems[:1])
    assert not bench.passes_hidden(LocalExecutor(), problems[0].task_id, bench.prepare(problems[0].task_id, "print(0)"))
