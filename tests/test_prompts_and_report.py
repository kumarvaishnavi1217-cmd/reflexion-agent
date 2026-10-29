from reflexion import prompts
from reflexion.humaneval import HumanEval, Task
from reflexion.report import internal_test_reliability, pass_at_attempt, summarise


def test_extract_code_prefers_fenced_block() -> None:
    assert prompts.extract_code("Sure!\n```python\ndef f():\n    return 1\n```\nDone.") == "def f():\n    return 1"
    assert prompts.extract_code("def f(): return 1") == "def f(): return 1"


def test_humaneval_parse_tests_keeps_valid_asserts_on_the_function() -> None:
    bench = HumanEval([Task("T/0", "def f(x):\n    pass\n", "f", "")])
    text = (
        "```python\nassert f(1) == 2\nassert f(1) == 2\nassert g(1) == 1\n"
        "assert f(1 ==\nprint(f(1))\nassert f(2) == 3\n```"
    )
    assert bench.parse_tests("T/0", text, n=6) == ["assert f(1) == 2", "assert f(2) == 3"]
    assert bench.parse_tests("T/0", text, n=1) == ["assert f(1) == 2"]


def test_humaneval_prepare_keeps_spec_imports() -> None:
    spec = "from typing import List\nimport math\n\n\ndef f(xs: List[int]) -> int:\n    pass"
    bench = HumanEval([Task("T/0", spec, "f", "")])
    assert bench.prepare("T/0", "def f(xs):\n    return 1").startswith("from typing import List\nimport math")


def row(*attempts: tuple[bool, bool]) -> dict:  # type: ignore[type-arg]
    return {
        "passed": attempts[-1][1],
        "attempts": [{"passed_internal": i, "passed_hidden": h, "guardrail_error": ""} for i, h in attempts],
        "llm_calls": 2 * len(attempts),
        "input_tokens": 100,
        "output_tokens": 50,
    }


ROWS = [
    row((True, True)),  # solved first try
    row((False, False), (True, True)),  # fixed on attempt 2
    row((False, True), (True, False)),  # correct first, broken by a bad test
    row((True, False)),  # false positive: internal tests passed, hidden failed
]


def test_pass_at_attempt_curve() -> None:
    assert pass_at_attempt(ROWS, 1) == 0.5
    assert pass_at_attempt(ROWS, 2) == 0.5


def test_summary_counts_fixed_and_broken() -> None:
    s = summarise(ROWS)
    assert (s["base"], s["final"], s["fixed"], s["broken"]) == (0.5, 0.5, 1, 1)


def test_reliability_conditioned_on_internal_verdict() -> None:
    rel = internal_test_reliability(ROWS)
    # internal pass: 3 correct-or-not -> TP 2 (row1, row2b), FP 2 (row3b, row4)
    assert (rel["n_pass"], rel["tp"], rel["fp"]) == (4, 0.5, 0.5)
    # internal fail: row2a (incorrect -> TN), row3a (correct -> FN)
    assert (rel["n_fail"], rel["fn"], rel["tn"]) == (2, 0.5, 0.5)


def test_mcnemar_exact() -> None:
    from reflexion.report import mcnemar_p

    assert mcnemar_p(0, 0) == 1.0
    assert mcnemar_p(5, 5) == 1.0
    assert round(mcnemar_p(10, 1), 4) == 0.0117  # binomial(11, 0.5), two-sided
