"""The LangGraph loop, driven by a scripted fake LLM and the local executor (no model calls)."""

from langgraph.checkpoint.memory import InMemorySaver

from reflexion import prompts
from reflexion.agent import build_graph, initial_state
from reflexion.humaneval import HumanEval, Task
from reflexion.llm import Completion
from reflexion.sandbox import LocalExecutor

TASK = Task(
    task_id="Demo/0",
    prompt='def add(a: int, b: int) -> int:\n    """Return a + b.\n    >>> add(2, 3)\n    5\n    """\n',
    entry_point="add",
    test="def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
)
WRONG = "```python\ndef add(a, b):\n    return a - b\n```"
RIGHT = "```python\ndef add(a, b):\n    return a + b\n```"
UNSAFE = "```python\nimport os\ndef add(a, b):\n    os.system('echo hi')\n    return a + b\n```"


class ScriptedLLM:
    name = "scripted"

    def __init__(self, implementations: list[str]) -> None:
        self.implementations = list(implementations)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> Completion:
        self.calls.append((system, prompt))
        if system == HumanEval.tests_system:
            return Completion("```python\nassert add(2, 3) == 5\nassert add(0, 0) == 0\nprint('x')\n```")
        if system == prompts.REFLECT_SYSTEM:
            return Completion("I subtracted instead of adding; use a + b.")
        return Completion(self.implementations.pop(0))

    def prompts_for(self, system: str) -> list[str]:
        return [p for s, p in self.calls if s == system]


def run(strategy: str, implementations: list[str], **kwargs: object) -> tuple[dict, ScriptedLLM]:  # type: ignore[type-arg]
    llm = ScriptedLLM(implementations)
    graph = build_graph(llm, LocalExecutor(), HumanEval([TASK]), strategy, **kwargs)  # type: ignore[arg-type]
    return graph.invoke(initial_state(TASK.task_id)), llm


def test_reflexion_reflects_on_failure_and_uses_memory_next_attempt() -> None:
    state, llm = run("reflexion", [WRONG, RIGHT])
    assert state["tests"] == ["assert add(2, 3) == 5", "assert add(0, 0) == 0"]
    assert [a["passed_internal"] for a in state["attempts"]] == [False, True]
    assert state["reflections"] == ["I subtracted instead of adding; use a + b."]
    second_prompt = llm.prompts_for(HumanEval.implement_system)[1]
    assert "I subtracted instead of adding" in second_prompt
    assert "FAILED: assert add(2, 3) == 5" in second_prompt
    assert state["llm_calls"] == 4  # tests, implement, reflect, implement


def test_retry_ablation_gets_test_feedback_but_no_reflection() -> None:
    state, llm = run("retry", [WRONG, RIGHT])
    assert len(state["attempts"]) == 2
    assert not llm.prompts_for(prompts.REFLECT_SYSTEM)
    assert "reflections" not in llm.prompts_for(HumanEval.implement_system)[1].lower()
    assert "FAILED" in llm.prompts_for(HumanEval.implement_system)[1]


def test_base_is_a_single_sample_without_tests() -> None:
    state, llm = run("base", [WRONG])
    assert len(state["attempts"]) == 1
    assert not llm.prompts_for(HumanEval.tests_system)


def test_stops_after_max_attempts() -> None:
    state, _ = run("reflexion", [WRONG] * 3, max_attempts=3)
    assert len(state["attempts"]) == 3
    assert len(state["reflections"]) == 2  # no reflection after the last attempt


def test_memory_is_bounded() -> None:
    _, llm = run("reflexion", [WRONG] * 5, max_attempts=5, memory_size=2)
    last_prompt = llm.prompts_for(HumanEval.implement_system)[-1]
    assert "1. I subtracted" in last_prompt and "2. I subtracted" in last_prompt
    assert "3. " not in last_prompt


def test_unsafe_code_is_rejected_before_execution_and_retried() -> None:
    state, llm = run("reflexion", [UNSAFE, RIGHT])
    first = state["attempts"][0]
    assert first["guardrail_error"] == "import of `os` is not allowed"
    assert not first["passed_internal"]
    assert "REJECTED before execution" in llm.prompts_for(prompts.REFLECT_SYSTEM)[0]
    assert state["attempts"][1]["passed_internal"]


def test_memory_is_checkpointed_per_thread() -> None:
    saver = InMemorySaver()
    graph = build_graph(
        ScriptedLLM([WRONG, RIGHT]), LocalExecutor(), HumanEval([TASK]), "reflexion", checkpointer=saver
    )
    config = {"configurable": {"thread_id": "reflexion:Demo/0"}}
    graph.invoke(initial_state(TASK.task_id), config)
    snapshot = graph.get_state(config)  # type: ignore[arg-type]
    assert snapshot.values["reflections"] == ["I subtracted instead of adding; use a + b."]
    assert len(list(saver.list(config))) > 4  # a checkpoint after every step  # type: ignore[arg-type]


def test_missing_import_from_spec_is_added() -> None:
    task = Task("Demo/1", "from typing import List\n\ndef first(xs: List[int]) -> int:\n    pass\n", "first", "")
    llm = ScriptedLLM(["```python\ndef first(xs: List[int]) -> int:\n    return xs[0]\n```"])
    state = build_graph(llm, LocalExecutor(), HumanEval([task]), "base").invoke(initial_state(task.task_id))
    assert state["code"].startswith("from typing import List")
