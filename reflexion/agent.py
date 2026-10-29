"""The Reflexion loop as a LangGraph state machine.

    generate_tests ─► implement ─► evaluate ──(internal tests pass, or out of attempts)──► END
                          ▲            │
                          │            ├─ strategy "reflexion": ─► reflect ─┐  (reflection appended to memory)
                          │            └─ strategy "retry": ────────────────┤  (test feedback only: ablation)
                          └─────────────────────────────────────────────────┘

The episodic memory (``reflections``) is part of the graph state, so it is checkpointed after every step and
a run can be resumed mid-problem. The agent only ever sees its own generated tests; the benchmark's hidden
tests are applied afterwards, outside the graph.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from reflexion import prompts
from reflexion.benchmark import Benchmark
from reflexion.llm import LLM
from reflexion.sandbox import Executor, GuardrailViolation

Strategy = Literal["base", "retry", "reflexion"]


class Attempt(TypedDict):
    code: str
    passed_internal: bool
    guardrail_error: str
    feedback: str


class AgentState(TypedDict, total=False):
    task_id: str
    tests: list[str]
    code: str
    guardrail_error: str
    feedback: str
    passed_internal: bool
    reflections: Annotated[list[str], operator.add]  # episodic memory, grows across attempts
    attempts: Annotated[list[Attempt], operator.add]
    llm_calls: Annotated[int, operator.add]
    input_tokens: Annotated[int, operator.add]
    output_tokens: Annotated[int, operator.add]


def build_graph(
    llm: LLM,
    executor: Executor,
    benchmark: Benchmark,
    strategy: Strategy,
    max_attempts: int = 4,
    memory_size: int = 3,
    n_tests: int = 6,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    def generate_tests(state: AgentState) -> dict[str, Any]:
        task_id = state["task_id"]
        out = llm.complete(benchmark.tests_system, benchmark.tests_prompt(task_id, n_tests))
        return {
            "tests": benchmark.parse_tests(task_id, out.text, n_tests),
            "llm_calls": 1,
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
        }

    def implement(state: AgentState) -> dict[str, Any]:
        task_id = state["task_id"]
        retrying = bool(state.get("attempts"))
        prompt = prompts.implement_prompt(
            benchmark.problem(task_id),
            previous_code=state.get("code", "") if retrying else "",
            feedback=state.get("feedback", "") if retrying else "",
            reflections=state.get("reflections", [])[-memory_size:] if strategy == "reflexion" else None,
        )
        out = llm.complete(benchmark.implement_system, prompt)
        program = benchmark.prepare(task_id, prompts.extract_code(out.text))
        guardrail_error = ""
        try:
            benchmark.check(task_id, program)
        except GuardrailViolation as e:
            guardrail_error = str(e)
        return {
            "code": program,
            "guardrail_error": guardrail_error,
            "llm_calls": 1,
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
        }

    def evaluate(state: AgentState) -> dict[str, Any]:
        if state["guardrail_error"]:
            passed, feedback = False, f"REJECTED before execution: {state['guardrail_error']}"
        elif strategy == "base" or not state.get("tests"):
            passed, feedback = True, ""  # nothing to check against: accept the single sample
        else:
            results = benchmark.run_internal(executor, state["task_id"], state["code"], state["tests"])
            passed = bool(results) and all(r.passed for r in results)
            feedback = prompts.format_feedback([(r.test, r.passed, r.error) for r in results])
        attempt: Attempt = {
            "code": state["code"],
            "passed_internal": passed,
            "guardrail_error": state["guardrail_error"],
            "feedback": feedback,
        }
        return {"passed_internal": passed, "feedback": feedback, "attempts": [attempt]}

    def reflect(state: AgentState) -> dict[str, Any]:
        problem = benchmark.problem(state["task_id"])
        out = llm.complete(prompts.REFLECT_SYSTEM, prompts.reflect_prompt(problem, state["code"], state["feedback"]))
        return {
            "reflections": [out.text.strip()],
            "llm_calls": 1,
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
        }

    def next_step(state: AgentState) -> str:
        if strategy == "base" or state["passed_internal"] or len(state["attempts"]) >= max_attempts:
            return END
        return "reflect" if strategy == "reflexion" else "implement"

    graph = StateGraph(AgentState)
    graph.add_node("generate_tests", generate_tests)
    graph.add_node("implement", implement)
    graph.add_node("evaluate", evaluate)
    graph.add_node("reflect", reflect)
    graph.add_edge(START, "implement" if strategy == "base" else "generate_tests")
    graph.add_edge("generate_tests", "implement")
    graph.add_edge("implement", "evaluate")
    graph.add_conditional_edges("evaluate", next_step, ["reflect", "implement", END])
    graph.add_edge("reflect", "implement")
    return graph.compile(checkpointer=checkpointer)


def initial_state(task_id: str) -> AgentState:
    return {"task_id": task_id}
