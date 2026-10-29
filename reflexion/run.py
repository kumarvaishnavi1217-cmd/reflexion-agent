"""Run the experiment.

    python -m reflexion.run --benchmark humaneval --strategy reflexion retry --model haiku --run-name haiku-full
    python -m reflexion.run --benchmark livecodebench --difficulty hard --strategy reflexion retry --run-name lcb-hard
    python -m reflexion.run --strategy reflexion --limit 20 --run-name smoke

Results go to results/<run-name>/<strategy>.jsonl, one line per problem. Re-running the same command skips
finished problems, and LangGraph checkpoints (results/<run-name>/checkpoints.sqlite) resume a problem
mid-loop after a crash.
"""

import argparse
import json
import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from reflexion.agent import Strategy, build_graph, initial_state
from reflexion.benchmark import Benchmark
from reflexion.humaneval import HumanEval, load_tasks
from reflexion.livecodebench import LiveCodeBench, load_problems
from reflexion.llm import LLM, AnthropicLLM, ClaudeCodeLLM, LLMError
from reflexion.sandbox import DockerExecutor, Executor, LocalExecutor

log = logging.getLogger("reflexion")
RESULTS = Path(__file__).resolve().parent.parent / "results"


def solve(graph: Any, executor: Executor, benchmark: Benchmark, task_id: str, strategy: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": f"{benchmark.name}:{strategy}:{task_id}"}, "recursion_limit": 50}
    state = graph.invoke(initial_state(task_id), config)
    attempts = state["attempts"]
    # Score every attempt on the hidden tests (analysis only; the agent never sees these results)
    hidden = [benchmark.passes_hidden(executor, task_id, a["code"]) for a in attempts]
    row: dict[str, Any] = {
        "task_id": task_id,
        "strategy": strategy,
        "passed": hidden[-1],  # pass@1: the final answer the agent commits to
        "attempts": [
            {
                "passed_internal": a["passed_internal"],
                "passed_hidden": h,
                "guardrail_error": a["guardrail_error"],
            }
            for a, h in zip(attempts, hidden, strict=True)
        ],
        "n_tests": len(state.get("tests") or []),
        "tests": state.get("tests") or [],
        "reflections": state.get("reflections") or [],
        "final_code": attempts[-1]["code"],
        "llm_calls": state.get("llm_calls", 0),
        "input_tokens": state.get("input_tokens", 0),
        "output_tokens": state.get("output_tokens", 0),
    }
    if isinstance(benchmark, LiveCodeBench):
        problem = benchmark.get(task_id)
        row.update(difficulty=problem.difficulty, platform=problem.platform, contest_date=problem.contest_date)
    return row


def run(
    strategies: list[Strategy],
    llm: LLM,
    executor: Executor,
    benchmark: Benchmark,
    run_dir: Path,
    workers: int,
    **graph_kwargs: Any,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {"model": llm.name, "benchmark": benchmark.name, "tasks": len(benchmark.task_ids()), **graph_kwargs},
            indent=2,
        )
        + "\n"
    )
    conn = sqlite3.connect(run_dir / "checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    write_lock = threading.Lock()
    for strategy in strategies:
        out_path = run_dir / f"{strategy}.jsonl"
        done_rows = []
        if out_path.exists():
            done_rows = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        done = {r["task_id"] for r in done_rows}
        todo = [t for t in benchmark.task_ids() if t not in done]
        log.info("%s: %d done, %d to run", strategy, len(done), len(todo))
        graph = build_graph(llm, executor, benchmark, strategy, checkpointer=checkpointer, **graph_kwargs)
        solved, finished = sum(r["passed"] for r in done_rows), len(done_rows)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(solve, graph, executor, benchmark, t, strategy): t for t in todo}
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    row = future.result()
                except LLMError as e:
                    log.warning("%s %s failed, will retry on next run: %s", strategy, task_id, e)
                    continue
                with write_lock, out_path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                finished += 1
                solved += row["passed"]
                log.info(
                    "%s %-22s %s  attempts=%d  running pass@1 %d/%d",
                    strategy,
                    task_id,
                    "PASS" if row["passed"] else "fail",
                    len(row["attempts"]),
                    solved,
                    finished,
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=["humaneval", "livecodebench"], default="humaneval")
    parser.add_argument("--difficulty", nargs="+", choices=["easy", "medium", "hard"], help="LiveCodeBench only")
    parser.add_argument("--strategy", nargs="+", choices=["base", "retry", "reflexion"], default=["reflexion"])
    parser.add_argument("--backend", choices=["claude-code", "api"], default="claude-code")
    parser.add_argument("--model", default="haiku", help="CLI alias (haiku, sonnet, opus) or API model id")
    parser.add_argument("--limit", type=int, help="first N problems only")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--memory-size", type=int, default=3, help="reflections kept in the prompt")
    parser.add_argument("--n-tests", type=int, default=6, help="self-generated unit tests (paper: max 6)")
    parser.add_argument("--llm-timeout", type=int, default=180, help="seconds per model call (CLI backend)")
    parser.add_argument("--executor", choices=["docker", "local"], default="docker")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    llm: LLM = (
        ClaudeCodeLLM(args.model, timeout_s=args.llm_timeout)
        if args.backend == "claude-code"
        else AnthropicLLM(args.model)
    )
    executor: Executor = DockerExecutor() if args.executor == "docker" else LocalExecutor()
    benchmark: Benchmark = (
        HumanEval(load_tasks(limit=args.limit))
        if args.benchmark == "humaneval"
        else LiveCodeBench(load_problems(difficulties=set(args.difficulty or []), limit=args.limit))
    )
    run(
        args.strategy,
        llm,
        executor,
        benchmark,
        RESULTS / args.run_name,
        args.workers,
        max_attempts=args.max_attempts,
        memory_size=args.memory_size,
        n_tests=args.n_tests,
    )


if __name__ == "__main__":
    main()
