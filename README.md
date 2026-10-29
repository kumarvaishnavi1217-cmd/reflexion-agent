# Reflexion, replicated with LangGraph

A replication of **Reflexion: Language Agents with Verbal Reinforcement Learning** (Shinn et al., NeurIPS 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) on code generation. The agent:
- writes its own unit tests;
- implements a solution and runs it against those tests in a sandbox;
- when the tests fail, writes a **self-reflection** that is stored in **episodic memory** and fed into the next attempt.

The experiments test the paper's claims on a current model (Claude Haiku 5.5):
1. **HumanEval**, the paper's benchmark;
2. **LiveCodeBench v6**, hard contest problems from 2025, because HumanEval turned out to be saturated;

each with the paper's **"no self-reflection" ablation** and its analysis of how reliable self-generated tests are.

## Agent design (LangGraph)

```
generate_tests ─► implement ─► evaluate ──(internal tests pass, or 4 attempts used)──► END
                      ▲            │
                      │            ├─ Reflexion: ─► reflect ─┐   reflection appended to episodic memory
                      │            └─ ablation: ─────────────┤   test feedback only, no reflection
                      └───────────────────────────────────────┘
```

- **State and memory:** reflections, attempts and test results are LangGraph graph state. A SQLite checkpointer saves the state after every step, so a crashed run resumes mid-problem. Memory is bounded to the last 3 reflections, following the paper's bounded memory.
- **Guardrails:** model-written code passes a static AST check before it runs. Blocked: `os`, `subprocess`, `socket`, `eval`/`exec`, `open`, dunder tricks, and similar. It then runs in a throwaway Docker container with no network, a read-only filesystem, memory/CPU/PID limits, all capabilities dropped and a non-root user. Hidden test data from LiveCodeBench is a pickle; it's decoded with an unpickler that refuses every class, because unpickling untrusted data can execute code.
- **No test leakage:** the agent only sees its own generated tests. Benchmark tests are applied after the agent commits to an answer, outside the graph.
- **Benchmark-agnostic:** the loop talks to a small `Benchmark` interface. HumanEval uses assert tests; LiveCodeBench uses stdin/stdout and LeetCode-style function tests.
- **Cost:** model calls go through the Claude Code CLI (`claude -p`) on a Claude subscription. The CLI's tools are disabled and its system prompt is replaced, so each call is a plain model call. A pay-per-token Anthropic API backend is included as well.

## Results

### 1. HumanEval (164 problems): saturated

| | first attempt | final pass@1 |
|---|---|---|
| Paper, GPT-4 | 80.1% | 91.0% (Reflexion) |
| **Claude Haiku 5.5, Reflexion** | **100%** | **100%** |
| Claude Haiku 5.5, retry without reflection | 100% | 100% |

Haiku 5.5 solves every problem on its first attempt, so Reflexion has nothing left to fix. In the few cases where the loop retried, the self-generated test was wrong, not the code: every internal "fail" was a false negative. HumanEval (2021) is likely in the training data of current models, so it can no longer measure this technique. Full report: [results/haiku-full/REPORT.md](results/haiku-full/REPORT.md).

### 2. LiveCodeBench v6, hard (80 problems, January–April 2025)

*Run in progress; results will be added here.*

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
scripts/download_data.sh                      # HumanEval, LiveCodeBench v6, sandbox image
claude auth login                             # or use --backend api with ANTHROPIC_API_KEY

.venv/bin/python -m reflexion.run --benchmark humaneval --strategy reflexion retry --run-name he
.venv/bin/python -m reflexion.run --benchmark livecodebench --difficulty hard --strategy reflexion retry --run-name lcb
.venv/bin/python -m reflexion.report --run-name lcb    # REPORT.md + accuracy-vs-attempts plot
```

Options: `--model haiku|sonnet|opus`, `--max-attempts 4`, `--memory-size 3`, `--n-tests 6` (the paper's maximum), `--workers`, `--limit`. Runs can be resumed: rerunning the same command skips finished problems.

## Tests

`pytest` (38 tests) covers:
- the graph's routing with a scripted fake LLM: reflection, the ablation, bounded memory, max attempts, guardrail rejection, checkpointed memory;
- the guardrails and the Docker sandbox (no network, read-only filesystem);
- the LiveCodeBench test harness, including a malicious pickle being refused;
- the metrics, including the McNemar test.

CI runs ruff, strict mypy and the tests.

## Differences from the paper

- **Model:** Claude Haiku 5.5 instead of GPT-4, and prompts written for this replication.
- **Tests:** generated with a direct prompt and filtered by AST/format, without the paper's chain-of-thought step.
- **Benchmark:** LiveCodeBench replaces the paper's LeetcodeHardGym, which needs LeetCode submissions.
- **Scope:** the "test generation omission" ablation and the paper's reasoning and decision-making tasks (HotpotQA, ALFWorld) are not replicated.
