"""Build results/<run>/REPORT.md and the accuracy-vs-attempts plot from the JSONL results.

python -m reflexion.report --run-name haiku-full
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from reflexion.run import RESULTS

# Reflexion paper, HumanEval (Python), GPT-4 (Table 1 and the test-reliability table)
PAPER = {"base": 0.80, "reflexion": 0.91, "tp": 0.99, "fn": 0.40, "fp": 0.01, "tn": 0.60}
PAPER_LINE = {
    "humaneval": "Paper (GPT-4, HumanEval Python): base 80% → Reflexion 91%.",
    "livecodebench": "The paper did not use LiveCodeBench; its closest setting, LeetCode Hard (Python), "
    "reports GPT-4 7.5% → Reflexion 15.0%.",
}


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar test on paired outcomes (problems solved by one arm but not the other)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail: float = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(run_dir: Path, strategy: str) -> list[dict[str, Any]]:
    path = run_dir / f"{strategy}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pass_at_attempt(rows: list[dict[str, Any]], k: int) -> float:
    """Accuracy if the agent had been stopped after k attempts (it commits to its latest code)."""
    return sum(bool(r["attempts"][:k][-1]["passed_hidden"]) for r in rows) / len(rows)


def internal_test_reliability(rows: list[dict[str, Any]]) -> dict[str, float]:
    """How trustworthy the self-generated tests are, conditioned on their verdict, as in the paper:
    TP/FP = share of internally-passing implementations that are actually correct/incorrect;
    FN/TN = share of internally-failing implementations that are actually correct/incorrect."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in rows:
        for a in r["attempts"]:
            if a["guardrail_error"]:
                continue
            key = ("t" if a["passed_internal"] == a["passed_hidden"] else "f") + ("p" if a["passed_internal"] else "n")
            counts[key] += 1
    positives, negatives = counts["tp"] + counts["fp"], counts["fn"] + counts["tn"]
    return {
        "tp": counts["tp"] / positives if positives else 0.0,
        "fp": counts["fp"] / positives if positives else 0.0,
        "fn": counts["fn"] / negatives if negatives else 0.0,
        "tn": counts["tn"] / negatives if negatives else 0.0,
        "n_pass": positives,
        "n_fail": negatives,
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "n": n,
        "base": pass_at_attempt(rows, 1),
        "final": sum(r["passed"] for r in rows) / n,
        "mean_attempts": sum(len(r["attempts"]) for r in rows) / n,
        "mean_llm_calls": sum(r["llm_calls"] for r in rows) / n,
        "guardrail_blocks": sum(bool(a["guardrail_error"]) for r in rows for a in r["attempts"]),
        "fixed": sum(not r["attempts"][0]["passed_hidden"] and r["passed"] for r in rows),
        "broken": sum(r["attempts"][0]["passed_hidden"] and not r["passed"] for r in rows),
        "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
    }


def plot(run_dir: Path, curves: dict[str, list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, ys in curves.items():
        ax.plot(range(1, len(ys) + 1), [y * 100 for y in ys], marker="o", label=name)
    ax.set_xlabel("attempts allowed")
    ax.set_ylabel("HumanEval pass@1 (%)")
    ax.set_xticks(range(1, max(len(v) for v in curves.values()) + 1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "accuracy_vs_attempts.png", dpi=150)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args(argv)
    run_dir = RESULTS / args.run_name
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}

    arms = {s: load(run_dir, s) for s in ("base", "retry", "reflexion")}
    arms = {s: rows for s, rows in arms.items() if rows}
    stats = {s: summarise(rows) for s, rows in arms.items()}
    max_k = max(len(r["attempts"]) for rows in arms.values() for r in rows)
    curves = {
        {"retry": "retry with test feedback (no reflection)", "reflexion": "Reflexion"}.get(s, s): [
            pass_at_attempt(rows, k) for k in range(1, max_k + 1)
        ]
        for s, rows in arms.items()
        if s != "base"
    }
    if curves:
        plot(run_dir, curves)

    lines = [
        f"# Results: {args.run_name}",
        "",
        f"Model: **{config.get('model', '?')}** · problems: {next(iter(stats.values()))['n']} · "
        f"max attempts {config.get('max_attempts', '?')} · memory {config.get('memory_size', '?')} reflections · "
        f"{config.get('n_tests', '?')} self-generated tests",
        "",
        "| Strategy | first attempt (base) | final pass@1 | fixed | broke | mean attempts | LLM calls / problem |",
        "|---|---|---|---|---|---|---|",
    ]
    for s, st in stats.items():
        lines.append(
            f"| {s} | {st['base']:.1%} | **{st['final']:.1%}** | {st['fixed']} | {st['broken']} | "
            f"{st['mean_attempts']:.2f} | {st['mean_llm_calls']:.2f} |"
        )
    lines += [
        "",
        PAPER_LINE.get(config.get("benchmark", "humaneval"), ""),
        "",
        "*fixed* = wrong at the first attempt, correct at the end; *broke* = the reverse (a correct first "
        "attempt abandoned because a self-generated test was wrong).",
    ]
    if "reflexion" in arms and "retry" in arms:
        refl = {r["task_id"]: r["passed"] for r in arms["reflexion"]}
        retry = {r["task_id"]: r["passed"] for r in arms["retry"]}
        common = refl.keys() & retry.keys()
        only_refl = sum(refl[t] and not retry[t] for t in common)
        only_retry = sum(retry[t] and not refl[t] for t in common)
        lines += [
            "",
            "## Does self-reflection help? (paired comparison)",
            "",
            f"Solved only with Reflexion: **{only_refl}** · only with plain retry: **{only_retry}** · "
            f"exact McNemar p = {mcnemar_p(only_refl, only_retry):.3f} (n = {len(common)})",
        ]
    by_platform = {s: rows for s, rows in arms.items() if rows and "platform" in rows[0]}
    if by_platform:
        lines += [
            "",
            "## By platform",
            "",
            "| Strategy | platform | n | first attempt | final pass@1 |",
            "|---|---|---|---|---|",
        ]
        for s, rows in by_platform.items():
            for platform in sorted({r["platform"] for r in rows}):
                sub = [r for r in rows if r["platform"] == platform]
                st = summarise(sub)
                lines.append(f"| {s} | {platform} | {st['n']} | {st['base']:.1%} | {st['final']:.1%} |")
    for s in ("reflexion", "retry"):
        if s in arms:
            rel = internal_test_reliability(arms[s])
            lines += [
                "",
                f"## Reliability of self-generated tests ({s} run)",
                "",
                "| verdict | correct | incorrect | n |",
                "|---|---|---|---|",
                f"| internal tests pass | TP {rel['tp']:.0%} (paper {PAPER['tp']:.0%}) | FP {rel['fp']:.0%} "
                f"(paper {PAPER['fp']:.0%}) | {rel['n_pass']} |",
                f"| internal tests fail | FN {rel['fn']:.0%} (paper {PAPER['fn']:.0%}) | TN {rel['tn']:.0%} "
                f"(paper {PAPER['tn']:.0%}) | {rel['n_fail']} |",
            ]
            break
    if curves:
        lines += ["", "![accuracy vs attempts](accuracy_vs_attempts.png)"]
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
