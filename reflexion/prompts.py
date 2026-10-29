"""Prompt pieces shared by all benchmarks. Written for this project; they follow the structure of the
paper's prompts (test writer, actor, self-reflection), not their wording."""

import re

REFLECT_SYSTEM = """\
You are an expert Python programmer reviewing your own failed solution.
Explain in a few sentences why it is wrong and what concretely must change. Unit tests can themselves be \
wrong; if a failing test contradicts the problem statement, say so. Do not write the corrected code."""


def implement_prompt(
    problem: str, previous_code: str = "", feedback: str = "", reflections: list[str] | None = None
) -> str:
    parts = [problem]
    if previous_code:
        parts.append(f"Your previous solution:\n```python\n{previous_code}\n```")
    if feedback:
        parts.append(f"Unit test results for it:\n{feedback}")
    if reflections:
        notes = "\n".join(f"{i}. {r.strip()}" for i, r in enumerate(reflections, start=1))
        parts.append(f"Your reflections on earlier attempts (most recent last):\n{notes}")
    if previous_code:
        parts.append("Write an improved solution.")
    return "\n\n".join(parts)


def reflect_prompt(problem: str, code: str, feedback: str) -> str:
    return (
        f"{problem}\n\nYour solution:\n```python\n{code}\n```\n\n"
        f"Unit test results:\n{feedback}\n\nWhy is the solution wrong, and what must change?"
    )


_BLOCK = re.compile(r"```(?:python|py|json)?\s*\n(.*?)```", re.DOTALL)


def code_blocks(text: str) -> list[str]:
    return _BLOCK.findall(text)


def extract_code(text: str) -> str:
    blocks = code_blocks(text)
    return (blocks[0] if blocks else text).strip()


def format_feedback(results: list[tuple[str, bool, str]]) -> str:
    lines = []
    for test, passed, error in results:
        lines.append(f"PASSED: {test}" if passed else f"FAILED: {test}  # {error}")
    return "\n".join(lines)
