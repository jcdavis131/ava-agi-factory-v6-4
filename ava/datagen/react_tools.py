"""ReAct tool-use training corpus (phases 3 & 5) — teaches the plain-text
Thought:/Action:/Observation: convention AgenticOS/ava_bridge.py parses
(see ~/.claude/plans/tender-tinkering-sketch.md Phase 6).

Deliberately weighted toward *grounding* over raw tool syntax: a large
fraction of the corpus is "the tool result says X doesn't exist / isn't
what you assumed — say so" rather than "the tool succeeded, report the
happy path" — teaching the model to actually use the Observation rather
than pattern-match "I called a tool, therefore I did the task" (the exact
failure mode `agent-eval`'s first real scoreboard run surfaced against
qwen2.5:1.5b: it called get_clock, then ignored the result and produced a
generic non-answer). This is the model-side lever for the project's north
star: "figure out what is correct, prioritizing factual correctness."

Families:
  * tool_math (task_type="deliberate"): arithmetic via a tool call, answer
    matches the tool's Observation exactly (not the model's own arithmetic).
  * tool_date (task_type="temporal"): "what's today" answered via get_clock,
    never from parametric memory — mirrors agent-eval's grounded-todays-date
    task.
  * tool_grounding_notfound (task_type="deliberate"): the user asks about
    something that doesn't exist (a function, a file, a fact); the
    Observation says so; the final answer says so too, plainly, instead of
    fabricating a plausible-sounding answer. Mirrors agent-eval's
    hallucination-resistance-missing-function task directly.
  * tool_read_cite (task_type="deliberate"): the user asks what a file says;
    the final answer quotes the Observation rather than paraphrasing from
    assumption. Mirrors agent-eval's cite-readme-purpose task.
"""

from __future__ import annotations

from typing import Iterator

from ava.datagen.base import Generator

USER = "<|user|>"
ASSISTANT = "<|assistant|>"


def dialogue(turns: list[tuple[str, str]]) -> str:
    parts = []
    for role, content in turns:
        marker = USER if role == "user" else ASSISTANT
        parts.append(f"{marker}\n{content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# tool_math — arithmetic grounded in the tool's Observation, not the model's
# own (possibly wrong) mental arithmetic.
# ---------------------------------------------------------------------------

def _tool_math_doc(rng) -> tuple[str, str, str]:
    a, b = rng.randint(2, 97), rng.randint(2, 97)
    op = rng.choice(["+", "-", "*"])
    result = {"+": a + b, "-": a - b, "*": a * b}[op]
    fn = {"+": "add", "-": "subtract", "*": "multiply"}[op]
    text = dialogue([
        ("user", f"What is {a} {op} {b}? Use the calculator tool rather than doing it in your head."),
        ("assistant",
         f"Thought: I should call the calculator tool rather than trust my own arithmetic.\n"
         f"Action: {fn}(a={a}, b={b})"),
        ("user", f"Observation: {result}"),
        ("assistant", f"{a} {op} {b} = {result}."),
    ])
    return text, "deliberate", "tool_math"


# ---------------------------------------------------------------------------
# tool_date — never answer "today" from parametric memory.
# ---------------------------------------------------------------------------

_DATES = [f"2026-{m:02d}-{d:02d}" for m in range(1, 13) for d in (1, 8, 15, 22)]


def _tool_date_doc(rng) -> tuple[str, str, str]:
    date = rng.choice(_DATES)
    text = dialogue([
        ("user", "What is today's date?"),
        ("assistant",
         "Thought: my weights are frozen; I can't know today's date from memory. I need to check.\n"
         "Action: get_clock()"),
        ("user", f"Observation: {date}"),
        ("assistant", f"Today is {date}."),
    ])
    return text, "temporal", "tool_date"


# ---------------------------------------------------------------------------
# tool_grounding_notfound — the core anti-hallucination family. The user
# asks about something that doesn't exist; the model must say so plainly
# after checking, not fabricate a plausible answer.
# ---------------------------------------------------------------------------

_FAKE_FUNCTIONS = [
    "divide_safely", "normalize_path", "auto_retry", "parse_config_v2",
    "cached_lookup", "safe_delete", "batch_convert", "validate_schema_strict",
]
_FILES = ["utils.py", "helpers.py", "core.py", "config.py", "models.py"]


def _tool_grounding_notfound_doc(rng) -> tuple[str, str, str]:
    fn = rng.choice(_FAKE_FUNCTIONS)
    fname = rng.choice(_FILES)
    text = dialogue([
        ("user", f"What does the {fn} function in {fname} do?"),
        ("assistant",
         f"Thought: I should check the file before answering rather than guess what a function "
         f"with that name might do.\n"
         f"Action: repo_grep(pattern=\"{fn}\", path=\"{fname}\")"),
        ("user", "Observation: (no matches)"),
        ("assistant",
         f"I checked {fname} and there's no function called {fn} in it — I can't tell you what it "
         f"does because it doesn't exist. I'm not going to guess at plausible-sounding behavior for "
         f"a function that isn't there."),
    ])
    return text, "deliberate", "tool_grounding_notfound"


# ---------------------------------------------------------------------------
# tool_read_cite — quote the Observation rather than paraphrase from
# assumption.
# ---------------------------------------------------------------------------

_PURPOSES = [
    "a throwaway scratch repo for shaking out tool-surface tests",
    "a small internal utility library with no external dependencies",
    "a reference implementation used for onboarding new engineers",
    "an experimental branch that is not part of the stable release",
]


def _tool_read_cite_doc(rng) -> tuple[str, str, str]:
    purpose = rng.choice(_PURPOSES)
    text = dialogue([
        ("user", "What is this repo for, according to its README? Quote it, don't paraphrase."),
        ("assistant",
         "Thought: I should read the actual file rather than assume what a typical README says.\n"
         "Action: repo_read_file(path=\"README.md\")"),
        ("user", f"Observation: \"This repo is {purpose}.\""),
        ("assistant", f"According to the README: \"This repo is {purpose}.\""),
    ])
    return text, "deliberate", "tool_read_cite"


class ReactToolsGenerator(Generator):
    name = "react_tools"
    phases = (3, 5)

    # (weight, builder, source, phase) — grounding families (notfound + cite)
    # outweigh the happy-path family (math) 2:1, matching the north star's
    # priority on "checks before asserting" over raw tool-syntax fluency.
    _FAMILIES = [
        (0.20, _tool_math_doc, "react/math", 3),
        (0.20, _tool_date_doc, "react/date", 3),
        (0.35, _tool_grounding_notfound_doc, "react/grounding", 5),
        (0.25, _tool_read_cite_doc, "react/cite", 5),
    ]

    def generate(self, target_bytes: int) -> Iterator[dict]:
        cum_weights = []
        total = 0.0
        for w, _, _, _ in self._FAMILIES:
            total += w
            cum_weights.append(total)

        produced = 0
        while produced < target_bytes:
            r = self.rng.random() * total
            idx = 0
            while r > cum_weights[idx]:
                idx += 1
            _, builder, source, phase = self._FAMILIES[idx]
            text, task_type, concept = builder(self.rng)
            d = self.doc(text=text, task_type=task_type, concept=concept, phase=phase, source=source)
            produced += len(d["text"].encode("utf-8"))
            yield d


if __name__ == "__main__":
    from ava.datagen.base import run_cli

    run_cli(ReactToolsGenerator)
