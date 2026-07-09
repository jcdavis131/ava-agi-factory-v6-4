"""Tests for the synthetic data generators (spec 02).

The properties that matter are: byte-level determinism, schema conformance,
and -- most importantly -- that the *content* is correct by construction.
So these tests do not merely check that docs are well-formed; they
independently re-verify the logic proofs, recompute the math answers,
re-exec the code snippets, and confirm the canonical eval facts are present
and never contradicted.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import re
from fractions import Fraction

import pytest

from ava.datagen.base import (
    DOC_KEYS,
    VALID_PHASES,
    VALID_TASK_TYPES,
    make_doc_id,
    validate_doc,
)
from ava.datagen.logic import LogicGenerator
from ava.datagen.math_gen import MathGenerator
from ava.datagen.encyclopedia import EncyclopediaGenerator
from ava.datagen.code_gen import CodeGenGenerator, SAFE_BUILTINS, run_sandboxed
from ava.datagen.chat_safety import ChatSafetyGenerator, _SCENARIO_TEMPLATES

ALL_GENERATORS = [
    LogicGenerator,
    MathGenerator,
    EncyclopediaGenerator,
    CodeGenGenerator,
    ChatSafetyGenerator,
]

# A small byte target keeps tests fast while still exercising every family.
SMALL_TARGET = 400_000


def _collect(gen_cls, seed=1234, target=SMALL_TARGET):
    gen = gen_cls(seed=seed)
    return list(gen.generate(target))


def _serialize(docs) -> bytes:
    return "".join(
        json.dumps(d, sort_keys=True, ensure_ascii=False) + "\n" for d in docs
    ).encode("utf-8")


def _sha(docs) -> str:
    return hashlib.sha256(_serialize(docs)).hexdigest()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gen_cls", ALL_GENERATORS, ids=lambda c: c.name)
def test_determinism_same_seed(gen_cls):
    a = _collect(gen_cls, seed=1234)
    b = _collect(gen_cls, seed=1234)
    assert _sha(a) == _sha(b), f"{gen_cls.name}: same seed produced different output"
    assert len(a) > 0


@pytest.mark.parametrize("gen_cls", ALL_GENERATORS, ids=lambda c: c.name)
def test_determinism_different_seed(gen_cls):
    a = _collect(gen_cls, seed=1234)
    c = _collect(gen_cls, seed=4321)
    assert _sha(a) != _sha(c), f"{gen_cls.name}: different seeds produced identical output"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gen_cls", ALL_GENERATORS, ids=lambda c: c.name)
def test_schema(gen_cls):
    docs = _collect(gen_cls)
    declared = set(gen_cls.phases)
    for d in docs:
        assert set(d.keys()) == DOC_KEYS
        for k in DOC_KEYS:
            assert isinstance(d[k], str) and d[k], f"{gen_cls.name}: empty/non-str key {k}"
        assert d["task_type"] in VALID_TASK_TYPES
        assert d["phase"] in VALID_PHASES
        assert d["concept"].strip(), "concept must be non-empty"
        # phase matches the generator's declared phases
        assert int(d["phase"][1:]) in declared, (
            f"{gen_cls.name}: emitted phase {d['phase']} not in declared {sorted(declared)}"
        )
        # doc_id is the documented function of source+text
        assert d["doc_id"] == make_doc_id(d["source"], d["text"])
        # validate_doc agrees
        validate_doc(d, allowed_phases=declared)


def test_task_types_are_accurate_per_generator():
    """task_type drives J-space losses, so spot-check that each generator's
    dominant/required task_types actually appear."""
    logic_tt = {d["task_type"] for d in _collect(LogicGenerator)}
    assert "deliberate" in logic_tt

    ency_tt = {d["task_type"] for d in _collect(EncyclopediaGenerator)}
    assert ency_tt == {"automatic"}, f"encyclopedia must be all automatic, got {ency_tt}"

    code_tt = {d["task_type"] for d in _collect(CodeGenGenerator)}
    assert code_tt == {"deliberate"}, f"code_gen must be all deliberate, got {code_tt}"

    math_tt = {d["task_type"] for d in _collect(MathGenerator)}
    assert "temporal" in math_tt and "deliberate" in math_tt

    chat_tt = {d["task_type"] for d in _collect(ChatSafetyGenerator)}
    assert {"safety", "automatic", "temporal", "deliberate"} <= chat_tt


# ---------------------------------------------------------------------------
# logic.py: independently verify proofs are valid by construction
# ---------------------------------------------------------------------------

def _eval_prop(formula, assign):
    """Independent re-implementation of propositional evaluation for the test
    (deliberately NOT importing the generator's eval so a bug there can't hide)."""
    tag = formula[0]
    if tag == "ATOM":
        return assign[formula[1]]
    if tag == "NOT":
        return not _eval_prop(formula[1], assign)
    if tag == "AND":
        return _eval_prop(formula[1], assign) and _eval_prop(formula[2], assign)
    if tag == "OR":
        return _eval_prop(formula[1], assign) or _eval_prop(formula[2], assign)
    if tag == "IMPLIES":
        return (not _eval_prop(formula[1], assign)) or _eval_prop(formula[2], assign)
    if tag == "IFF":
        return _eval_prop(formula[1], assign) == _eval_prop(formula[2], assign)
    raise AssertionError(f"bad tag {tag}")


def test_natded_proofs_are_semantically_valid():
    """For sampled natural-deduction docs, parse the premises and the final
    conclusion out of the rendered text and verify -- by exhaustive truth
    assignment over the atoms -- that the conclusion is a logical consequence
    of the premises. If the generator ever emitted an invalid derivation,
    this catches it."""
    from ava.datagen import logic as L

    checked = 0
    gen = LogicGenerator(seed=2024)
    for d in gen.generate(1_500_000):
        if d["source"] != "logic/natded":
            continue
        text = d["text"]
        # collect premises and the derived (non-premise, non-assumption) lines
        premises = []
        derived = []
        assumptions = []
        for line in text.splitlines():
            m = re.match(r"\s*\d+\.\s*(\|\s*)?(Assume:\s*)?(.+?)\s+\[(.+?)\]\s*$", line)
            if not m:
                continue
            is_sub = bool(m.group(1))
            is_assume = bool(m.group(2))
            formula_str = m.group(3).strip()
            rule = m.group(4)
            formula = _parse_formula(formula_str)
            if rule == "Premise":
                premises.append(formula)
            elif is_assume:
                assumptions.append(formula)
            else:
                derived.append((formula, is_sub, rule))

        assert premises, f"no premises parsed from:\n{text}"
        atoms = sorted(_atoms_of_all([f for f in premises] + [f for f, _, _ in derived] + assumptions))

        # The top-level conclusion is the last NON-subproof derived line
        # (subproof lines are only valid under their assumption).
        toplevel = [f for f, is_sub, rule in derived if not is_sub]
        if not toplevel:
            # pure-assumption edge case: nothing to check at top level
            checked += 1
            continue
        conclusion = toplevel[-1]

        for combo in itertools.product([True, False], repeat=len(atoms)):
            assign = dict(zip(atoms, combo))
            if all(_eval_prop(p, assign) for p in premises):
                assert _eval_prop(conclusion, assign), (
                    f"INVALID derivation: premises hold but conclusion fails under {assign}\n{text}"
                )
        checked += 1
        if checked >= 60:
            break
    assert checked >= 30, f"only checked {checked} natded proofs"


def _atoms_of_all(formulas):
    out = set()
    for f in formulas:
        _atoms_of(f, out)
    return out


def _atoms_of(f, out):
    if f[0] == "ATOM":
        out.add(f[1])
    elif f[0] == "NOT":
        _atoms_of(f[1], out)
    else:
        _atoms_of(f[1], out)
        _atoms_of(f[2], out)


_SYM2TAG = {"∧": "AND", "∨": "OR", "→": "IMPLIES", "↔": "IFF"}


def _parse_formula(s):
    """Recursive-descent parser matching the generator's render() output.
    Grammar: iff < implies < or < and < not < atom/paren."""
    s = s.strip()
    tokens = _tokenize(s)
    parser = _Parser(tokens)
    result = parser.parse_iff()
    assert parser.pos == len(tokens), f"trailing tokens parsing {s!r}: {tokens[parser.pos:]}"
    return result


def _tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
        elif ch in "()":
            tokens.append(ch)
            i += 1
        elif ch == "¬":
            tokens.append("¬")
            i += 1
        elif ch in _SYM2TAG:
            tokens.append(ch)
            i += 1
        else:
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            assert j > i, f"cannot tokenize at {i}: {s!r}"
            tokens.append(s[i:j])
            i = j
    return tokens


class _Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _binop(self, sym, sub):
        left = sub()
        while self.peek() == sym:
            self.pos += 1
            right = sub()
            left = (_SYM2TAG[sym], left, right)
        return left

    def parse_iff(self):
        return self._binop("↔", self.parse_implies)

    def parse_implies(self):
        # right-associative implication
        left = self.parse_or()
        if self.peek() == "→":
            self.pos += 1
            right = self.parse_implies()
            return ("IMPLIES", left, right)
        return left

    def parse_or(self):
        return self._binop("∨", self.parse_and)

    def parse_and(self):
        return self._binop("∧", self.parse_not)

    def parse_not(self):
        if self.peek() == "¬":
            self.pos += 1
            return ("NOT", self.parse_not())
        return self.parse_atom()

    def parse_atom(self):
        tok = self.peek()
        if tok == "(":
            self.pos += 1
            inner = self.parse_iff()
            assert self.peek() == ")", "expected )"
            self.pos += 1
            return inner
        assert tok is not None and re.match(r"^[A-Za-z]\w*$", tok), f"expected atom, got {tok!r}"
        self.pos += 1
        return ("ATOM", tok)


def test_logic_truth_table_verdicts():
    """Recompute each truth-table verdict independently and compare."""
    gen = LogicGenerator(seed=7)
    checked = 0
    for d in gen.generate(600_000):
        if d["source"] != "logic/truth_table":
            continue
        text = d["text"]
        m = re.search(r"formula: (.+)", text)
        formula = _parse_formula(m.group(1))
        atoms = sorted(_atoms_of_all([formula]))
        results = [_eval_prop(formula, dict(zip(atoms, combo)))
                   for combo in itertools.product([True, False], repeat=len(atoms))]
        if all(results):
            expected = "TAUTOLOGY"
        elif not any(results):
            expected = "CONTRADICTION"
        else:
            expected = "CONTINGENT"
        assert expected in text, f"verdict mismatch (expected {expected}):\n{text}"
        checked += 1
        if checked >= 40:
            break
    assert checked >= 20


# ---------------------------------------------------------------------------
# math_gen.py: recompute stated answers
# ---------------------------------------------------------------------------

def _parse_num(s):
    s = s.strip()
    if "/" in s:
        return Fraction(s)
    return Fraction(int(s))


def test_math_answers_are_correct():
    """Parse the final answer out of >=200 sampled math docs and confirm it
    equals a value recomputed by the test."""
    gen = MathGenerator(seed=1234)
    verified = 0
    for d in gen.generate(2_000_000):
        text = d["text"]
        concept = d["concept"]
        if concept == "addition":
            a, b = map(int, re.search(r"Compute (\d+) \+ (\d+)", text).groups())
            ans = int(re.search(r"Final answer: (-?\d+)", text).group(1))
            assert ans == a + b
            verified += 1
        elif concept == "subtraction":
            a, b = map(int, re.search(r"Compute (\d+) - (\d+)", text).groups())
            ans = int(re.search(r"Final answer: (-?\d+)", text).group(1))
            assert ans == a - b
            verified += 1
        elif concept == "multiplication":
            a, b = map(int, re.search(r"Compute (\d+) x (\d+)", text).groups())
            ans = int(re.search(r"Final answer: (-?\d+)", text).group(1))
            assert ans == a * b
            verified += 1
        elif concept == "linear_equation":
            a, b, c = map(int, re.search(r"Solve for x: (-?\d+)x \+ (-?\d+) = (-?\d+)", text).groups())
            ans = _parse_num(re.search(r"Final answer: x = (-?\d+(?:/\d+)?)", text).group(1))
            assert a * ans + b == c
            verified += 1
        elif concept == "modular_arithmetic" and "Compute" in text:
            a, m = map(int, re.search(r"Compute (\d+) mod (\d+)", text).groups())
            ans = int(re.search(r"Final answer: \d+ mod \d+ = (\d+)", text).group(1))
            assert ans == a % m
            verified += 1
        elif concept == "unit_conversion":
            m = re.search(r"Convert (\d+) (\w+) to (\w+), then add (\d+)", text)
            amount, _, _, extra = int(m.group(1)), m.group(2), m.group(3), int(m.group(4))
            factor = int(re.search(r"1 \w+ = (\d+) \w+", text).group(1))
            ans = int(re.search(r"Final answer: (\d+)", text).group(1))
            assert ans == amount * factor + extra
            verified += 1
        if verified >= 250:
            break
    assert verified >= 200, f"only verified {verified} math answers"


def test_math_temporal_docs_present():
    gen = MathGenerator(seed=1234)
    saw_temporal = False
    for d in itertools.islice(gen.generate(2_000_000), 20000):
        if d["task_type"] == "temporal":
            assert d["concept"] in {"deadline", "schedule", "delay"}
            assert "deadline" in d["text"].lower()
            saw_temporal = True
            break
    assert saw_temporal


# ---------------------------------------------------------------------------
# code_gen.py: re-exec snippets, and verify sandbox rejects dangerous ops
# ---------------------------------------------------------------------------

def _exec_and_check_doctest(code_text):
    """Extract the function/class body (drop the docstring) and the doctest
    lines, re-exec in the same sandbox the generator used, and confirm each
    >>> line's output matches the recorded expected output."""
    lines = code_text.splitlines()
    # split docstring examples from code
    steps = []  # (src, expected or None)
    in_doc = False
    doc_depth = None
    code_lines = []
    i = 0
    # The docstring is delimited by the first triple-quote block.
    # Rebuild code without the docstring, and collect >>> examples.
    doc_started = False
    doc_ended = False
    pending_expected_for = None
    for line in lines:
        stripped = line.strip()
        if not doc_started and stripped.startswith('"""'):
            doc_started = True
            # single-line docstring?
            if stripped.count('"""') == 2 and len(stripped) > 3:
                doc_ended = True
            continue
        if doc_started and not doc_ended:
            if stripped.startswith('"""'):
                doc_ended = True
                continue
            if stripped.startswith(">>> "):
                src = stripped[4:]
                is_expr = "=" not in src.split("(")[0] or src.startswith(("print(",))
                # heuristic: a bare assignment 'x = ...' is a statement
                is_assignment = bool(re.match(r"^[A-Za-z_]\w*\s*=", src)) and not src.startswith("==")
                steps.append([src, None, not is_assignment])
            elif steps and steps[-1][1] is None and steps[-1][2] and not stripped.startswith(">>>") and stripped:
                steps[-1][1] = stripped
            continue
        code_lines.append(line)

    code = "\n".join(code_lines)
    # Re-run through the generator's own sandbox for parity.
    replay = [(s[0], s[2]) for s in steps]
    result = run_sandboxed(code, replay)
    assert result is not None, f"snippet failed to execute in sandbox:\n{code_text}"
    for (src, expected, is_expr), (rsrc, got) in zip(steps, result):
        if is_expr and expected is not None:
            assert got == expected, f"doctest mismatch for {src}: expected {expected!r} got {got!r}"
    return len([s for s in steps if s[2] and s[1] is not None])


def test_code_snippets_execute_and_match_doctests():
    gen = CodeGenGenerator(seed=1234)
    checked = 0
    for d in gen.generate(800_000):
        n = _exec_and_check_doctest(d["text"])
        assert n >= 1, f"no verifiable doctest expressions in:\n{d['text']}"
        checked += 1
        if checked >= 100:
            break
    assert checked >= 50


def test_sandbox_rejects_open():
    assert run_sandboxed("def f():\n    return open('x')\n", [("f()", True)]) is None


def test_sandbox_rejects_import():
    assert run_sandboxed("import os\ndef f():\n    return 1\n", [("f()", True)]) is None
    assert run_sandboxed("def f():\n    return __import__('os')\n", [("f()", True)]) is None


def test_sandbox_rejects_eval_exec_compile():
    assert run_sandboxed("def f():\n    return eval('1+1')\n", [("f()", True)]) is None
    assert run_sandboxed("def f():\n    return exec('x=1')\n", [("f()", True)]) is None
    assert run_sandboxed("def f():\n    return compile('1', '<s>', 'eval')\n", [("f()", True)]) is None


def test_sandbox_whitelist_excludes_dangerous_builtins():
    for forbidden in ("open", "__import__", "eval", "exec", "compile", "input", "globals"):
        assert forbidden not in SAFE_BUILTINS, f"{forbidden} must not be in the sandbox whitelist"


# ---------------------------------------------------------------------------
# encyclopedia.py: canonical entities, paraphrase coverage, no contradictions
# ---------------------------------------------------------------------------

CANONICAL_FACTS = {
    "spider": ("8 legs", ["6 legs", "4 legs", "2 legs"]),
    "ant": ("6 legs", ["8 legs", "4 legs", "2 legs"]),
}


def test_encyclopedia_canonical_leg_counts_consistent():
    """spider->8, ant->6, always. A single contradictory doc would poison the
    intervention evals, so we assert NO doc about spider ever says anything but
    8 legs (and likewise ant->6)."""
    gen = EncyclopediaGenerator(seed=1234)
    saw = {"spider": 0, "ant": 0}
    for d in gen.generate(3_000_000):
        text = d["text"]
        # any doc mentioning the entity's legs must state the right count
        for entity, (correct, wrong_list) in CANONICAL_FACTS.items():
            # only inspect leg statements that name this entity
            if re.search(rf"\b{entity}\b", text, re.IGNORECASE) and "legs" in text:
                # find "<entity> ... N legs" style claims
                for wrong in wrong_list:
                    # a wrong leg-count sentence naming this entity is a contradiction
                    pattern = rf"(a |an |the |every |{entity}[^.]*?){wrong}"
                    for mobj in re.finditer(rf"[^.]*\b{entity}\b[^.]*legs[^.]*", text, re.IGNORECASE):
                        seg = mobj.group(0)
                        # skip segments that are about a different animal in a compendium
                        if re.search(rf"\b{entity}\b", seg, re.IGNORECASE):
                            assert wrong not in seg or correct in seg, (
                                f"contradiction: '{entity}' segment mentions {wrong}:\n{seg}"
                            )
                if d["concept"] == entity:
                    saw[entity] += 1
        if saw["spider"] > 5 and saw["ant"] > 5:
            break
    assert saw["spider"] > 0 and saw["ant"] > 0


def test_encyclopedia_paraphrase_coverage():
    """Each canonical entity must appear with >=40 DISTINCT paraphrase
    sentences across the corpus."""
    gen = EncyclopediaGenerator(seed=1234)
    entity_sentences = {e: set() for e in ["spider", "ant", "france", "china", "soccer", "rugby", "spanish", "french"]}
    for d in gen.generate(6_000_000):
        concept = d["concept"]
        if concept in entity_sentences:
            for line in d["text"].splitlines():
                line = line.strip()
                if line and not line.startswith("--") and ":" not in line[:20] or "ES:" in line:
                    entity_sentences[concept].add(line)
        if all(len(v) >= 60 for v in entity_sentences.values()):
            break
    for entity, sents in entity_sentences.items():
        assert len(sents) >= 40, f"{entity}: only {len(sents)} distinct paraphrases (need >=40)"


def test_encyclopedia_canonical_country_facts():
    gen = EncyclopediaGenerator(seed=1234)
    facts_seen = {"Paris": False, "Beijing": False, "Mandarin": False, "Yuan": False, "Euro": False}
    france_concept_doc = False
    for d in gen.generate(3_000_000):
        for token in facts_seen:
            if token in d["text"]:
                facts_seen[token] = True
        if d["concept"] == "france":
            france_concept_doc = True
        if all(facts_seen.values()) and france_concept_doc:
            break
    assert all(facts_seen.values()), f"missing canonical tokens: {[k for k,v in facts_seen.items() if not v]}"
    assert france_concept_doc


def test_encyclopedia_no_country_capital_contradiction():
    """France's capital must always be Paris; China's must always be Beijing."""
    gen = EncyclopediaGenerator(seed=99)
    for d in itertools.islice(gen.generate(3_000_000), 40000):
        t = d["text"]
        if "France" in t and "capital" in t.lower():
            # no other city should be called France's capital
            for bad in ("Beijing", "Berlin", "Madrid", "Rome"):
                assert not re.search(rf"capital of France is {bad}", t)
        if "China" in t and "capital" in t.lower():
            for bad in ("Paris", "Tokyo", "Seoul"):
                assert not re.search(rf"capital of China is {bad}", t)


# ---------------------------------------------------------------------------
# chat_safety.py: safety docs refuse; every safety scenario has a benign twin
# ---------------------------------------------------------------------------

_REFUSAL_MARKERS = ["can't", "cannot", "won't", "will not", "i won't", "i can't"]


def test_chat_safety_docs_contain_refusal():
    gen = ChatSafetyGenerator(seed=1234)
    checked = 0
    for d in gen.generate(1_000_000):
        if d["task_type"] != "safety":
            continue
        assert "<|user|>" in d["text"] and "<|assistant|>" in d["text"]
        assistant_part = d["text"].split("<|assistant|>", 1)[1].lower()
        assert any(m in assistant_part for m in _REFUSAL_MARKERS), (
            f"safety doc has no refusal:\n{d['text']}"
        )
        checked += 1
        if checked >= 80:
            break
    assert checked >= 40


def test_every_safety_template_has_benign_twin():
    """Each safety scenario template must define a matched benign twin with the
    same structural fields, so the Critic AUC has a contrast class."""
    for tmpl in _SCENARIO_TEMPLATES:
        assert tmpl["coercive_user"] and tmpl["benign_user"]
        assert tmpl["refusal"] and tmpl["helpful"]
        assert tmpl["concept"] and tmpl["benign_concept"]
        assert tmpl["concept"] != tmpl["benign_concept"]


def test_chat_safety_and_benign_concepts_disjoint():
    """The coercive and benign families must be distinguishable by concept, so
    the safety label is meaningful."""
    gen = ChatSafetyGenerator(seed=1234)
    safety_concepts = set()
    benign_concepts = set()
    for d in gen.generate(1_500_000):
        if d["source"] == "chat/safety":
            safety_concepts.add(d["concept"])
        elif d["source"] == "chat/benign":
            benign_concepts.add(d["concept"])
        if len(safety_concepts) >= 5 and len(benign_concepts) >= 5:
            break
    assert safety_concepts and benign_concepts
    assert safety_concepts.isdisjoint(benign_concepts)


def test_chat_markers_and_phases():
    gen = ChatSafetyGenerator(seed=1234)
    phases = set()
    for d in itertools.islice(gen.generate(1_000_000), 3000):
        assert "<|user|>" in d["text"] and "<|assistant|>" in d["text"]
        phases.add(d["phase"])
    assert {"p3", "p5"} <= phases


# ---------------------------------------------------------------------------
# Phase coverage across all generators
# ---------------------------------------------------------------------------

def test_phase_coverage_union():
    seen = set()
    for gen_cls in ALL_GENERATORS:
        for d in _collect(gen_cls, target=1_000_000):
            seen.add(d["phase"])
    assert {"p0", "p1", "p2", "p3", "p4", "p5"} <= seen, f"missing phases: {sorted({'p0','p1','p2','p3','p4','p5'} - seen)}"


# ---------------------------------------------------------------------------
# write_shards round-trip determinism (file level)
# ---------------------------------------------------------------------------

def test_write_shards_deterministic(tmp_path):
    from ava.datagen.base import write_shards

    a = write_shards(LogicGenerator(seed=1234), str(tmp_path / "a"), target_mb=0.5)
    b = write_shards(LogicGenerator(seed=1234), str(tmp_path / "b"), target_mb=0.5)
    assert a["sha256"] == b["sha256"]
    assert a["bytes"] >= 0.5 * 1024 * 1024
