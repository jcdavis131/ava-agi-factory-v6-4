"""
logic_textbook_pipeline.py — generates 50B logic + 300B math textbooks (Phi Method B)
Solo personal project, no connection to employer, built with public/free-tier only
"""
import argparse, json, random, pathlib
from pathlib import Path

PHI_PROMPTS=[
    "Explain {topic} like a textbook chapter with definitions, theorems, examples.",
    "Create a problem set for {topic} with step-by-step solutions.",
    "Write a Socratic dialogue exploring {topic} misconceptions.",
]

TOPICS_LOGIC=["propositional logic","first-order logic","modal logic","proof by contradiction","induction","pigeonhole"]
TOPICS_MATH=["arithmetic","algebra","geometry","discrete","calculus","linear algebra","probability"]

def gen_textbook(topic, method="Phi B"):
    return f"# {topic}\n\nDefinition: ...\nTheorem: If ... then ...\nProof: ...\nExample: ...\nExercise: ... (Method {method})"

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--logic_tokens", default="50B")
    parser.add_argument("--math_tokens", default="300B")
    parser.add_argument("--out", default="data/synthetic")
    args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"Generating {args.logic_tokens} logic + {args.math_tokens} math textbooks via Phi Method B")
    # mock: create few files instead of 350B
    for i, t in enumerate(TOPICS_LOGIC):
        (out/f"logic_{i}_{t.replace(' ','_')}.md").write_text(gen_textbook(t))
    for i, t in enumerate(TOPICS_MATH):
        (out/f"math_{i}_{t.replace(' ','_')}.md").write_text(gen_textbook(t))
    print(f"Wrote {len(TOPICS_LOGIC)+len(TOPICS_MATH)} mock textbooks to {out} — replace with full Nemotron-70B reward >0.8 pipeline for 350B")
    # Simulate reward filtering
    print("Filtering with Nemotron-70B reward >0.8 — kept ~85% (mock)")

if __name__=="__main__":
    main()
