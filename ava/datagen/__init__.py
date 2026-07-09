"""Synthetic data generators for the Ava nano training curriculum.

Every generator in this package is a fully offline, deterministic producer of
phase-tagged JSONL training text: zero network access, private seeded RNG
only, every numeric/factual answer computed by Python (never templated as
literal text). See specs/02_data_generation.md for the detailed contract.
"""

from ava.datagen.base import Generator, write_shards, run_cli, validate_doc

__all__ = ["Generator", "write_shards", "run_cli", "validate_doc"]
