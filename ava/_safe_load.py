"""Hardened ``torch.load`` wrapper.

Prefers ``weights_only=True`` (the safe unpickler that refuses arbitrary code
execution) and only falls back to ``weights_only=False`` when the checkpoint
carries non-tensor Python objects (e.g. a bundled config/optimizer blob) that
the safe loader rejects. Checkpoints here are trusted local artifacts, so the
fallback keeps every existing load working while closing the door on the
default full-pickle path.
"""

from __future__ import annotations

import torch


def safe_torch_load(*args, **kwargs):
    """Drop-in ``torch.load`` that tries ``weights_only=True`` first."""
    kwargs.pop("weights_only", None)
    try:
        return torch.load(*args, weights_only=True, **kwargs)
    except Exception:
        return torch.load(*args, weights_only=False, **kwargs)
