"""LM factory for self-evolution.

Use Hermes-backed routing when the model string starts with `hermes/`.
Otherwise fall back to DSPy's normal LiteLLM-backed LM.
"""

from __future__ import annotations

import dspy

from evolution.core.hermes_lm import HermesLM


def create_lm(model: str):
    if model.startswith("hermes/"):
        return HermesLM(model=model)
    return dspy.LM(model)
