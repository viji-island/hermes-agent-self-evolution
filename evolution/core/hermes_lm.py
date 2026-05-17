"""DSPy LM adapter that routes completions through Hermes' authenticated runtime.

This lets self-evolution runs reuse Hermes' existing gpt-5.4 openai-codex
session without needing a standard API key that LiteLLM can call directly.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Any

from dspy.clients.base_lm import BaseLM
from openai import OpenAI


_HERMES_BOOTSTRAPPED = False
_CodexAuxiliaryClient = None
_resolve_codex_runtime_credentials = None


def _bootstrap_hermes_imports() -> None:
    global _HERMES_BOOTSTRAPPED, _CodexAuxiliaryClient, _resolve_codex_runtime_credentials
    if _HERMES_BOOTSTRAPPED:
        return

    hermes_repo = Path.home() / ".hermes" / "hermes-agent"
    if not hermes_repo.exists():
        raise FileNotFoundError(f"Hermes repo not found at {hermes_repo}")

    site_packages = glob.glob(str(hermes_repo / "venv" / "lib" / "python*" / "site-packages"))
    for p in reversed(site_packages):
        if p not in sys.path:
            sys.path.insert(0, p)
    repo_str = str(hermes_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from agent.auxiliary_client import CodexAuxiliaryClient  # type: ignore
    from hermes_cli.auth import resolve_codex_runtime_credentials  # type: ignore

    _CodexAuxiliaryClient = CodexAuxiliaryClient
    _resolve_codex_runtime_credentials = resolve_codex_runtime_credentials
    _HERMES_BOOTSTRAPPED = True


class HermesLM(BaseLM):
    """Custom DSPy LM backed by Hermes' Codex/OpenAI runtime."""

    def __init__(
        self,
        model: str = "hermes/gpt-5.4",
        *,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        cache: bool = False,
        **kwargs,
    ):
        super().__init__(model=model, model_type="chat", temperature=temperature, max_tokens=max_tokens, cache=cache, **kwargs)
        self.hermes_model = model.split("/", 1)[1] if "/" in model else model
        self._client = None

    def copy(self, **kwargs):
        base_kwargs = dict(self.kwargs)
        for key, value in kwargs.items():
            if key not in {"model", "cache"}:
                if value is None:
                    base_kwargs.pop(key, None)
                else:
                    base_kwargs[key] = value
        return HermesLM(
            model=kwargs.get("model", self.model),
            cache=kwargs.get("cache", self.cache),
            temperature=base_kwargs.pop("temperature", self.kwargs.get("temperature", 0.0)),
            max_tokens=base_kwargs.pop("max_tokens", self.kwargs.get("max_tokens", 4000)),
            **base_kwargs,
        )

    def dump_state(self):
        return {
            "model": self.model,
            "model_type": self.model_type,
            "cache": self.cache,
            **self.kwargs,
        }

    def _get_client(self):
        if self._client is None:
            _bootstrap_hermes_imports()
            creds = _resolve_codex_runtime_credentials()
            real_client = OpenAI(api_key=creds["api_key"], base_url=creds["base_url"])
            self._client = _CodexAuxiliaryClient(real_client, self.hermes_model)
        return self._client

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        client = self._get_client()
        merged_kwargs = {**self.kwargs, **kwargs}
        chat_messages = messages or [{"role": "user", "content": prompt or ""}]
        response = client.chat.completions.create(
            model=self.hermes_model,
            messages=chat_messages,
            **merged_kwargs,
        )
        usage = getattr(response, "usage", None)
        if usage is not None and not isinstance(usage, dict):
            response.usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return response
