from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


class LLMError(RuntimeError):
    pass


@dataclass
class LLMClient:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_retries: int = 3
    timeout: float = 120.0
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_available: bool = False
    _client: object | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.model:
            raise LLMError("no model configured; set model.name in the config or the "
                           "environment variable named by model.model_env")
        if not self.api_key:
            raise LLMError("no API key found; export the environment variable named by "
                           "model.api_key_env (see .env.example)")
        try:
            from openai import OpenAI
        except ModuleNotFoundError as error:
            raise LLMError("the openai package is required for model calls") from error
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url or None,
                              timeout=self.timeout)

    def complete(self, system: str, user: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                self.api_calls += 1
                self._accumulate_usage(getattr(response, "usage", None))
                return (response.choices[0].message.content or "").strip()
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
        raise LLMError(f"model call failed after {self.max_retries} attempts: "
                       f"{last_error!r}")

    def _accumulate_usage(self, usage) -> None:
        if usage is None:
            return
        self.usage_available = True
        for name in USAGE_FIELDS:
            value = getattr(usage, name, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(name)
            if value is not None:
                setattr(self, name, getattr(self, name) + int(value))

    def usage_summary(self) -> dict:
        return {"available": self.usage_available, "api_calls": self.api_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}

    @classmethod
    def from_env(cls, model: str, *, api_key_env: str = "MODEL_API_KEY",
                 base_url_env: str = "MODEL_BASE_URL", temperature: float = 0.0,
                 max_retries: int = 3) -> "LLMClient":
        return cls(model=model, api_key=os.environ.get(api_key_env),
                   base_url=os.environ.get(base_url_env), temperature=temperature,
                   max_retries=max_retries)
