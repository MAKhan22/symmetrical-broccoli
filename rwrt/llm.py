from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import httpx

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openrouter/free"


@dataclass
class LLMConfig:
    """OpenRouter API settings (typically loaded from environment)."""

    api_key: str
    model: str = OPENROUTER_DEFAULT_MODEL
    base_url: str = OPENROUTER_DEFAULT_BASE_URL
    timeout_s: float = 60.0
    app_title: str = "rwrt"

    @classmethod
    def from_env(cls, *, env_file: str | None = ".env") -> LLMConfig:
        if env_file is not None:
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file, override=True)
            except ImportError:
                pass

        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add it to .env or export it in your shell."
            )

        return cls(
            api_key=api_key,
            model=os.environ.get("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL).strip(),
            base_url=os.environ.get(
                "OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL
            ).strip(),
            timeout_s=float(os.environ.get("OPENROUTER_TIMEOUT_S", "60")),
            app_title=os.environ.get("OPENROUTER_APP_TITLE", "rwrt").strip(),
        )


@dataclass
class OpenRouterClient:
    """Minimal OpenRouter chat-completions client."""

    config: LLMConfig
    _client: httpx.Client | None = field(default=None, repr=False)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.4,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rwrt",
            "X-Title": self.config.app_title,
        }

        client = self._client or httpx.Client(timeout=self.config.timeout_s)
        own_client = self._client is None
        try:
            response = client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            if exc.response.status_code == 401:
                raise RuntimeError(
                    "OpenRouter authentication failed (401). "
                    "Check OPENROUTER_API_KEY in your .env file. "
                    "If the key is correct, a stale OPENROUTER_API_KEY in your shell "
                    "environment may be overriding it; restart the shell or unset it. "
                    f"API response: {detail}"
                ) from exc
            raise RuntimeError(
                f"OpenRouter request failed ({exc.response.status_code}): {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
        finally:
            if own_client:
                client.close()

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OpenRouter response: {data!r}") from exc

    def explain_word(
        self,
        word: str,
        *,
        known_words: set[str] | frozenset[str] | None = None,
    ) -> str:
        known_hint = ""
        if known_words:
            sample = sorted(known_words)[:24]
            known_hint = (
                f"\nThe learner already knows words such as: {', '.join(sample)}."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful Turkish language tutor for intermediate learners. "
                    "Be concise, accurate, and pedagogical."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Explain the Turkish word "{word}" for a language learner.{known_hint}\n\n'
                    "Include these sections with clear headings:\n"
                    "1. **English meaning** — short gloss\n"
                    "2. **Turkish explanation** — 1–2 simple Turkish sentences defining the word\n"
                    "3. **Examples** — 2–3 Turkish example sentences with English translations\n"
                    "4. **Usage note** — register, collocation, or common mistake (one line)\n"
                    "5. **grammar** — sometimes words have several suffixes, explain the suffixes and give relevant exampes in section 3."
                ),
            },
        ]
        return self.complete(messages, temperature=0.3)

    def follow_up(
        self,
        word: str,
        history: list[dict[str, str]],
        question: str,
    ) -> str:
        if not history:
            raise ValueError("follow_up requires non-empty conversation history")

        messages = [
            {
                "role": "system",
                "content": (
                    f'You are a Turkish language tutor. The learner is studying "{word}". '
                    "Answer follow-up questions briefly; use the prior explanation as context."
                ),
            },
            *history,
            {"role": "user", "content": question},
        ]
        return self.complete(messages, temperature=0.5)
