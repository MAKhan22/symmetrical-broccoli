import json
from pathlib import Path

import httpx
import pytest

from rwrt.llm import LLMConfig, OpenRouterClient


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(api_key="test-key", model="test/model", base_url="https://example.com/api/v1")


def _mock_response(content: str) -> httpx.Response:
    body = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    return httpx.Response(
        200,
        json=body,
        request=httpx.Request("POST", "https://example.com/api/v1/chat/completions"),
    )


def test_complete_returns_assistant_content(llm_config: LLMConfig) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: _mock_response("Merhaba means hello.")
        )
    )
    llm = OpenRouterClient(llm_config, _client=client)

    text = llm.complete([{"role": "user", "content": "hi"}])

    assert text == "Merhaba means hello."
    client.close()


def test_complete_raises_on_http_error(llm_config: LLMConfig) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text="unauthorized",
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenRouterClient(llm_config, _client=client)

    with pytest.raises(RuntimeError, match="OpenRouter authentication failed"):
        llm.complete([{"role": "user", "content": "hi"}])

    client.close()


def test_explain_word_includes_word_in_request(llm_config: LLMConfig) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return _mock_response("**English meaning**: hello")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenRouterClient(llm_config, _client=client)

    text = llm.explain_word("merhaba", known_words={"evet", "hayır"})

    assert "hello" in text
    payload = json.loads(captured["body"])
    user_msg = payload["messages"][-1]["content"]
    assert "merhaba" in user_msg
    assert "evet" in user_msg
    client.close()


def test_follow_up_appends_question(llm_config: LLMConfig) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.content.decode()
        return _mock_response("It is informal.")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenRouterClient(llm_config, _client=client)

    answer = llm.follow_up(
        "merhaba",
        [{"role": "assistant", "content": "Means hello."}],
        "Is it formal?",
    )

    assert answer == "It is informal."
    payload = json.loads(captured["body"])
    assert payload["messages"][-1]["content"] == "Is it formal?"
    client.close()


def test_llm_config_from_env_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LLMConfig.from_env(env_file=None)


def test_llm_config_from_env_overrides_shell_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY=file-key\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "shell-key")

    cfg = LLMConfig.from_env(env_file=str(env_file))

    assert cfg.api_key == "file-key"
