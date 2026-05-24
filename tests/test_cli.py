from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rwrt.cli import build_index_main, chat_main, learn_main, recommend_main


def test_recommend_requires_known_or_profile() -> None:
    assert recommend_main([]) == 1


def test_recommend_empty_profile_errors(tmp_path: Path, tiny_db: Path) -> None:
    profile = tmp_path / "empty.json"
    argv = [
        "--db",
        str(tiny_db),
        "--index-dir",
        str(tmp_path / "faiss"),
        "--min-weight",
        "1",
        "--profile",
        str(profile),
    ]
    assert recommend_main(argv) == 1


@patch("rwrt.cli.EmbeddingIndex")
@patch("rwrt.cli.RecommendationPipeline")
def test_recommend_with_profile_merges_words(
    mock_pipe_cls: MagicMock,
    mock_index_cls: MagicMock,
    tmp_path: Path,
    tiny_db: Path,
) -> None:
    mock_pipe = MagicMock()
    from rwrt.types import Candidate

    mock_pipe.recommend.return_value = [
        Candidate(word="kitap", bi_score=0.5, frequency=10)
    ]
    mock_pipe_cls.return_value = mock_pipe
    mock_index_cls.return_value = MagicMock()

    profile = tmp_path / "learner.json"
    argv = [
        "--db",
        str(tiny_db),
        "--index-dir",
        str(tmp_path / "faiss"),
        "--min-weight",
        "1",
        "--profile",
        str(profile),
        "bir",
        "ev",
    ]
    assert recommend_main(argv) == 0
    assert profile.is_file()
    data = profile.read_text(encoding="utf-8")
    assert "bir" in data and "ev" in data


@patch("rwrt.cli.EmbeddingIndex")
def test_build_index_main(mock_index_cls: MagicMock, tmp_path: Path, tiny_db: Path) -> None:
    mock_index = MagicMock()
    mock_index_cls.return_value = mock_index

    argv = [
        "--db",
        str(tiny_db),
        "--index-dir",
        str(tmp_path / "faiss"),
        "--min-weight",
        "1",
    ]
    assert build_index_main(argv) == 0
    mock_index.build.assert_called_once_with(save_embeddings=True)


def test_build_index_passes_bi_model(tmp_path: Path, tiny_db: Path) -> None:
    with patch("rwrt.cli.EmbeddingIndex") as mock_index_cls:
        mock_index = MagicMock()
        mock_index_cls.return_value = mock_index
        argv = [
            "--db",
            str(tiny_db),
            "--index-dir",
            str(tmp_path / "faiss"),
            "--min-weight",
            "1",
            "--bi-model",
            "custom/bi-model",
        ]
        assert build_index_main(argv) == 0


@patch("rwrt.cli.run_learn_session")
@patch("rwrt.cli.EmbeddingIndex")
def test_learn_main_requires_profile_words(
    mock_index_cls: MagicMock,
    mock_session: MagicMock,
    tmp_path: Path,
    tiny_db: Path,
) -> None:
    mock_index_cls.return_value = MagicMock()
    mock_session.return_value = 0

    profile = tmp_path / "learner.json"
    argv = [
        "--db",
        str(tiny_db),
        "--index-dir",
        str(tmp_path / "faiss"),
        "--min-weight",
        "1",
        "--profile",
        str(profile),
        "bir",
    ]
    assert learn_main(argv) == 0
    mock_session.assert_called_once()


def test_learn_main_empty_profile_errors(tmp_path: Path, tiny_db: Path) -> None:
    profile = tmp_path / "empty.json"
    argv = [
        "--db",
        str(tiny_db),
        "--index-dir",
        str(tmp_path / "faiss"),
        "--min-weight",
        "1",
        "--profile",
        str(profile),
    ]
    assert learn_main(argv) == 1


@patch("rwrt.cli.EmbeddingIndex")
@patch("rwrt.cli.RecommendationPipeline")
def test_recommend_honors_pipeline_flags(
    mock_pipe_cls: MagicMock,
    mock_index_cls: MagicMock,
    tmp_path: Path,
    tiny_db: Path,
) -> None:
    mock_pipe = MagicMock()
    mock_pipe.recommend.return_value = []
    mock_pipe_cls.return_value = mock_pipe
    mock_index_cls.return_value = MagicMock()

    argv = [
        "--db",
        str(tiny_db),
        "--index-dir",
        str(tmp_path / "faiss"),
        "--min-weight",
        "1",
        "--query-strategy",
        "weighted_mean",
        "--max-query-words",
        "32",
        "--frequency-boost",
        "0.5",
        "bir",
    ]
    assert recommend_main(argv) == 0
    cfg = mock_pipe_cls.call_args[0][0]
    assert cfg.query_strategy == "weighted_mean"
    assert cfg.max_query_words == 32
    assert cfg.frequency_boost == 0.5


@patch("rwrt.cli.run_chat_session")
@patch("rwrt.cli.OpenRouterClient")
@patch("rwrt.cli.EmbeddingIndex")
def test_chat_main_loads_profile_and_llm(
    mock_index_cls: MagicMock,
    mock_llm_cls: MagicMock,
    mock_session: MagicMock,
    tmp_path: Path,
    tiny_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    mock_index_cls.return_value = MagicMock()
    mock_session.return_value = 0

    profile = tmp_path / "learner.json"
    argv = [
        "--db",
        str(tiny_db),
        "--index-dir",
        str(tmp_path / "faiss"),
        "--min-weight",
        "1",
        "--profile",
        str(profile),
        "--env-file",
        str(tmp_path / "missing.env"),
        "bir",
    ]
    assert chat_main(argv) == 0
    mock_session.assert_called_once()
