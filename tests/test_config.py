"""Tests for config and Hugging Face token resolution."""

import pytest

from unsaid.config import load_hf_token, resolve_hf_token


def _clear_hf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)


def test_missing_config_returns_none(monkeypatch, tmp_path):
    _clear_hf_env(monkeypatch)

    assert resolve_hf_token(tmp_path / "missing.toml") is None


def test_load_hf_token_from_config(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[huggingface]\ntoken = " hf_config "\n')

    assert load_hf_token(config) == "hf_config"


def test_hf_token_env_precedes_config(monkeypatch, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[huggingface]\ntoken = "hf_config"\n')
    monkeypatch.setenv("HF_TOKEN", " hf_env ")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")

    assert resolve_hf_token(config) == "hf_env"


def test_hugging_face_hub_token_env_precedes_config(monkeypatch, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[huggingface]\ntoken = "hf_config"\n')
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", " hf_legacy ")

    assert resolve_hf_token(config) == "hf_legacy"


def test_blank_env_falls_back_to_config(monkeypatch, tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[huggingface]\ntoken = "hf_config"\n')
    monkeypatch.setenv("HF_TOKEN", "  ")
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    assert resolve_hf_token(config) == "hf_config"


def test_missing_huggingface_section_returns_none(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[other]\ntoken = "hf_config"\n')

    assert load_hf_token(config) is None


def test_invalid_huggingface_section_raises(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('huggingface = "not a table"\n')

    with pytest.raises(ValueError, match="must be a table"):
        load_hf_token(config)


def test_invalid_token_type_raises(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[huggingface]\ntoken = 123\n')

    with pytest.raises(ValueError, match="must be a string"):
        load_hf_token(config)


def test_invalid_toml_raises(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text('[huggingface\n')

    with pytest.raises(ValueError, match="invalid config TOML"):
        load_hf_token(config)
