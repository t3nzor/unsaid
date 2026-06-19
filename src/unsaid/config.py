"""Application configuration helpers."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "unsaid" / "config.toml"


def _clean_token(token: str | None) -> str | None:
    if token is None:
        return None
    token = token.strip()
    return token or None


def _read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid config TOML at {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"could not read config at {path}: {exc}") from exc
    return data


def load_hf_token(config_path: str | Path | None = None) -> str | None:
    """Load a Hugging Face token from ``config_path``.

    The file is optional. When present, the expected format is:

    ``[huggingface]`` with ``token = "hf_..."``.
    """
    path = Path(config_path).expanduser() if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return None

    data = _read_config(path)
    huggingface = data.get("huggingface")
    if huggingface is None:
        return None
    if not isinstance(huggingface, dict):
        raise ValueError(f"invalid config at {path}: [huggingface] must be a table")

    token = huggingface.get("token")
    if token is None:
        return None
    if not isinstance(token, str):
        raise ValueError(f"invalid config at {path}: huggingface.token must be a string")
    return _clean_token(token)


def load_preamble(config_path: str | Path | None = None) -> str | None:
    """Load the preamble (initial prompt) from ``config_path``.

    The file is optional. When present, the expected format is:

    ``[unsaid]`` with ``preamble = "..."``.
    """
    path = Path(config_path).expanduser() if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        return None

    data = _read_config(path)
    unsaid = data.get("unsaid")
    if unsaid is None:
        return None
    if not isinstance(unsaid, dict):
        raise ValueError(f"invalid config at {path}: [unsaid] must be a table")

    value = unsaid.get("preamble")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"invalid config at {path}: unsaid.preamble must be a string")
    return value


def resolve_hf_token(config_path: str | Path | None = None) -> str | None:
    """Resolve a Hugging Face token from env vars, then config.

    Environment variables win so one-off shell overrides never require editing
    the persisted config file.
    """
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = _clean_token(os.environ.get(name))
        if token is not None:
            return token
    return load_hf_token(config_path)
