"""Artefacts loaded once at startup and shared by the routes.

Nothing is trained at request time: the app reads what the pipeline already wrote.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from detour.config import load_config
from detour.recommend import Recommender

CONFIG_PATH = "evals/configs/smoke.yaml"

_recommender: Recommender | None = None
_config: dict[str, Any] | None = None


def load(config_path: str = CONFIG_PATH) -> None:
    """Read the config and the model artefacts into memory."""
    global _recommender, _config
    _config = load_config(config_path)
    _recommender = Recommender()


def config() -> dict[str, Any]:
    """The loaded config."""
    if _config is None:
        raise HTTPException(status_code=503, detail="Artefacts are not loaded yet.")
    return _config


def recommender() -> Recommender:
    """The loaded recommender, or a 503 explaining how to produce the artefacts."""
    if _recommender is None:
        raise HTTPException(
            status_code=503,
            detail="No model artefacts loaded. Run: uv run python run.py pipeline",
        )
    return _recommender
