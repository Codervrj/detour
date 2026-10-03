"""The map, loaded once at startup and shared by the routes.

Nothing is trained at request time: the app reads what the pipeline already wrote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from fastapi import HTTPException

from detour.config import load_config
from detour.mapservice import MapService

CONFIG_PATH = "evals/configs/main.yaml"
DEMO_LISTENS = "data/processed/val.parquet"
PERSONAL_LISTENS = "data/personal/listens.parquet"

_service: MapService | None = None
_config: dict[str, Any] | None = None


def load(config_path: str = CONFIG_PATH) -> None:
    """Read the config and the map into memory."""
    global _service, _config
    _config = load_config(config_path)
    _service = MapService()


def config() -> dict[str, Any]:
    if _config is None:
        raise HTTPException(status_code=503, detail="The map is not loaded yet.")
    return _config


def service() -> MapService:
    """The loaded map, or a 503 saying exactly how to build it."""
    if _service is None:
        raise HTTPException(
            status_code=503,
            detail="No map loaded. Build it with: uv run python run.py pipeline",
        )
    return _service


def personal_listens() -> pl.DataFrame | None:
    """Your own imported history, if you have run the Last.fm import."""
    path = Path(PERSONAL_LISTENS)
    return pl.read_parquet(path) if path.exists() else None


def demo_listens(listener: str) -> pl.DataFrame | None:
    """One real listener from the held-out split, for when no personal history exists."""
    path = Path(DEMO_LISTENS)
    if not path.exists():
        return None
    frame = pl.read_parquet(path).filter(pl.col("user_id") == listener)
    if frame.height == 0:
        return None
    return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias("artist_name"))


def demo_listeners(limit: int = 12) -> list[str]:
    """A handful of held-out listeners to choose from."""
    path = Path(DEMO_LISTENS)
    if not path.exists():
        return []
    counted = (
        pl.read_parquet(path, columns=["user_id"])
        .group_by("user_id")
        .agg(pl.len().alias("plays"))
        .sort("plays", descending=True)
        .head(limit)
    )
    return [str(u) for u in counted["user_id"].to_list()]
