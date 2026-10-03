"""Config loading and the JSON sidecar every pipeline stage writes beside its Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DEDUPE_WINDOW_SECONDS = 30


def load_config(path: str) -> dict[str, Any]:
    """Read a YAML config from evals/configs/."""
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data


def write_sidecar(parquet_path: Path, stats: dict[str, Any]) -> None:
    """Write <stem>.stats.json next to a Parquet file: row counts, date range, filter stats."""
    sidecar = parquet_path.with_suffix(".stats.json")
    sidecar.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")


def read_sidecar(parquet_path: Path) -> dict[str, Any]:
    """Read the sidecar stats for a Parquet file."""
    sidecar = parquet_path.with_suffix(".stats.json")
    data: dict[str, Any] = json.loads(sidecar.read_text(encoding="utf-8"))
    return data
