"""Report writing and selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from detour.eval import report as report_module


@pytest.fixture
def reports_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(report_module, "REPORTS_DIR", tmp_path)
    return tmp_path


def write_report(directory: Path, name: str, generated_at: str, run_id: str) -> None:
    payload = {"run_id": run_id, "generated_at": generated_at, "schema_version": 1}
    (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def test_latest_uses_the_timestamp_not_the_filename(reports_dir: Path) -> None:
    # The run id is random hex, so "9..." sorts after "6..." by name while being older.
    write_report(reports_dir, "2026-10-03_90917eb9.json", "2026-10-03T04:28:27+00:00", "90917eb9")
    write_report(reports_dir, "2026-10-03_6571d559.json", "2026-10-03T04:33:08+00:00", "6571d559")

    latest = report_module.latest()
    assert latest is not None
    assert latest["run_id"] == "6571d559"


def test_latest_is_none_when_no_reports_exist(reports_dir: Path) -> None:
    assert report_module.latest() is None


def test_latest_skips_unreadable_reports(reports_dir: Path) -> None:
    (reports_dir / "broken.json").write_text("{not json", encoding="utf-8")
    write_report(reports_dir, "good.json", "2026-10-03T04:00:00+00:00", "good0001")

    latest = report_module.latest()
    assert latest is not None
    assert latest["run_id"] == "good0001"


def test_config_hash_is_stable_and_order_independent() -> None:
    a = report_module.config_hash({"name": "smoke", "als": {"factors": 16}})
    b = report_module.config_hash({"als": {"factors": 16}, "name": "smoke"})
    assert a == b
    assert a != report_module.config_hash({"name": "main", "als": {"factors": 16}})
