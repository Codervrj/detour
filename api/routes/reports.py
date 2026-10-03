"""GET /report/latest - the newest eval report JSON."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.schemas import ReportResponse
from detour.eval import report as report_reader

router = APIRouter(tags=["reports"])


@router.get("/report/latest", response_model=ReportResponse)
def latest_report() -> ReportResponse:
    """The most recent eval report, or a 404 saying how to produce one."""
    payload = report_reader.latest()
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail="No eval report yet. Run: uv run python run.py eval",
        )
    return ReportResponse(report=payload)
