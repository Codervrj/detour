"""Pydantic v2 response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ListenerSummary(BaseModel):
    """A sample listener as the picker shows them."""

    user_id: str
    explorer_score: float = Field(ge=0.0, le=1.0)
    personalised_lambda: float = Field(ge=0.0, le=1.0)
    train_listens: int
    top_artists: list[str]


class TrackRecommendation(BaseModel):
    """One row of the re-ranked list, with the numbers the UI puts beside it."""

    item_id: str
    rank: int
    relevance: float
    novelty: float
    is_anchor: bool
    artist_name: str
    track_name: str
    why: str


class RecommendationsResponse(BaseModel):
    """A finished list plus the dial position that produced it."""

    user_id: str
    explorer_score: float
    personalised_lambda: float
    lambda_used: float
    k: int
    anchor_count: int
    items: list[TrackRecommendation]


class ReportResponse(BaseModel):
    """The latest eval report, passed through for the results page."""

    report: dict[str, Any]


class HealthResponse(BaseModel):
    """Whether the artefacts loaded and how much they contain."""

    status: str
    listeners: int
    has_report: bool
