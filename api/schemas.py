"""Pydantic v2 response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ArtistPoint(BaseModel):
    """One artist as drawn on the map."""

    artist_mbid: str
    name: str
    plays: int
    x: float
    y: float


class MapResponse(BaseModel):
    """The artist map itself, for the background layer."""

    artists: list[ArtistPoint]
    total_artists: int
    showing: int


class Coverage(BaseModel):
    """How much of a listener's history the map could actually read."""

    matched_artists: int
    unmatched_artists: int
    artist_coverage: float = Field(ge=0.0, le=1.0)
    play_coverage: float = Field(ge=0.0, le=1.0)
    biggest_misses: list[str]


class IslandOut(BaseModel):
    """One cluster of a listener's taste."""

    label: str
    size: int
    plays: int
    isolation: float
    artists: list[ArtistPoint]


class NeighbourOut(BaseModel):
    """An artist near something, with how near."""

    artist_mbid: str
    name: str
    similarity: float
    x: float
    y: float


class ListenerMap(BaseModel):
    """Where one listener sits, and what that says about them."""

    listener: str
    coverage: Coverage
    total_plays: int
    artists: list[ArtistPoint]
    islands: list[IslandOut]
    edges: list[ArtistPoint]
    frontier: list[NeighbourOut]


class ArtistNeighbours(BaseModel):
    """One artist and who sits next to it."""

    artist_mbid: str
    name: str
    neighbours: list[NeighbourOut]


class ReportResponse(BaseModel):
    """The latest eval report, passed through for the results page."""

    report: dict[str, Any]


class HealthResponse(BaseModel):
    """Whether the map loaded, and how big it is."""

    status: str
    artists: int
    has_report: bool
    has_coordinates: bool
