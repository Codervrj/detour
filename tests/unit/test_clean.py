"""Dedupe and identity rules from CLAUDE.md section 5."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from detour.clean.run import clean, dedupe, with_item_id

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)


def listens(*offsets_seconds: int, recording: str = "rec-1", user: str = "u1") -> pl.DataFrame:
    """A frame of plays of one recording by one user at the given second offsets."""
    return pl.DataFrame(
        {
            "user_id": [user] * len(offsets_seconds),
            "artist_mbid": ["a-1"] * len(offsets_seconds),
            "artist_name": ["Artist One"] * len(offsets_seconds),
            "recording_mbid": [recording] * len(offsets_seconds),
            "track_name": ["Track One"] * len(offsets_seconds),
            "listened_at": [T0 + timedelta(seconds=s) for s in offsets_seconds],
        }
    )


def test_two_plays_ten_seconds_apart_count_as_one() -> None:
    assert dedupe(with_item_id(listens(0, 10))).height == 1


def test_two_plays_forty_seconds_apart_count_as_two() -> None:
    assert dedupe(with_item_id(listens(0, 40))).height == 2


def test_exactly_thirty_seconds_is_inside_the_window() -> None:
    # The rule is "within 30 seconds", so a 30 second gap collapses.
    assert dedupe(with_item_id(listens(0, 30))).height == 1
    assert dedupe(with_item_id(listens(0, 31))).height == 2


def test_rapid_burst_collapses_to_one() -> None:
    assert dedupe(with_item_id(listens(0, 5, 10, 15))).height == 1


def test_different_users_are_not_deduped_against_each_other() -> None:
    frame = pl.concat([listens(0, user="u1"), listens(10, user="u2")])
    assert dedupe(with_item_id(frame)).height == 2


def test_different_recordings_are_not_deduped_against_each_other() -> None:
    frame = pl.concat([listens(0, recording="rec-1"), listens(10, recording="rec-2")])
    assert dedupe(with_item_id(frame)).height == 2


def test_item_id_prefers_recording_mbid() -> None:
    assert with_item_id(listens(0))["item_id"].to_list() == ["rec-1"]


def test_item_id_falls_back_to_normalised_artist_and_track() -> None:
    frame = listens(0).with_columns(pl.lit(None, dtype=pl.Utf8).alias("recording_mbid"))
    assert with_item_id(frame)["item_id"].to_list() == ["artist one|track one"]


def test_fallback_normalisation_collapses_case_and_whitespace() -> None:
    frame = listens(0).with_columns(
        pl.lit("").alias("recording_mbid"),
        pl.lit("  ARTIST   One ").alias("artist_name"),
        pl.lit("Track\tOne").alias("track_name"),
    )
    assert with_item_id(frame)["item_id"].to_list() == ["artist one|track one"]


def test_clean_is_idempotent() -> None:
    once = clean(listens(0, 10, 40))
    assert clean(once).height == once.height
