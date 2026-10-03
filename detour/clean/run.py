"""Dedupe scrobbles and settle on a stable item identity.

Dedupe rule: the same user playing the same recording within 30 seconds is one listen.
The rule is applied pairwise over consecutive plays, so a burst of rapid repeats
collapses to its first play.

Identity: recording MBID when present, otherwise a normalised `(artist, track)` string.
Artist-level analysis uses the artist MBID.

User and item filters are deliberately NOT applied here. They must be computed from the
train period only, so they live in `detour.split` after the time split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from detour.config import DEDUPE_WINDOW_SECONDS, write_sidecar


def normalise(column: str) -> pl.Expr:
    """Lowercase, trim and collapse internal whitespace, for the no-MBID fallback."""
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
    )


def with_item_id(frame: pl.DataFrame) -> pl.DataFrame:
    """Add `item_id`: the recording MBID, falling back to normalised artist and track.

    MLHD carries MusicBrainz ids and no names at all, so the fallback columns may be
    absent. In that case the recording id is the only identity available and is used
    directly; rows missing it are dropped by the caller rather than silently merged.
    """
    has_names = {"artist_name", "track_name"} <= set(frame.columns)
    if not has_names:
        return frame.with_columns(pl.col("recording_mbid").alias("item_id"))

    fallback = normalise("artist_name") + pl.lit("|") + normalise("track_name")
    return frame.with_columns(
        pl.when(pl.col("recording_mbid").is_null() | (pl.col("recording_mbid") == ""))
        .then(fallback)
        .otherwise(pl.col("recording_mbid"))
        .alias("item_id")
    )


def dedupe(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse repeat plays of the same item by the same user inside the dedupe window."""
    gap = pl.col("listened_at").diff().dt.total_seconds().over(["user_id", "item_id"]).alias("_gap")
    return (
        frame.sort(["user_id", "item_id", "listened_at"])
        .with_columns(gap)
        .filter(pl.col("_gap").is_null() | (pl.col("_gap") > DEDUPE_WINDOW_SECONDS))
        .drop("_gap")
        .sort(["user_id", "listened_at"])
    )


def clean(frame: pl.DataFrame) -> pl.DataFrame:
    """Apply identity resolution then dedupe."""
    return dedupe(with_item_id(frame))


def main(
    source: str = "tests/fixtures/listens.parquet",
    out_path: str = "data/processed/listens.parquet",
) -> None:
    """Clean a raw listens Parquet into data/processed/ with a JSON sidecar."""
    raw = pl.read_parquet(source)
    cleaned = clean(raw)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(target)

    write_sidecar(
        target,
        {
            "stage": "clean",
            "source": source,
            "rows_in": raw.height,
            "rows_out": cleaned.height,
            "duplicates_removed": raw.height - cleaned.height,
            "dedupe_window_seconds": DEDUPE_WINDOW_SECONDS,
            "users": cleaned["user_id"].n_unique(),
            "items": cleaned["item_id"].n_unique(),
            "artists": cleaned["artist_mbid"].n_unique(),
            "date_min": cleaned["listened_at"].min(),
            "date_max": cleaned["listened_at"].max(),
        },
    )
    print(
        f"clean: {raw.height} -> {cleaned.height} rows ({raw.height - cleaned.height} duplicates)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedupe and normalise raw listens.")
    parser.add_argument("--source", default="tests/fixtures/listens.parquet")
    parser.add_argument("--out", default="data/processed/listens.parquet")
    args = parser.parse_args()
    main(args.source, args.out)
