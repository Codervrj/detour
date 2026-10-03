"""Listening sessions: the sequences the artist map is learned from.

A session is a run of plays by one listener with no gap longer than 30 minutes. The gap
rule is what makes a session meaningful: plays inside one sitting share a mood and a
context, so artists that keep appearing together in sessions really are related, whereas
"played in the same calendar month" says almost nothing.

Each session becomes a sequence of artist ids, which is what `item2vec` consumes. The
analogy is a sentence: artists are words, sessions are sentences, and artists used in
similar contexts end up with similar vectors.

Consecutive repeats are collapsed. Somebody playing one album start to finish would
otherwise produce `A A A A A`, teaching the model only that an artist resembles itself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from detour.config import write_sidecar

SESSION_GAP_MINUTES = 30
MIN_SESSION_LENGTH = 2


def with_session_id(frame: pl.DataFrame, gap_minutes: int = SESSION_GAP_MINUTES) -> pl.DataFrame:
    """Label every listen with a session number, counted per listener."""
    gap_seconds = gap_minutes * 60
    ordered = frame.sort(["user_id", "listened_at"])
    return ordered.with_columns(
        (
            (
                pl.col("listened_at").diff().dt.total_seconds().over("user_id") > gap_seconds
            ).fill_null(True)
        )
        .cum_sum()
        .over("user_id")
        .alias("session_id")
    )


def collapse_repeats(artists: pl.Expr) -> pl.Expr:
    """Drop immediately repeated artists within a session."""
    return artists.list.eval(
        pl.element().filter(pl.element() != pl.element().shift(1).fill_null(""))
    )


def build(
    frame: pl.DataFrame,
    gap_minutes: int = SESSION_GAP_MINUTES,
    min_length: int = MIN_SESSION_LENGTH,
) -> pl.DataFrame:
    """One row per session: the listener and their ordered artist sequence."""
    labelled = with_session_id(frame, gap_minutes)
    sessions = (
        labelled.group_by(["user_id", "session_id"], maintain_order=True)
        .agg(pl.col("artist_mbid").alias("artists"))
        .with_columns(collapse_repeats(pl.col("artists")).alias("artists"))
        .with_columns(pl.col("artists").list.len().alias("length"))
    )
    # A session of one teaches the model nothing: there is no context to learn from.
    return sessions.filter(pl.col("length") >= min_length)


def main(
    source: str = "data/processed/train.parquet",
    out_path: str = "data/processed/sessions.parquet",
    gap_minutes: int = SESSION_GAP_MINUTES,
) -> None:
    """Write the session sequences the embedding is trained on."""
    frame = pl.read_parquet(source)
    sessions = build(frame, gap_minutes)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sessions.write_parquet(target)

    lengths = sessions["length"]
    write_sidecar(
        target,
        {
            "stage": "sessions",
            "computed_from": source,
            "gap_minutes": gap_minutes,
            "min_session_length": MIN_SESSION_LENGTH,
            "sessions": sessions.height,
            "listeners": sessions["user_id"].n_unique(),
            "length_median": float(lengths.median() or 0),  # type: ignore[arg-type]
            "length_max": int(lengths.max() or 0),  # type: ignore[arg-type]
            "total_positions": int(lengths.sum() or 0),
        },
    )
    print(
        f"sessions: {sessions.height:,} from {sessions['user_id'].n_unique():,} listeners, "
        f"median length {float(lengths.median() or 0):.0f}"  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build listening sessions.")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out", default="data/processed/sessions.parquet")
    parser.add_argument("--gap-minutes", type=int, default=SESSION_GAP_MINUTES)
    args = parser.parse_args()
    main(args.source, args.out, args.gap_minutes)
