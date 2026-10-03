"""Per-user explorer score from train history.

For each month, the score looks at the share of that month's listens going to artists the
user is hearing for the first time. Those monthly shares are smoothed with a rolling mean
and averaged into a single number in [0, 1].

A user's first active month is excluded: every artist is new in the first month, so
including it would hand a loyalist a high score for no reason. With that exclusion a user
who only ever replays one artist scores exactly 0, which is the behaviour section 8 asks
for. A user with only one active month has no evidence of exploration and scores 0.

Parameters live in the config under `explorer_score` (see evals/configs/).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from detour.config import load_config, write_sidecar


def with_month_index(frame: pl.DataFrame) -> pl.DataFrame:
    """Add `month`: a 0-based month counter relative to the earliest listen in the frame."""
    origin = frame.select(pl.col("listened_at").min()).item()
    return frame.with_columns(
        (
            (pl.col("listened_at").dt.year() - origin.year) * 12
            + (pl.col("listened_at").dt.month() - origin.month)
        ).alias("month")
    )


def monthly_new_artist_share(frame: pl.DataFrame) -> pl.DataFrame:
    """Per user and month: the share of listens going to artists first heard that month."""
    dated = with_month_index(frame)
    first_seen = dated.group_by(["user_id", "artist_mbid"]).agg(
        pl.col("month").min().alias("first_month")
    )
    return (
        dated.join(first_seen, on=["user_id", "artist_mbid"], how="left")
        .with_columns((pl.col("month") == pl.col("first_month")).alias("is_new"))
        .group_by(["user_id", "month"])
        .agg(pl.col("is_new").sum().alias("new_listens"), pl.len().alias("total_listens"))
        .with_columns((pl.col("new_listens") / pl.col("total_listens")).alias("share"))
        .sort(["user_id", "month"])
    )


def drop_first_active_month(monthly: pl.DataFrame) -> pl.DataFrame:
    """Remove each user's first active month, where every artist is new by definition."""
    first = monthly.group_by("user_id").agg(pl.col("month").min().alias("first_active"))
    return monthly.join(first, on="user_id", how="left").filter(
        pl.col("month") > pl.col("first_active")
    )


def smooth_and_average(monthly: pl.DataFrame, window: int) -> pl.DataFrame:
    """Rolling-mean the monthly shares, then average them into one score per user."""
    smoothed = monthly.with_columns(
        pl.col("share")
        .rolling_mean(window_size=window, min_samples=1)
        .over("user_id")
        .alias("smoothed")
    )
    return smoothed.group_by("user_id").agg(
        pl.col("smoothed").mean().alias("explorer_score"),
        pl.len().alias("months_counted"),
    )


def explorer_scores(train: pl.DataFrame, window: int, clip: tuple[float, float]) -> pl.DataFrame:
    """Explorer score per train user, clipped to the configured range."""
    monthly = drop_first_active_month(monthly_new_artist_share(train))
    scored = smooth_and_average(monthly, window)

    # Users with a single active month drop out above; give them 0 and keep every train user.
    all_users = train.select("user_id").unique()
    return (
        all_users.join(scored, on="user_id", how="left")
        .with_columns(
            pl.col("explorer_score").fill_null(0.0).clip(clip[0], clip[1]),
            pl.col("months_counted").fill_null(0),
        )
        .sort("user_id")
    )


def main(
    config_path: str = "evals/configs/smoke.yaml",
    source: str = "data/processed/train.parquet",
    out_path: str = "artifacts/explorer_scores.parquet",
) -> None:
    """Write artifacts/explorer_scores.parquet."""
    config = load_config(config_path)["explorer_score"]
    low, high = config["clip"]

    train = pl.read_parquet(source)
    scores = explorer_scores(train, config["smoothing_months"], (low, high))

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scores.write_parquet(target)

    write_sidecar(
        target,
        {
            "stage": "explorer_score",
            "computed_from": source,
            "users": scores.height,
            "smoothing_months": config["smoothing_months"],
            "clip": [low, high],
            "score_min": scores["explorer_score"].min(),
            "score_max": scores["explorer_score"].max(),
            "score_mean": scores["explorer_score"].mean(),
        },
    )
    column = scores["explorer_score"]
    low, high = float(column.min()), float(column.max())  # type: ignore[arg-type]
    print(
        f"explorer score: {scores.height} users, "
        f"range [{low:.3f}, {high:.3f}], mean {float(column.mean()):.3f}"  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-user explorer score from train.")
    parser.add_argument("--config", default="evals/configs/smoke.yaml")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out", default="artifacts/explorer_scores.parquet")
    args = parser.parse_args()
    main(args.config, args.source, args.out)
