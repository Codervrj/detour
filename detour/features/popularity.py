"""Item popularity from train only.

`p_i` is the share of train users who played item i. It feeds Novelty@K, the popularity
baseline and the novelty term in the re-ranker. Computed from train so that nothing in
val or test can influence it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from detour.config import write_sidecar


def popularity_table(train: pl.DataFrame) -> pl.DataFrame:
    """Per item: distinct listeners, play count, and p = listeners / train users."""
    n_users = train["user_id"].n_unique()
    return (
        train.group_by("item_id")
        .agg(
            pl.col("user_id").n_unique().alias("listeners"),
            pl.len().alias("plays"),
            pl.col("artist_mbid").first().alias("artist_mbid"),
        )
        .with_columns((pl.col("listeners") / n_users).alias("p"))
        .sort("listeners", descending=True)
    )


def main(
    source: str = "data/processed/train.parquet",
    out_path: str = "artifacts/popularity.parquet",
) -> None:
    """Write artifacts/popularity.parquet: item id, listener count, play count, p."""
    train = pl.read_parquet(source)
    table = popularity_table(train)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(target)

    write_sidecar(
        target,
        {
            "stage": "popularity",
            "computed_from": source,
            "items": table.height,
            "train_users": train["user_id"].n_unique(),
            "p_min": table["p"].min(),
            "p_max": table["p"].max(),
        },
    )
    low, high = float(table["p"].min()), float(table["p"].max())  # type: ignore[arg-type]
    print(f"popularity: {table.height} items, p in [{low:.3f}, {high:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Item popularity from train only.")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out", default="artifacts/popularity.parquet")
    args = parser.parse_args()
    main(args.source, args.out)
