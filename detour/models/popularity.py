"""Popularity baseline: the most played train items the user has not already heard.

Every report includes this baseline, and section 7 requires "ours" to beat it on NDCG@20.
It also defines the top-50 list that Serendipity@K measures against.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from detour.config import write_sidecar
from detour.features.popularity import popularity_table
from detour.models.matrix import played_items


def candidates(train: pl.DataFrame, n: int) -> pl.DataFrame:
    """Top-n most popular unplayed items per user, ranked by distinct listeners."""
    ranked = popularity_table(train).select("item_id", "listeners").to_dicts()
    already = played_items(train)

    frames: list[pl.DataFrame] = []
    for user_id in sorted(already):
        seen = already[user_id]
        picks = [row for row in ranked if row["item_id"] not in seen][:n]
        frames.append(
            pl.DataFrame(
                {
                    "user_id": [user_id] * len(picks),
                    "item_id": [row["item_id"] for row in picks],
                    "score": [float(row["listeners"]) for row in picks],
                    "rank": list(range(1, len(picks) + 1)),
                },
                schema={
                    "user_id": pl.Utf8,
                    "item_id": pl.Utf8,
                    "score": pl.Float64,
                    "rank": pl.Int32,
                },
            )
        )
    return pl.concat(frames)


def main(
    source: str = "data/processed/train.parquet",
    out_path: str = "artifacts/popularity_candidates.parquet",
    n: int = 200,
) -> None:
    """Write top-n popularity candidates per user."""
    train = pl.read_parquet(source)
    table = candidates(train, n)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(target)
    write_sidecar(
        target,
        {
            "stage": "popularity_baseline",
            "computed_from": source,
            "users": table["user_id"].n_unique(),
            "rows": table.height,
            "candidates_per_user": n,
        },
    )
    print(f"popularity baseline: {table.height} candidates for {table['user_id'].n_unique()} users")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Popularity baseline candidates.")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out", default="artifacts/popularity_candidates.parquet")
    parser.add_argument("-n", type=int, default=200)
    args = parser.parse_args()
    main(args.source, args.out, args.n)
