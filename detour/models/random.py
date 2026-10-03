"""Random baseline. Seeded, so a report can be reproduced exactly.

This is the floor every other model has to clear. It samples uniformly from the train
catalogue, skipping what the user already played.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from detour.config import write_sidecar
from detour.models.matrix import played_items


def candidates(train: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Up to n random unplayed items per user."""
    rng = np.random.default_rng(seed)
    catalogue = sorted(train["item_id"].unique().to_list())
    already = played_items(train)

    frames: list[pl.DataFrame] = []
    for user_id in sorted(already):
        seen = already[user_id]
        pool = [item for item in catalogue if item not in seen]
        size = min(n, len(pool))
        picks = rng.choice(pool, size=size, replace=False) if size else np.array([], dtype=str)
        frames.append(
            pl.DataFrame(
                {
                    "user_id": [user_id] * size,
                    "item_id": [str(item) for item in picks],
                    "score": np.linspace(1.0, 0.0, size, dtype=np.float64) if size else [],
                    "rank": list(range(1, size + 1)),
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
    out_path: str = "artifacts/random_candidates.parquet",
    n: int = 200,
    seed: int = 13,
) -> None:
    """Write top-n random candidates per user."""
    train = pl.read_parquet(source)
    table = candidates(train, n, seed)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(target)
    write_sidecar(
        target,
        {
            "stage": "random_baseline",
            "computed_from": source,
            "seed": seed,
            "users": table["user_id"].n_unique(),
            "rows": table.height,
            "candidates_per_user": n,
        },
    )
    print(f"random baseline: {table.height} candidates for {table['user_id'].n_unique()} users")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random baseline candidates.")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out", default="artifacts/random_candidates.parquet")
    parser.add_argument("-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    main(args.source, args.out, args.n, args.seed)
