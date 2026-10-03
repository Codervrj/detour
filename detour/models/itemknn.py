"""Item-kNN baseline on co-listening.

Item-item cosine similarity over the binary user-item matrix, then score each candidate
by how similar it is to what the user already played. Simple, strong on relevance, and
famously conservative on discovery, which makes it the right thing to measure against.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from detour.config import write_sidecar
from detour.models.matrix import build


def similarity(matrix: csr_matrix, neighbours: int) -> np.ndarray:
    """Item-item cosine similarity, keeping only the top `neighbours` per item."""
    binary = (matrix > 0).astype(np.float64)
    norms = np.sqrt(np.asarray(binary.multiply(binary).sum(axis=0)).ravel())
    norms[norms < 1e-12] = 1.0

    sim = np.asarray((binary.T @ binary).todense())
    sim = sim / norms[:, None] / norms[None, :]
    np.fill_diagonal(sim, 0.0)

    if neighbours < sim.shape[1]:
        cutoff = np.partition(sim, -neighbours, axis=1)[:, -neighbours][:, None]
        sim[sim < cutoff] = 0.0
    return np.asarray(sim, dtype=np.float64)


def candidates(train: pl.DataFrame, n: int, neighbours: int = 50) -> pl.DataFrame:
    """Top-n unplayed items per user, scored by similarity to their train history."""
    interactions = build(train, alpha=0.0)
    sim = similarity(interactions.matrix, neighbours)
    played = (interactions.matrix > 0).astype(np.float64)
    scores = np.asarray(played @ sim)

    # Never recommend something the user already played.
    scores[played.nonzero()] = -np.inf

    frames: list[pl.DataFrame] = []
    for user_id in interactions.users:
        row = scores[interactions.user_index[user_id]]
        take = min(n, int(np.isfinite(row).sum()))
        order = np.argsort(-row)[:take]
        frames.append(
            pl.DataFrame(
                {
                    "user_id": [user_id] * take,
                    "item_id": [interactions.items[i] for i in order],
                    "score": row[order].astype(np.float64),
                    "rank": np.arange(1, take + 1, dtype=np.int32),
                }
            )
        )
    return pl.concat(frames)


def main(
    source: str = "data/processed/train.parquet",
    out_path: str = "artifacts/itemknn_candidates.parquet",
    n: int = 200,
) -> None:
    """Write top-n item-kNN candidates per user."""
    train = pl.read_parquet(source)
    table = candidates(train, n)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(target)
    write_sidecar(
        target,
        {
            "stage": "itemknn",
            "computed_from": source,
            "users": table["user_id"].n_unique(),
            "rows": table.height,
            "candidates_per_user": n,
        },
    )
    print(f"item-knn: {table.height} candidates for {table['user_id'].n_unique()} users")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Item-kNN baseline candidates.")
    parser.add_argument("--source", default="data/processed/train.parquet")
    parser.add_argument("--out", default="artifacts/itemknn_candidates.parquet")
    parser.add_argument("-n", type=int, default=200)
    args = parser.parse_args()
    main(args.source, args.out, args.n)
