"""The user-item confidence matrix shared by the implicit-feedback models.

Confidence follows CLAUDE.md section 6: `1 + alpha * log(1 + play_count)`, so a user who
played something forty times counts for more than one who played it once, but not forty
times more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Interactions:
    """A users-by-items confidence matrix plus the id lookups in both directions."""

    matrix: csr_matrix
    users: list[str]
    items: list[str]
    user_index: dict[str, int]
    item_index: dict[str, int]

    def row_for(self, user_id: str) -> csr_matrix:
        """The single-row matrix `implicit` expects when recommending for one user."""
        return self.matrix[self.user_index[user_id]]


def confidence(plays: pl.Expr, alpha: float) -> pl.Expr:
    """1 + alpha * log(1 + play_count)."""
    return 1.0 + alpha * (1.0 + plays).log()


def build(train: pl.DataFrame, alpha: float) -> Interactions:
    """Aggregate train listens into one confidence-weighted row per user."""
    counts = train.group_by(["user_id", "item_id"]).agg(pl.len().alias("plays"))
    counts = counts.with_columns(confidence(pl.col("plays"), alpha).alias("c"))

    users = sorted(counts["user_id"].unique().to_list())
    items = sorted(counts["item_id"].unique().to_list())
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {t: i for i, t in enumerate(items)}

    rows = np.array([user_index[u] for u in counts["user_id"].to_list()], dtype=np.int32)
    cols = np.array([item_index[t] for t in counts["item_id"].to_list()], dtype=np.int32)
    values = np.asarray(counts["c"].to_list(), dtype=np.float32)

    matrix = csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))
    return Interactions(matrix, users, items, user_index, item_index)


def played_items(train: pl.DataFrame) -> dict[str, set[str]]:
    """Items each user already played in train, used to keep them out of recommendations."""
    grouped = train.group_by("user_id").agg(pl.col("item_id").unique().alias("items"))
    return {row["user_id"]: set(row["items"]) for row in grouped.iter_rows(named=True)}


def known_artists(train: pl.DataFrame) -> dict[str, set[str]]:
    """Artists each user already heard in train. Drives the familiarity anchor rule."""
    grouped = train.group_by("user_id").agg(pl.col("artist_mbid").unique().alias("artists"))
    return {row["user_id"]: set(row["artists"]) for row in grouped.iter_rows(named=True)}
