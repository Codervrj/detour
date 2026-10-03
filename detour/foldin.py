"""Place a listener on the learned map.

This is what makes the project about you. The embedding is trained on thousands of other
people; you were never in it. Your position is the play-count-weighted mean of the vectors
of the artists you listen to, which is the standard fold-in for an unseen user and needs no
retraining.

Two honesty rules are enforced here:

1. **Coverage is reported, never hidden.** Artists the model has never seen cannot be
   placed. You are told how much of your history was actually readable.
2. **Weighting is by log play count, not raw count.** Somebody with 4,000 plays of one band
   would otherwise be represented as a point sitting on that band, drowning the rest of
   their taste.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl


@dataclass
class Placement:
    """Where a listener sits, and how much of them the model could read."""

    position: np.ndarray
    artist_vectors: dict[str, np.ndarray] = field(repr=False)
    weights: dict[str, float] = field(repr=False)
    matched_artists: int = 0
    unmatched_artists: int = 0
    matched_plays: int = 0
    unmatched_plays: int = 0

    @property
    def artist_coverage(self) -> float:
        """Share of this listener's distinct artists the model recognised."""
        total = self.matched_artists + self.unmatched_artists
        return self.matched_artists / total if total else 0.0

    @property
    def play_coverage(self) -> float:
        """Share of this listener's plays the model recognised.

        Usually higher than artist coverage, because the artists it misses are the obscure
        ones in the tail that were played only once or twice.
        """
        total = self.matched_plays + self.unmatched_plays
        return self.matched_plays / total if total else 0.0


def unit(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length, leaving a zero vector alone."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def play_counts(listens: pl.DataFrame) -> dict[str, int]:
    """Plays per artist for one listener."""
    counted = listens.group_by("artist_mbid").agg(pl.len().alias("plays"))
    return {
        str(row["artist_mbid"]): int(row["plays"])
        for row in counted.iter_rows(named=True)
        if row["artist_mbid"]
    }


def place(counts: dict[str, int], vectors: dict[str, np.ndarray]) -> Placement:
    """Fold a listener into the embedding space from their artist play counts."""
    matched: dict[str, np.ndarray] = {}
    weights: dict[str, float] = {}
    matched_plays = unmatched_plays = 0
    unmatched_artists = 0

    for artist, plays in counts.items():
        vector = vectors.get(artist)
        if vector is None:
            unmatched_artists += 1
            unmatched_plays += plays
            continue
        matched[artist] = vector
        # log1p keeps a heavy rotation influential without letting it dominate.
        weights[artist] = math.log1p(plays)
        matched_plays += plays

    if matched:
        stacked = np.vstack([unit(matched[a]) for a in matched])
        weight_vector = np.array([weights[a] for a in matched], dtype=np.float64)
        position = unit(np.average(stacked, axis=0, weights=weight_vector))
    else:
        dimension = len(next(iter(vectors.values()))) if vectors else 0
        position = np.zeros(dimension, dtype=np.float64)

    return Placement(
        position=position,
        artist_vectors=matched,
        weights=weights,
        matched_artists=len(matched),
        unmatched_artists=unmatched_artists,
        matched_plays=matched_plays,
        unmatched_plays=unmatched_plays,
    )


def place_listens(listens: pl.DataFrame, vectors: dict[str, np.ndarray]) -> Placement:
    """Convenience wrapper: raw listens straight to a placement."""
    return place(play_counts(listens), vectors)


def distances_from(
    position: np.ndarray, vectors: dict[str, np.ndarray]
) -> tuple[list[str], np.ndarray]:
    """Cosine distance from a position to every artist, as aligned lists."""
    artists = list(vectors)
    if not artists:
        return [], np.zeros(0)
    matrix = np.vstack([unit(vectors[a]) for a in artists])
    return artists, 1.0 - matrix @ unit(position)


def nearest(
    position: np.ndarray, vectors: dict[str, np.ndarray], k: int, exclude: set[str] | None = None
) -> list[tuple[str, float]]:
    """The k artists closest to a position, optionally skipping ones already known."""
    artists, distance = distances_from(position, vectors)
    skip = exclude or set()
    ranked = sorted(
        ((a, float(d)) for a, d in zip(artists, distance, strict=True) if a not in skip),
        key=lambda pair: pair[1],
    )
    return ranked[:k]


def edge_artists(placement: Placement, k: int = 10) -> list[tuple[str, float]]:
    """The listener's own artists that sit furthest from their centre.

    These are the bridges: the places where their taste already touches territory they have
    not otherwise explored.
    """
    if not placement.artist_vectors:
        return []
    artists, distance = distances_from(placement.position, placement.artist_vectors)
    ranked = sorted(
        zip(artists, (float(d) for d in distance), strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:k]
