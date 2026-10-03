"""The serving layer: the map, and where a person sits on it.

Everything the API and the UI need comes from here, so a number shown on screen is produced
by the same code the evaluation measured. Nothing is trained at request time; the map is
read from `artifacts/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from detour.foldin import Placement, nearest, place
from detour.ingest.resolve import Resolution, resolve


@dataclass(frozen=True)
class Artist:
    """One artist as the UI shows them."""

    artist_mbid: str
    name: str
    plays: int
    x: float
    y: float


@dataclass(frozen=True)
class Island:
    """A cluster of a listener's artists: one coherent corner of their taste."""

    label: str
    artists: list[str]
    size: int
    plays: int
    isolation: float


class MapService:
    """Holds the artist map in memory and answers questions about one listener."""

    def __init__(self, artifacts_dir: str = "artifacts", processed_dir: str = "data/processed"):
        root = Path(artifacts_dir)
        frame = pl.read_parquet(root / "artist_vectors.parquet")

        self.artists: list[str] = frame["artist_mbid"].to_list()
        self.plays: dict[str, int] = dict(zip(self.artists, frame["count"].to_list(), strict=True))
        matrix = np.asarray(frame["vector"].to_list(), dtype=np.float64)
        self.matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
        self.vectors: dict[str, np.ndarray] = dict(zip(self.artists, matrix, strict=True))
        self.index = {artist: i for i, artist in enumerate(self.artists)}

        names_path = Path(processed_dir) / "artist_names.parquet"
        self.names: dict[str, str] = {}
        if names_path.exists():
            names = pl.read_parquet(names_path)
            self.names = dict(
                zip(names["artist_mbid"].to_list(), names["artist_name"].to_list(), strict=True)
            )
        self.name_table = (
            pl.read_parquet(names_path)
            if names_path.exists()
            else pl.DataFrame({"artist_mbid": [], "artist_name": []})
        )

        coords_path = root / "map_2d.parquet"
        self.coords: dict[str, tuple[float, float]] = {}
        if coords_path.exists():
            coords = pl.read_parquet(coords_path)
            self.coords = {
                str(row["artist_mbid"]): (float(row["x"]), float(row["y"]))
                for row in coords.iter_rows(named=True)
            }

    @property
    def vocabulary(self) -> set[str]:
        return set(self.index)

    def name_of(self, artist_mbid: str) -> str:
        """A display name, falling back to the id when MusicBrainz has no name for it."""
        return self.names.get(artist_mbid, artist_mbid[:8])

    def position_of(self, artist_mbid: str) -> tuple[float, float]:
        return self.coords.get(artist_mbid, (0.0, 0.0))

    def to_artist(self, artist_mbid: str, plays: int) -> Artist:
        x, y = self.position_of(artist_mbid)
        return Artist(artist_mbid, self.name_of(artist_mbid), plays, x, y)

    def resolve_history(self, listens: pl.DataFrame) -> Resolution:
        """Match a personal history to the map's artists."""
        return resolve(listens, self.name_table, self.vocabulary)

    def place_counts(self, counts: dict[str, int]) -> Placement:
        """Fold a listener in from resolved play counts."""
        return place(counts, self.vectors)

    def neighbours(self, artist_mbid: str, k: int = 10) -> list[tuple[str, float]]:
        """The k artists closest to one artist."""
        if artist_mbid not in self.index:
            return []
        similarity = self.matrix @ self.matrix[self.index[artist_mbid]]
        order = np.argsort(-similarity)[: k + 1]
        return [
            (self.artists[i], float(similarity[i])) for i in order if self.artists[i] != artist_mbid
        ][:k]

    def frontier(
        self, placement: Placement, counts: dict[str, int], k: int = 20
    ) -> list[tuple[str, float]]:
        """Artists nearest this listener's position that they have not played.

        Measured from the listener's centre, which is **the exact ranking the evaluation
        scored** at an adoption rank percentile of 0.1158. Anchoring on their edge artists
        was tried and rejected: edges are obscure outliers, so the result was obscure
        artists near other obscure artists, and nothing on screen would have had evidence
        behind it.
        """
        return nearest(placement.position, self.vectors, k, exclude=set(counts))

    def islands(self, counts: dict[str, int], max_islands: int = 6) -> list[Island]:
        """Cluster a listener's own artists into coherent groups.

        KMeans with k chosen by silhouette score, so the number of islands is discovered
        rather than asserted. Each island is labelled with its most played artist, which is
        more honest than inventing a genre name the data does not contain.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        known = [a for a in counts if a in self.vectors]
        if len(known) < 4:
            return []

        points = np.vstack([self.vectors[a] for a in known])
        points = points / (np.linalg.norm(points, axis=1, keepdims=True) + 1e-12)

        best_k, best_score, best_labels = 2, -1.0, None
        for k in range(2, min(max_islands, len(known) - 1) + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=13).fit_predict(points)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(points, labels))
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        if best_labels is None:
            return []

        islands: list[Island] = []
        centre = points.mean(axis=0)
        for cluster in range(best_k):
            members = [known[i] for i, label in enumerate(best_labels) if label == cluster]
            if not members:
                continue
            member_points = np.vstack([self.vectors[a] for a in members])
            member_points /= np.linalg.norm(member_points, axis=1, keepdims=True) + 1e-12
            cluster_centre = member_points.mean(axis=0)
            isolation = float(
                1.0
                - (cluster_centre @ centre)
                / ((np.linalg.norm(cluster_centre) * np.linalg.norm(centre)) + 1e-12)
            )
            top = max(members, key=lambda a: counts.get(a, 0))
            islands.append(
                Island(
                    label=self.name_of(top),
                    artists=sorted(members, key=lambda a: -counts.get(a, 0)),
                    size=len(members),
                    plays=sum(counts.get(a, 0) for a in members),
                    isolation=round(isolation, 4),
                )
            )
        return sorted(islands, key=lambda island: -island.plays)
