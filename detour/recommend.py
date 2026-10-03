"""The shared serving path: artefacts in, a re-ranked list out.

Both the eval harness and the API go through here, so a number shown in the UI is produced
by exactly the code the report measured. Nothing is trained at this point; everything is
read from `artifacts/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from detour.rerank.mmr import Candidate, prepare, rerank
from detour.rerank.personalized_lambda import lambda_for


@dataclass
class Recommendation:
    """One row of a finished list, with the numbers the UI shows next to it."""

    item_id: str
    rank: int
    relevance: float
    novelty: float
    is_anchor: bool
    artist_mbid: str
    artist_name: str
    track_name: str
    why: str


class Recommender:
    """Holds the artefacts in memory and re-ranks on request."""

    def __init__(self, artifacts_dir: str = "artifacts", processed_dir: str = "data/processed"):
        root = Path(artifacts_dir)
        self.candidates = pl.read_parquet(root / "als_candidates.parquet")
        self.popularity = pl.read_parquet(root / "popularity.parquet")
        self.explorer = pl.read_parquet(root / "explorer_scores.parquet")
        self.factors = pl.read_parquet(root / "item_factors.parquet")

        train = pl.read_parquet(Path(processed_dir) / "train.parquet")
        self.catalogue = self._catalogue(train)
        self.known_artists = {
            row["user_id"]: set(row["artists"])
            for row in train.group_by("user_id")
            .agg(pl.col("artist_mbid").unique().alias("artists"))
            .iter_rows(named=True)
        }
        self.top_artists = self._top_artists(train)
        self.train_listens = dict(train.group_by("user_id").agg(pl.len().alias("n")).iter_rows())

        self._p = dict(zip(self.popularity["item_id"], self.popularity["p"], strict=True))
        self._factors = {
            row["item_id"]: np.asarray(row["factors"], dtype=np.float64)
            for row in self.factors.iter_rows(named=True)
        }
        self._explorer = dict(
            zip(self.explorer["user_id"], self.explorer["explorer_score"], strict=True)
        )

    @staticmethod
    def _catalogue(train: pl.DataFrame) -> dict[str, dict[str, str]]:
        """Item id to its artist and track names, for display and explanations."""
        rows = train.group_by("item_id").agg(
            pl.col("artist_mbid").first().alias("artist_mbid"),
            pl.col("artist_name").first().alias("artist_name"),
            pl.col("track_name").first().alias("track_name"),
        )
        return {row["item_id"]: row for row in rows.iter_rows(named=True)}

    @staticmethod
    def _top_artists(train: pl.DataFrame) -> dict[str, list[str]]:
        """Each user's most played artist names, used in the 'why this' line."""
        counted = (
            train.group_by(["user_id", "artist_name"])
            .agg(pl.len().alias("plays"))
            .sort(["user_id", "plays"], descending=[False, True])
        )
        grouped = counted.group_by("user_id").agg(pl.col("artist_name").alias("artists"))
        return {row["user_id"]: list(row["artists"])[:3] for row in grouped.iter_rows(named=True)}

    def users(self) -> list[str]:
        """Every user who has candidates and an explorer score."""
        return sorted(set(self.candidates["user_id"].to_list()) & set(self._explorer))

    def explorer_score(self, user_id: str) -> float:
        """The user's explorer score, or 0 if they were never scored."""
        return float(self._explorer.get(user_id, 0.0))

    def personalised_lambda(self, user_id: str, config: dict[str, Any]) -> float:
        """The lambda this user would get from their explorer score."""
        return lambda_for(self.explorer_score(user_id), config["personalized_lambda"])

    def candidates_for(self, user_id: str) -> list[Candidate]:
        """Normalised candidates for one user, ready for the re-ranker."""
        rows = self.candidates.filter(pl.col("user_id") == user_id).sort("rank")
        if rows.height == 0:
            return []

        item_ids = rows["item_id"].to_list()
        known = self.known_artists.get(user_id, set())
        anchors = [
            self.catalogue.get(item, {}).get("artist_mbid", "") in known for item in item_ids
        ]
        return prepare(
            item_ids=item_ids,
            relevances=rows["score"].to_list(),
            popularities=[float(self._p.get(item, 1e-6)) for item in item_ids],
            anchors=anchors,
            factors=[self._factors.get(item, np.zeros(1)) for item in item_ids],
        )

    def why(self, user_id: str, item_id: str, is_anchor: bool) -> str:
        """A short plain-language reason this track is in the list."""
        meta = self.catalogue.get(item_id, {})
        artist = meta.get("artist_name", "this artist")
        if is_anchor:
            return f"You already listen to {artist}."
        favourites = self.top_artists.get(user_id, [])
        if favourites:
            return f"A step out from {favourites[0]}, who you play often."
        return "A new artist picked from listeners with taste like yours."

    def recommend(
        self, user_id: str, lam: float, k: int, beta: float, anchor_min: int
    ) -> list[Recommendation]:
        """Re-rank this user's candidates at the given lambda."""
        candidates = self.candidates_for(user_id)
        if not candidates:
            return []

        picked = rerank(candidates, lam=lam, beta=beta, k=k, anchor_min=anchor_min)
        results: list[Recommendation] = []
        for position, candidate in enumerate(picked, start=1):
            meta = self.catalogue.get(candidate.item_id, {})
            results.append(
                Recommendation(
                    item_id=candidate.item_id,
                    rank=position,
                    relevance=round(candidate.relevance, 4),
                    novelty=round(candidate.novelty, 4),
                    is_anchor=candidate.is_anchor,
                    artist_mbid=meta.get("artist_mbid", ""),
                    artist_name=meta.get("artist_name", "Unknown artist"),
                    track_name=meta.get("track_name", "Unknown track"),
                    why=self.why(user_id, candidate.item_id, candidate.is_anchor),
                )
            )
        return results
