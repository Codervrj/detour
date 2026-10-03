"""Match a personal history to the artists the map knows about.

Last.fm supplies a MusicBrainz id for many scrobbles but not all, and the ids it does
supply are sometimes ones MusicBrainz has since merged. So artists are matched in two
passes: by id first, then by normalised name.

Coverage is always reported. Being told "847 of your 1,021 artists were recognised" is the
difference between a map you can trust and one that quietly dropped a third of your taste.
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

# Punctuation and accents vary wildly between sources; "Beyoncé" and "Beyonce" are one artist.
PUNCTUATION = re.compile(r"[^\w\s]")
WHITESPACE = re.compile(r"\s+")
LEADING_THE = re.compile(r"^the\s+")


def normalise_name(name: str) -> str:
    """A comparable form of an artist name."""
    folded = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in folded if not unicodedata.combining(ch))
    cleaned = PUNCTUATION.sub(" ", stripped.lower())
    collapsed = WHITESPACE.sub(" ", cleaned).strip()
    return LEADING_THE.sub("", collapsed)


def name_index(artist_names: pl.DataFrame, vocabulary: set[str]) -> dict[str, str]:
    """Normalised name to artist id, restricted to artists the map actually knows.

    Where two ids share a name, the first wins; both are in the map, so either is a
    reasonable answer and picking deterministically beats picking randomly.
    """
    index: dict[str, str] = {}
    for row in artist_names.iter_rows(named=True):
        mbid = str(row["artist_mbid"])
        if mbid not in vocabulary:
            continue
        key = normalise_name(str(row["artist_name"]))
        if key and key not in index:
            index[key] = mbid
    return index


class Resolution:
    """The outcome of matching one person's history to the map."""

    def __init__(
        self,
        counts: dict[str, int],
        matched_by_id: int,
        matched_by_name: int,
        unmatched: list[tuple[str, int]],
    ) -> None:
        self.counts = counts
        self.matched_by_id = matched_by_id
        self.matched_by_name = matched_by_name
        self.unmatched = unmatched

    @property
    def matched_artists(self) -> int:
        return len(self.counts)

    @property
    def artist_coverage(self) -> float:
        total = self.matched_artists + len(self.unmatched)
        return self.matched_artists / total if total else 0.0

    @property
    def play_coverage(self) -> float:
        matched_plays = sum(self.counts.values())
        total = matched_plays + sum(plays for _, plays in self.unmatched)
        return matched_plays / total if total else 0.0

    def summary(self) -> dict[str, object]:
        """Figures suitable for showing a person directly."""
        return {
            "matched_artists": self.matched_artists,
            "unmatched_artists": len(self.unmatched),
            "matched_by_id": self.matched_by_id,
            "matched_by_name": self.matched_by_name,
            "artist_coverage": round(self.artist_coverage, 4),
            "play_coverage": round(self.play_coverage, 4),
            "biggest_misses": [name for name, _ in self.unmatched[:10]],
        }


def resolve(listens: pl.DataFrame, artist_names: pl.DataFrame, vocabulary: set[str]) -> Resolution:
    """Turn a personal history into play counts keyed by the map's artist ids."""
    grouped = listens.group_by(["artist_mbid", "artist_name"]).agg(pl.len().alias("plays"))

    by_name = name_index(artist_names, vocabulary)
    counts: dict[str, int] = {}
    unmatched: list[tuple[str, int]] = []
    matched_by_id = matched_by_name = 0

    for row in grouped.iter_rows(named=True):
        mbid = row["artist_mbid"]
        name = str(row["artist_name"] or "")
        plays = int(row["plays"])

        if mbid and str(mbid) in vocabulary:
            counts[str(mbid)] = counts.get(str(mbid), 0) + plays
            matched_by_id += 1
            continue

        candidate = by_name.get(normalise_name(name)) if name else None
        if candidate:
            counts[candidate] = counts.get(candidate, 0) + plays
            matched_by_name += 1
            continue

        unmatched.append((name or str(mbid), plays))

    unmatched.sort(key=lambda pair: pair[1], reverse=True)
    return Resolution(counts, matched_by_id, matched_by_name, unmatched)
