"""Generate tiny synthetic listens so the project runs with no network or keys.

Around 50 users across 12 months, spanning the explorer spectrum: the first few
replay a single artist (their explorer score must come out 0), the rest adopt new
artists at an increasing rate.

Artist popularity follows a Zipf curve so novelty and popularity metrics have real
spread, and a slice of near-duplicate scrobbles is injected so the 30 second dedupe
rule has something to collapse.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

N_USERS = 50
N_ARTISTS = 300
TRACKS_PER_ARTIST = 5
N_MONTHS = 12
START = datetime(2024, 1, 1, tzinfo=UTC)

# Users below this index replay exactly one artist, so explorer score must be 0.
LOYALIST_USERS = 5

# Share of rows that get a near-duplicate copy, to exercise the dedupe rule.
DUP_WITHIN_WINDOW = 0.02  # +10 seconds, must collapse to one listen
DUP_OUTSIDE_WINDOW = 0.01  # +40 seconds, must stay two listens

# Latent taste. Each artist belongs to one topic and each user has a concentrated taste
# vector over topics, so listeners who overlap in train genuinely share taste rather than
# merely sharing the charts. Without this the only signal in the data is global popularity
# and no collaborative model can beat the popularity baseline, which makes the eval unable
# to say anything about the project's claim.
N_TOPICS = 8
TASTE_CONCENTRATION = 0.4  # Dirichlet alpha; below 1 concentrates taste on a few topics


def artist_weights() -> Any:
    """Zipf-ish popularity over artists, normalised to a probability vector."""
    ranks = np.arange(1, N_ARTISTS + 1, dtype=np.float64)
    weights = 1.0 / ranks**1.1
    return weights / weights.sum()


def artist_topics(rng: np.random.Generator) -> Any:
    """Assign every artist to one latent topic."""
    return rng.integers(0, N_TOPICS, size=N_ARTISTS)


def taste_weights(rng: np.random.Generator, weights: Any, topics: Any) -> Any:
    """One user's artist distribution: global popularity tilted by their personal taste."""
    taste = rng.dirichlet(np.full(N_TOPICS, TASTE_CONCENTRATION))
    personal = weights * taste[topics]
    total = personal.sum()
    return personal / total if total > 0 else weights


def month_start(index: int) -> datetime:
    """Start of the calendar month `index` months after START."""
    year = START.year + (START.month - 1 + index) // 12
    month = (START.month - 1 + index) % 12 + 1
    return datetime(year, month, 1, tzinfo=UTC)


def make_row(user_id: str, artist: int, track: int, listened_at: datetime) -> dict[str, Any]:
    """One listen event, shaped like a ListenBrainz listen with MBIDs present."""
    return {
        "user_id": user_id,
        "artist_mbid": f"artist-{artist:04d}",
        "artist_name": f"Artist {artist:04d}",
        "recording_mbid": f"rec-{artist:04d}-{track}",
        "track_name": f"Track {track} by Artist {artist:04d}",
        "listened_at": listened_at,
    }


def adventurousness(user_index: int) -> float:
    """Spread users evenly across the explorer spectrum, with hard loyalists at the start."""
    if user_index < LOYALIST_USERS:
        return 0.0
    return user_index / (N_USERS - 1)


def user_listens(
    user_index: int, rng: np.random.Generator, weights: Any, topics: Any
) -> list[dict[str, Any]]:
    """Twelve months of listens for one user, adopting new artists as their taste allows."""
    user_id = f"user-{user_index:03d}"
    appetite = adventurousness(user_index)
    personal = taste_weights(rng, weights, topics)

    seed_count = 1 if appetite == 0.0 else int(rng.integers(2, 5))
    known = [int(a) for a in rng.choice(N_ARTISTS, size=seed_count, replace=False, p=personal)]

    rows: list[dict[str, Any]] = []
    for month in range(N_MONTHS):
        lo = month_start(month)
        span = int((month_start(month + 1) - lo).total_seconds())
        for _ in range(int(rng.integers(15, 45))):
            artist = pick_artist(appetite, known, rng, personal)
            track = int(rng.integers(TRACKS_PER_ARTIST))
            listened_at = lo + timedelta(seconds=int(rng.integers(span)))
            rows.append(make_row(user_id, artist, track, listened_at))
    return rows


def pick_artist(appetite: float, known: list[int], rng: np.random.Generator, personal: Any) -> int:
    """Either adopt an artist this user has never heard, or replay one they know.

    Adoption samples from the user's own taste-tilted distribution, not the global charts,
    so what a listener discovers next is predictable from who they resemble.
    """
    if appetite > 0.0 and rng.random() < appetite * 0.35:
        unseen = [a for a in range(N_ARTISTS) if a not in set(known)]
        if unseen:
            slice_ = personal[unseen]
            total = slice_.sum()
            probabilities = slice_ / total if total > 0 else None
            adopted = int(rng.choice(unseen, p=probabilities))
            known.append(adopted)
            return adopted
    return int(rng.choice(known))


def add_near_duplicates(
    rows: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    """Copy a slice of rows a few seconds later, on both sides of the 30 second window."""
    extra: list[dict[str, Any]] = []
    for row in rows:
        draw = rng.random()
        if draw < DUP_WITHIN_WINDOW:
            offset = 10
        elif draw < DUP_WITHIN_WINDOW + DUP_OUTSIDE_WINDOW:
            offset = 40
        else:
            continue
        extra.append({**row, "listened_at": row["listened_at"] + timedelta(seconds=offset)})
    return rows + extra


def generate(seed: int) -> pl.DataFrame:
    """Build the full fixture frame, sorted by user then time."""
    rng = np.random.default_rng(seed)
    weights = artist_weights()
    topics = artist_topics(rng)
    rows: list[dict[str, Any]] = []
    for user_index in range(N_USERS):
        rows.extend(user_listens(user_index, rng, weights, topics))
    rows = add_near_duplicates(rows, rng)
    return pl.DataFrame(rows).sort(["user_id", "listened_at"])


def sidecar_stats(frame: pl.DataFrame, seed: int) -> dict[str, Any]:
    """Row counts and date range, matching the JSON sidecar every pipeline stage writes."""
    return {
        "stage": "fixtures",
        "seed": seed,
        "rows": frame.height,
        "users": frame["user_id"].n_unique(),
        "recordings": frame["recording_mbid"].n_unique(),
        "artists": frame["artist_mbid"].n_unique(),
        "date_min": str(frame["listened_at"].min()),
        "date_max": str(frame["listened_at"].max()),
    }


def main(out_dir: str = "tests/fixtures", seed: int = 13) -> None:
    """Write a deterministic listens Parquet file; print rows and date range."""
    frame = generate(seed)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    parquet_path = target / "listens.parquet"
    frame.write_parquet(parquet_path)

    stats = sidecar_stats(frame, seed)
    (target / "listens.stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"wrote {parquet_path}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="tests/fixtures")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    main(args.out_dir, args.seed)
