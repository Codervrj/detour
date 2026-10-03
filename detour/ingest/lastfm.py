"""Import your own listening history from Last.fm.

This is what makes the project about you rather than about strangers. The model is trained
on MLHD+, which is itself Last.fm history, so your scrobbles arrive in the same shape the
model already understands.

Only a free API key is needed. Reading public scrobbles requires no secret and no OAuth:
https://www.last.fm/api/account/create

    LASTFM_API_KEY=... uv run python -m detour.ingest.lastfm --user yourname

Last.fm returns an artist MBID for many but not all scrobbles. Rows without one keep the
artist name, and `detour.ingest.resolve` matches those to the trained vocabulary by name.
Coverage is reported rather than quietly dropped: you should know how much of your history
the model could actually read.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from detour.config import write_sidecar

API_ROOT = "https://ws.audioscrobbler.com/2.0/"
PAGE_SIZE = 200
USER_AGENT = "detour/0.1 (personal listening analysis)"

# Last.fm asks for no more than 5 requests a second averaged over 5 minutes.
REQUEST_PAUSE_SECONDS = 0.25
MAX_RETRIES = 3


class LastfmError(RuntimeError):
    """A Last.fm API call failed in a way worth showing the person."""


def api_key(explicit: str | None = None) -> str:
    """The API key, from the argument or the environment."""
    key = explicit or os.environ.get("LASTFM_API_KEY", "")
    if not key:
        raise SystemExit(
            "No Last.fm API key.\n"
            "  1. Get a free one: https://www.last.fm/api/account/create\n"
            "  2. Put it in .env as LASTFM_API_KEY=...  (see .env.example)\n"
            "  3. Or pass --api-key"
        )
    return key


def fetch_page(user: str, key: str, page: int) -> dict[str, Any]:
    """One page of recent tracks, with a short retry on transient failures."""
    query = urllib.parse.urlencode(
        {
            "method": "user.getrecenttracks",
            "user": user,
            "api_key": key,
            "format": "json",
            "limit": PAGE_SIZE,
            "page": page,
        }
    )
    request = urllib.request.Request(f"{API_ROOT}?{query}", headers={"User-Agent": USER_AGENT})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                payload: dict[str, Any] = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt == MAX_RETRIES:
                raise LastfmError(f"Could not reach Last.fm: {error}") from error
            time.sleep(attempt * 2)
            continue

        if "error" in payload:
            message = payload.get("message", "unknown error")
            if payload["error"] == 6:
                raise LastfmError(f"No Last.fm user '{user}'. Check the spelling.")
            if payload["error"] == 10:
                raise LastfmError("Last.fm rejected the API key. Check LASTFM_API_KEY.")
            raise LastfmError(f"Last.fm said: {message}")
        return payload

    raise LastfmError("Last.fm did not respond.")


def parse_track(entry: dict[str, Any]) -> dict[str, Any] | None:
    """One scrobble, or None when it is the currently-playing track with no timestamp."""
    date = entry.get("date")
    if not date or "uts" not in date:
        return None  # "now playing" has no timestamp and is not history yet

    artist = entry.get("artist") or {}
    album = entry.get("album") or {}
    return {
        "listened_at": datetime.fromtimestamp(int(date["uts"]), tz=UTC),
        "artist_mbid": (artist.get("mbid") or "") or None,
        "artist_name": artist.get("#text") or artist.get("name") or "",
        "recording_mbid": (entry.get("mbid") or "") or None,
        "track_name": entry.get("name") or "",
        "release_mbid": (album.get("mbid") or "") or None,
    }


def scrobbles(user: str, key: str, max_pages: int | None = None) -> Iterator[dict[str, Any]]:
    """Every scrobble, newest first, paging until the history runs out."""
    first = fetch_page(user, key, 1)
    attributes = first.get("recenttracks", {}).get("@attr", {})
    total_pages = int(attributes.get("totalPages", 1) or 1)
    total = int(attributes.get("total", 0) or 0)
    pages = min(total_pages, max_pages) if max_pages else total_pages
    print(f"{user}: {total:,} scrobbles across {total_pages:,} pages, fetching {pages:,}")

    for page in range(1, pages + 1):
        payload = first if page == 1 else fetch_page(user, key, page)
        for entry in payload.get("recenttracks", {}).get("track", []):
            if row := parse_track(entry):
                yield row
        if page % 25 == 0:
            print(f"  page {page:,} of {pages:,}")
        if page < pages:
            time.sleep(REQUEST_PAUSE_SECONDS)


def main(
    user: str,
    key: str | None = None,
    max_pages: int | None = None,
    out_path: str = "data/personal/listens.parquet",
) -> None:
    """Write your scrobbles to Parquet, and report how many carry a MusicBrainz id."""
    rows = list(scrobbles(user, api_key(key), max_pages))
    if not rows:
        raise SystemExit(f"No scrobbles found for '{user}'. Is the profile public?")

    frame = pl.DataFrame(rows).sort("listened_at")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)

    with_mbid = frame.filter(pl.col("artist_mbid").is_not_null()).height
    write_sidecar(
        target,
        {
            "stage": "lastfm_import",
            "user": user,
            "rows": frame.height,
            "artists_by_name": frame["artist_name"].n_unique(),
            "rows_with_artist_mbid": with_mbid,
            "mbid_coverage": round(with_mbid / frame.height, 4),
            "date_min": frame["listened_at"].min(),
            "date_max": frame["listened_at"].max(),
        },
    )
    print(
        f"\n{frame.height:,} scrobbles, {frame['artist_name'].n_unique():,} artists, "
        f"{with_mbid / frame.height:.0%} carry a MusicBrainz id -> {target}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import your Last.fm listening history.")
    parser.add_argument("--user", required=True, help="your Last.fm username")
    parser.add_argument("--api-key", default=None, help="defaults to $LASTFM_API_KEY")
    parser.add_argument("--max-pages", type=int, default=None, help="limit for a quick test")
    parser.add_argument("--out", default="data/personal/listens.parquet")
    args = parser.parse_args()
    main(args.user, args.api_key, args.max_pages, args.out)
