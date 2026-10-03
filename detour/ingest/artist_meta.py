"""Artist names and tags from the ListenBrainz metadata dump.

MLHD gives listening history as MusicBrainz artist ids and nothing else, so without this
step every artist on the map would be a UUID. This reads `artists_cache.jsonl` out of the
ListenBrainz sample dump and writes a lookup table.

The sample dump is about 235 MB compressed and the JSONL inside it is 200 MB, so it is
streamed rather than loaded.

    uv run python -m detour.ingest.artist_meta
"""

from __future__ import annotations

import argparse
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import polars as pl
import zstandard

from detour.config import write_sidecar

SAMPLE_DUMP = "data/raw/lb-sample.tar.zst"
MEMBER_SUFFIX = "artists_cache.jsonl"
MAX_TAGS = 5


def stream_jsonl(dump_path: Path, suffix: str) -> Iterator[dict[str, Any]]:
    """Yield each JSON record from a named member inside the zstd tar, without unpacking it."""
    decompressor = zstandard.ZstdDecompressor()
    with (
        dump_path.open("rb") as handle,
        decompressor.stream_reader(handle) as reader,
        tarfile.open(fileobj=reader, mode="r|") as tar,
    ):
        for member in tar:
            if not (member.isfile() and member.name.endswith(suffix)):
                continue
            stream = tar.extractfile(member)
            if stream is None:
                continue
            for line in stream:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            return


def top_tags(record: dict[str, Any], limit: int = MAX_TAGS) -> list[str]:
    """The most-used tags for an artist, strongest first.

    Tags are folksonomy, so counts matter: one person tagging a band "best band ever"
    should not outweigh three hundred people tagging it "post-punk".
    """
    tag_data = record.get("tag_data") or {}
    entries = tag_data.get("artist") or []
    if not isinstance(entries, list):
        return []
    ranked = sorted(
        (entry for entry in entries if isinstance(entry, dict) and entry.get("tag")),
        key=lambda entry: int(entry.get("count", 0) or 0),
        reverse=True,
    )
    return [str(entry["tag"]) for entry in ranked[:limit]]


def parse(record: dict[str, Any]) -> dict[str, Any] | None:
    """Flatten one cache record into a row, or None when it carries no usable name."""
    mbid = record.get("artist_mbid")
    data = record.get("artist_data") or {}
    name = data.get("name")
    if not mbid or not name:
        return None
    return {
        "artist_mbid": str(mbid),
        "artist_name": str(name),
        "area": data.get("area"),
        "artist_type": data.get("type"),
        "begin_year": data.get("begin_year"),
        "tags": top_tags(record),
    }


def main(
    dump_path: str = SAMPLE_DUMP,
    out_path: str = "data/processed/artists.parquet",
) -> None:
    """Write the artist lookup table used everywhere a name or tag is shown."""
    source = Path(dump_path)
    if not source.exists():
        raise SystemExit(
            f"{dump_path} not found. Download it with:\n"
            "  uv run python -m detour.ingest.fetch --sample"
        )

    rows = [row for record in stream_jsonl(source, MEMBER_SUFFIX) if (row := parse(record))]
    frame = pl.DataFrame(rows).unique(subset=["artist_mbid"], keep="first")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)

    tagged = frame.filter(pl.col("tags").list.len() > 0).height
    write_sidecar(
        target,
        {
            "stage": "artist_meta",
            "source": dump_path,
            "artists": frame.height,
            "with_tags": tagged,
            "with_area": frame.filter(pl.col("area").is_not_null()).height,
        },
    )
    print(f"artists: {frame.height:,} named, {tagged:,} with tags")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artist names and tags from the LB dump.")
    parser.add_argument("--dump", default=SAMPLE_DUMP)
    parser.add_argument("--out", default="data/processed/artists.parquet")
    args = parser.parse_args()
    main(args.dump, args.out)
