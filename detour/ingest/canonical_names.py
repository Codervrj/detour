"""Artist names from the MusicBrainz canonical dump.

MLHD gives artist ids and no names, and the ListenBrainz sample cache only covers a few
thousand artists, so a map built from it would be almost entirely unlabelled UUIDs.

The canonical dump holds a name for essentially every artist, but the CSV inside it is
7.6 GB. It is therefore streamed, never unpacked, and filtered down to only the artists
that actually appear in our listens as it goes.

    uv run python -m detour.ingest.canonical_names
"""

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path

import polars as pl
import zstandard

from detour.config import write_sidecar

CANONICAL_DUMP = "data/raw/mb-canonical.tar.zst"
MEMBER = "canonical_musicbrainz_data.csv"

# A row's artist_mbids column may list several ids for a collaboration credit.
SEPARATOR = ","


def wanted_artists(listens_path: Path) -> set[str]:
    """The artist ids that appear in our listens, so the 7.6 GB scan can discard the rest."""
    frame = pl.read_parquet(listens_path, columns=["artist_mbid"])
    return set(frame["artist_mbid"].unique().to_list())


def extract(dump_path: Path, wanted: set[str]) -> dict[str, str]:
    """Stream the canonical CSV and keep the first name seen for each wanted artist.

    Rows are ordered by a MusicBrainz score, so the first credit naming an artist is the
    most canonical one available.
    """
    names: dict[str, str] = {}
    decompressor = zstandard.ZstdDecompressor()

    with (
        dump_path.open("rb") as handle,
        decompressor.stream_reader(handle) as reader,
        tarfile.open(fileobj=reader, mode="r|") as tar,
    ):
        for member in tar:
            if not (member.isfile() and member.name.endswith(MEMBER)):
                continue
            stream = tar.extractfile(member)
            if stream is None:
                continue

            # TextIOWrapper needs a seekable stream and a tar member is not one, so the
            # bytes are decoded line by line and handed straight to the CSV reader.
            lines = (raw.decode("utf-8", "replace") for raw in stream)
            for row in csv.DictReader(lines):
                ids = (row.get("artist_mbids") or "").split(SEPARATOR)
                # Only single-artist credits give an unambiguous name for one id.
                if len(ids) != 1:
                    continue
                mbid = ids[0].strip()
                if mbid in wanted and mbid not in names:
                    name = (row.get("artist_credit_name") or "").strip()
                    if name:
                        names[mbid] = name
                if len(names) == len(wanted):
                    return names
            return names
    return names


def main(
    dump_path: str = CANONICAL_DUMP,
    listens_path: str = "data/raw/listens.parquet",
    out_path: str = "data/processed/artist_names.parquet",
) -> None:
    """Write the artist id to name lookup for every artist in our listens."""
    source = Path(dump_path)
    if not source.exists():
        raise SystemExit(f"{dump_path} not found. Download the canonical dump first.")

    wanted = wanted_artists(Path(listens_path))
    print(f"looking for {len(wanted):,} artists in {source.name}")

    names = extract(source, wanted)
    frame = pl.DataFrame({"artist_mbid": list(names), "artist_name": [names[a] for a in names]})

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)

    coverage = frame.height / len(wanted) if wanted else 0.0
    write_sidecar(
        target,
        {
            "stage": "canonical_names",
            "source": dump_path,
            "artists_wanted": len(wanted),
            "artists_named": frame.height,
            "coverage": round(coverage, 4),
        },
    )
    print(f"named {frame.height:,} of {len(wanted):,} artists ({coverage:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Artist names from the canonical dump.")
    parser.add_argument("--dump", default=CANONICAL_DUMP)
    parser.add_argument("--listens", default="data/raw/listens.parquet")
    parser.add_argument("--out", default="data/processed/artist_names.parquet")
    args = parser.parse_args()
    main(args.dump, args.listens, args.out)
