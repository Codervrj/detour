"""Real listening histories from MLHD+ (Music Listening Histories Dataset).

MLHD+ is 27 billion plays by 583,000 Last.fm listeners, mapped onto MusicBrainz ids and
published by MetaBrainz for research. One listener is one zstd-compressed file inside a
sharded tar, with tab-separated lines:

    timestamp <TAB> artist_mbid <TAB> release_mbid <TAB> recording_mbid

Why this and not the ListenBrainz listens dump: that dump is 229 GB and a single day of
the incremental feed is 352 MB, so a year would be roughly 128 GB. One MLHD+ shard is
15 GB and holds about a sixteenth of the listeners with their full multi-year histories.

The archive is **streamed over HTTP and abandoned once enough listeners are collected**,
so a run that wants a few thousand listeners transfers a few GB rather than all 15.

    uv run python -m detour.ingest.mlhd_loader --users 4000 --from 2011-01 --months 12
"""

from __future__ import annotations

import argparse
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import zstandard

from detour.config import write_sidecar
from detour.ingest.resumable import open_stream

SHARD_URL = (
    "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/mlhd/mlhdplus-complete-{shard}.tar"
)
COLUMNS = ["listened_at", "artist_mbid", "release_mbid", "recording_mbid"]

# A listener needs enough plays inside the window to say anything about their taste.
MIN_LISTENS_IN_WINDOW = 120


@dataclass(frozen=True)
class Window:
    """The 12 months of history being collected."""

    start: datetime
    end: datetime

    @property
    def start_ts(self) -> int:
        return int(self.start.timestamp())

    @property
    def end_ts(self) -> int:
        return int(self.end.timestamp())


def make_window(start_month: str, months: int) -> Window:
    """Build the collection window from a YYYY-MM string."""
    year, month = (int(part) for part in start_month.split("-"))
    start = datetime(year, month, 1, tzinfo=UTC)
    end_year = year + (month - 1 + months) // 12
    end_month = (month - 1 + months) % 12 + 1
    return Window(start, datetime(end_year, end_month, 1, tzinfo=UTC))


def parse_user(raw: bytes, window: Window) -> list[tuple[int, str, str, str]]:
    """Parse one listener's file, keeping only plays inside the window with an artist id."""
    rows: list[tuple[int, str, str, str]] = []
    for line in raw.decode("utf-8", "replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        stamp = parts[0]
        if not stamp.isdigit():
            continue
        seconds = int(stamp)
        if not (window.start_ts <= seconds < window.end_ts):
            continue
        artist = parts[1]
        if not artist:
            continue
        rows.append((seconds, artist, parts[2], parts[3]))
    return rows


def stream_listeners(
    url: str, window: Window, wanted: int, min_listens: int
) -> Iterator[tuple[str, list[tuple[int, str, str, str]]]]:
    """Yield (listener_id, rows) from the remote shard until `wanted` listeners are found."""
    decompressor = zstandard.ZstdDecompressor()

    found = 0
    # The connection to the mirror gets cut on long transfers, so the stream reconnects
    # and resumes rather than losing a multi-gigabyte read.
    with (
        open_stream(url) as stream,
        tarfile.open(fileobj=stream, mode="r|") as tar,
    ):
        for member in tar:
            if not (member.isfile() and member.name.endswith(".txt.zst")):
                continue
            member_file = tar.extractfile(member)
            if member_file is None:
                continue
            try:
                raw = decompressor.decompress(member_file.read(), max_output_size=512_000_000)
            except zstandard.ZstdError:
                continue

            rows = parse_user(raw, window)
            if len(rows) < min_listens:
                continue

            listener_id = Path(member.name).name.removesuffix(".txt.zst")
            yield listener_id, rows
            found += 1
            if found >= wanted:
                return


def collect(
    shard: str,
    window: Window,
    wanted: int,
    min_listens: int,
    parts_dir: Path,
    flush_every: int = 400,
) -> pl.DataFrame:
    """Collect `wanted` listeners, flushing to part files so memory stays flat.

    Holding every listener in memory until the end would mean tens of millions of rows in
    a list of frames. Parts are written as we go and scanned back at the end.
    """
    url = SHARD_URL.format(shard=shard)
    print(f"streaming {url}")
    print(f"window {window.start:%Y-%m} to {window.end:%Y-%m}, want {wanted:,} listeners")

    parts_dir.mkdir(parents=True, exist_ok=True)
    for stale in parts_dir.glob("part-*.parquet"):
        stale.unlink()

    batch: list[pl.DataFrame] = []
    part_index = 0
    total_rows = 0

    def flush() -> None:
        nonlocal batch, part_index
        if not batch:
            return
        pl.concat(batch).write_parquet(parts_dir / f"part-{part_index:04d}.parquet")
        part_index += 1
        batch = []

    for collected, (listener_id, rows) in enumerate(
        stream_listeners(url, window, wanted, min_listens), start=1
    ):
        batch.append(
            pl.DataFrame(rows, schema=COLUMNS, orient="row").with_columns(
                pl.lit(listener_id).alias("user_id")
            )
        )
        total_rows += len(rows)
        if collected % flush_every == 0:
            flush()
            print(f"  {collected:,} listeners, {total_rows:,} listens")
    flush()

    parts = sorted(parts_dir.glob("part-*.parquet"))
    if not parts:
        raise SystemExit("No listeners matched the window. Try a different --from month.")

    frame = pl.concat([pl.read_parquet(part) for part in parts])
    return frame.with_columns(
        pl.from_epoch(pl.col("listened_at"), time_unit="s").alias("listened_at")
    )


def main(
    shard: str = "0",
    start_month: str = "2011-01",
    months: int = 12,
    users: int = 4000,
    min_listens: int = MIN_LISTENS_IN_WINDOW,
    out_path: str = "data/raw/listens.parquet",
) -> None:
    """Write real listens to data/raw/ with a sidecar describing what was collected."""
    window = make_window(start_month, months)
    frame = collect(shard, window, users, min_listens, Path(out_path).parent / "parts")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)

    write_sidecar(
        target,
        {
            "stage": "mlhd_ingest",
            "source": SHARD_URL.format(shard=shard),
            "licence": "MLHD+ by MetaBrainz, research use",
            "shard": shard,
            "window_start": window.start,
            "window_end": window.end,
            "min_listens_in_window": min_listens,
            "rows": frame.height,
            "users": frame["user_id"].n_unique(),
            "artists": frame["artist_mbid"].n_unique(),
            "recordings": frame["recording_mbid"].n_unique(),
            "date_min": frame["listened_at"].min(),
            "date_max": frame["listened_at"].max(),
        },
    )
    print(
        f"\n{frame.height:,} listens, {frame['user_id'].n_unique():,} listeners, "
        f"{frame['artist_mbid'].n_unique():,} artists -> {target}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect real listens from MLHD+.")
    parser.add_argument("--shard", default="0", help="shard 0-9 or a-f")
    parser.add_argument("--from", dest="start_month", default="2011-01", help="YYYY-MM")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--users", type=int, default=4000)
    parser.add_argument("--min-listens", type=int, default=MIN_LISTENS_IN_WINDOW)
    parser.add_argument("--out", default="data/raw/listens.parquet")
    args = parser.parse_args()
    main(args.shard, args.start_month, args.months, args.users, args.min_listens, args.out)
