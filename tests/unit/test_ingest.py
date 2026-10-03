"""Ingest parsing and the resumable stream. No network: everything here is local."""

from __future__ import annotations

import io
from datetime import UTC, datetime

import pytest

from detour.ingest.artist_meta import parse, top_tags
from detour.ingest.lastfm import parse_track
from detour.ingest.mlhd_loader import make_window, parse_user
from detour.ingest.resumable import ResumableHTTPStream

# --- MLHD window and parsing -----------------------------------------------------------


def test_window_covers_twelve_whole_months() -> None:
    window = make_window("2011-01", 12)
    assert window.start == datetime(2011, 1, 1, tzinfo=UTC)
    assert window.end == datetime(2012, 1, 1, tzinfo=UTC)


def test_window_rolls_over_a_year_boundary() -> None:
    window = make_window("2011-07", 12)
    assert window.end == datetime(2012, 7, 1, tzinfo=UTC)


def line(timestamp: int, artist: str = "a-mbid") -> str:
    return f"{timestamp}\t{artist}\trelease-mbid\trecording-mbid"


def test_parse_user_keeps_only_listens_inside_the_window() -> None:
    window = make_window("2011-01", 12)
    inside = int(datetime(2011, 6, 1, tzinfo=UTC).timestamp())
    before = int(datetime(2010, 6, 1, tzinfo=UTC).timestamp())
    after = int(datetime(2013, 6, 1, tzinfo=UTC).timestamp())

    raw = "\n".join([line(before), line(inside), line(after)]).encode()
    rows = parse_user(raw, window)

    assert len(rows) == 1
    assert rows[0][0] == inside


def test_parse_user_is_inclusive_of_the_start_and_exclusive_of_the_end() -> None:
    window = make_window("2011-01", 12)
    raw = "\n".join([line(window.start_ts), line(window.end_ts)]).encode()
    rows = parse_user(raw, window)
    assert [row[0] for row in rows] == [window.start_ts]


def test_parse_user_skips_rows_with_no_artist_id() -> None:
    # The artist id is what the map is built from, so a row without one is unusable.
    window = make_window("2011-01", 12)
    inside = int(datetime(2011, 6, 1, tzinfo=UTC).timestamp())
    raw = "\n".join([line(inside, artist=""), line(inside)]).encode()
    assert len(parse_user(raw, window)) == 1


def test_parse_user_ignores_malformed_lines() -> None:
    window = make_window("2011-01", 12)
    inside = int(datetime(2011, 6, 1, tzinfo=UTC).timestamp())
    raw = b"\n".join([b"not a line", b"also\tbad", line(inside).encode(), b""])
    assert len(parse_user(raw, window)) == 1


# --- artist metadata -------------------------------------------------------------------


def test_top_tags_ranks_by_count_not_order() -> None:
    record = {
        "tag_data": {
            "artist": [
                {"tag": "best band ever", "count": 1},
                {"tag": "post-punk", "count": 300},
                {"tag": "new wave", "count": 50},
            ]
        }
    }
    assert top_tags(record) == ["post-punk", "new wave", "best band ever"]


def test_top_tags_is_empty_when_there_are_none() -> None:
    assert top_tags({}) == []
    assert top_tags({"tag_data": {}}) == []


def test_parse_artist_record() -> None:
    row = parse(
        {
            "artist_mbid": "abc",
            "artist_data": {"name": "Portishead", "area": "United Kingdom", "type": "Group"},
        }
    )
    assert row is not None
    assert row["artist_name"] == "Portishead"
    assert row["area"] == "United Kingdom"


def test_parse_artist_record_without_a_name_is_dropped() -> None:
    assert parse({"artist_mbid": "abc", "artist_data": {}}) is None
    assert parse({"artist_data": {"name": "x"}}) is None


# --- Last.fm ---------------------------------------------------------------------------


def test_parse_scrobble_with_mbids() -> None:
    row = parse_track(
        {
            "date": {"uts": "1300000000"},
            "artist": {"mbid": "artist-id", "#text": "Burial"},
            "album": {"mbid": "release-id"},
            "mbid": "recording-id",
            "name": "Archangel",
        }
    )
    assert row is not None
    assert row["artist_name"] == "Burial"
    assert row["artist_mbid"] == "artist-id"
    assert row["listened_at"] == datetime.fromtimestamp(1300000000, tz=UTC)


def test_parse_scrobble_without_mbids_keeps_the_name() -> None:
    # Last.fm often omits ids; the name still resolves later, so the row is kept.
    row = parse_track(
        {"date": {"uts": "1300000000"}, "artist": {"mbid": "", "#text": "Burial"}, "name": "X"}
    )
    assert row is not None
    assert row["artist_mbid"] is None
    assert row["artist_name"] == "Burial"


def test_now_playing_track_is_not_history() -> None:
    # The currently-playing track has no timestamp and must not become a listen.
    assert parse_track({"artist": {"#text": "Burial"}, "name": "Archangel"}) is None


# --- resumable stream ------------------------------------------------------------------


class FlakyResponse:
    """A fake HTTP response that dies part way through, like the real mirror did."""

    def __init__(self, payload: bytes, die_after: int) -> None:
        self.payload = payload
        self.die_after = die_after
        self.served = 0
        self.headers = {"Content-Length": str(len(payload))}

    def readinto(self, buffer: memoryview) -> int:
        if self.served >= self.die_after:
            raise ConnectionError("connection reset by peer")
        count = min(len(buffer), self.die_after - self.served, len(self.payload) - self.served)
        if count <= 0:
            return 0
        chunk = self.payload[self.served : self.served + count]
        buffer[: len(chunk)] = chunk
        self.served += len(chunk)
        return len(chunk)

    def close(self) -> None:
        return None


@pytest.fixture
def payload() -> bytes:
    return bytes(range(256)) * 64  # 16 KiB of known bytes


def test_resumable_stream_reconnects_and_loses_no_bytes(
    payload: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[int] = []

    def fake_connect(self: ResumableHTTPStream) -> None:
        opened.append(self.pos)
        remaining = payload[self.pos :]
        self._response = FlakyResponse(remaining, die_after=1000)
        if self.total is None:
            self.total = len(payload)

    monkeypatch.setattr(ResumableHTTPStream, "_connect", fake_connect)
    monkeypatch.setattr("detour.ingest.resumable.time.sleep", lambda _: None)

    stream = ResumableHTTPStream("http://example.invalid/x", max_retries=100)
    got = io.BufferedReader(stream, buffer_size=4096).read()

    assert got == payload
    assert stream.reconnects > 0
    assert opened[0] == 0 and opened[1] == 1000  # resumed from exactly where it stopped


def test_resumable_stream_gives_up_rather_than_looping_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fails(self: ResumableHTTPStream) -> None:
        self._response = FlakyResponse(b"", die_after=0)
        self.total = 10

    monkeypatch.setattr(ResumableHTTPStream, "_connect", always_fails)
    monkeypatch.setattr("detour.ingest.resumable.time.sleep", lambda _: None)

    stream = ResumableHTTPStream("http://example.invalid/x", max_retries=3)
    with pytest.raises(OSError, match="Gave up after 3 reconnects"):
        stream.readinto(memoryview(bytearray(16)))
