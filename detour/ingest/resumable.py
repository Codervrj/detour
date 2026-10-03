"""An HTTP stream that survives the connection being cut.

Long transfers from the MetaBrainz mirror get dropped part way through on this network:
collecting listeners died after about 2 GB with `tarfile.ReadError: unexpected end of
data`, and curl exited 56 on a separate download. Retrying from scratch is not viable when
the archive is 15 GB.

This reader tracks its absolute byte position and, when the socket dies, reconnects with a
`Range` header and carries on from exactly where it stopped. Anything reading it, including
`tarfile`, sees one uninterrupted file.
"""

from __future__ import annotations

import http.client
import io
import time
import urllib.error
import urllib.request

USER_AGENT = "detour/0.1 (research)"
MAX_RETRIES = 12
TIMEOUT_SECONDS = 120

# Errors that mean "the connection died", as opposed to "the server said no".
TRANSIENT = (
    urllib.error.URLError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    ConnectionError,
    TimeoutError,
    OSError,
)


class ResumableHTTPStream(io.RawIOBase):
    """A read-only byte stream over HTTP that reconnects on failure.

    The server must honour range requests; the MetaBrainz mirror does.
    """

    def __init__(self, url: str, max_retries: int = MAX_RETRIES) -> None:
        self.url = url
        self.max_retries = max_retries
        self.pos = 0
        self.total: int | None = None
        self.reconnects = 0
        self._response: http.client.HTTPResponse | None = None
        self._connect()

    def _connect(self) -> None:
        """Open the stream at the current byte position."""
        headers = {"User-Agent": USER_AGENT}
        if self.pos:
            headers["Range"] = f"bytes={self.pos}-"
        request = urllib.request.Request(self.url, headers=headers)
        self._response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)  # noqa: S310

        if self.total is None:
            length = self._response.headers.get("Content-Length")
            if length and length.isdigit():
                self.total = int(length) + self.pos

    def _reconnect(self, attempt: int) -> None:
        """Drop the dead response and reopen from `self.pos`."""
        self.reconnects += 1
        try:
            if self._response is not None:
                self._response.close()
        except Exception:  # noqa: BLE001 - the socket is already broken
            pass
        time.sleep(min(2**attempt, 30))
        self._connect()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        """Fill `buffer`, reconnecting as needed. Returns 0 only at genuine end of file."""
        for attempt in range(self.max_retries):
            if self._response is None:
                self._reconnect(attempt)
                continue
            try:
                count = self._response.readinto(buffer)
            except TRANSIENT:
                self._reconnect(attempt)
                continue

            if count:
                self.pos += count
                return count

            # Zero bytes is EOF only if we actually reached the end; otherwise the
            # connection was closed early and we resume from where we stopped.
            if self.total is not None and self.pos < self.total:
                self._reconnect(attempt)
                continue
            return 0

        raise OSError(
            f"Gave up after {self.max_retries} reconnects at byte {self.pos:,} of {self.url}"
        )

    def close(self) -> None:
        try:
            if self._response is not None:
                self._response.close()
        finally:
            super().close()


def open_stream(url: str, buffer_size: int = 1 << 20) -> io.BufferedReader:
    """A buffered, reconnecting reader, ready to hand to `tarfile.open(mode='r|')`."""
    return io.BufferedReader(ResumableHTTPStream(url), buffer_size=buffer_size)
