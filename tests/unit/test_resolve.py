"""Matching a personal history to the map's vocabulary."""

from __future__ import annotations

import polars as pl

from detour.ingest.resolve import normalise_name, resolve

VOCAB = {"mb-radiohead", "mb-beyonce", "mb-pixies"}
NAMES = pl.DataFrame(
    {
        "artist_mbid": ["mb-radiohead", "mb-beyonce", "mb-pixies", "mb-notinmap"],
        "artist_name": ["Radiohead", "Beyoncé", "The Pixies", "Someone Else"],
    }
)


def listens(rows: list[tuple[str | None, str]]) -> pl.DataFrame:
    return pl.DataFrame({"artist_mbid": [r[0] for r in rows], "artist_name": [r[1] for r in rows]})


def test_accents_and_case_are_folded() -> None:
    assert normalise_name("Beyoncé") == normalise_name("BEYONCE")
    assert normalise_name("Sigur Rós") == "sigur ros"


def test_leading_the_is_ignored() -> None:
    assert normalise_name("The Pixies") == normalise_name("Pixies")


def test_punctuation_is_ignored() -> None:
    assert normalise_name("Godspeed You! Black Emperor") == "godspeed you black emperor"


def test_matches_by_id_when_present() -> None:
    result = resolve(listens([("mb-radiohead", "Radiohead")]), NAMES, VOCAB)
    assert result.counts == {"mb-radiohead": 1}
    assert result.matched_by_id == 1
    assert result.matched_by_name == 0


def test_falls_back_to_name_when_the_id_is_missing() -> None:
    result = resolve(listens([(None, "Radiohead")]), NAMES, VOCAB)
    assert result.counts == {"mb-radiohead": 1}
    assert result.matched_by_name == 1


def test_falls_back_to_name_when_the_id_is_not_in_the_map() -> None:
    # Last.fm sometimes returns an id MusicBrainz has since merged away.
    result = resolve(listens([("mb-stale", "Radiohead")]), NAMES, VOCAB)
    assert result.counts == {"mb-radiohead": 1}
    assert result.matched_by_name == 1


def test_accented_name_resolves_to_the_map_artist() -> None:
    result = resolve(listens([(None, "beyonce")]), NAMES, VOCAB)
    assert result.counts == {"mb-beyonce": 1}


def test_unmatched_artists_are_reported_not_dropped() -> None:
    result = resolve(listens([(None, "Nobody At All"), (None, "Radiohead")]), NAMES, VOCAB)
    assert result.matched_artists == 1
    assert [name for name, _ in result.unmatched] == ["Nobody At All"]
    assert result.artist_coverage == 0.5


def test_plays_are_summed_per_artist() -> None:
    rows = [("mb-radiohead", "Radiohead")] * 3 + [(None, "Radiohead")] * 2
    result = resolve(listens(rows), NAMES, VOCAB)
    assert result.counts == {"mb-radiohead": 5}


def test_play_coverage_weights_by_plays_not_artists() -> None:
    rows = [("mb-radiohead", "Radiohead")] * 9 + [(None, "Obscure Band")]
    result = resolve(listens(rows), NAMES, VOCAB)
    assert result.artist_coverage == 0.5
    assert result.play_coverage == 0.9


def test_summary_lists_the_biggest_misses_first() -> None:
    rows = [(None, "Small Miss")] + [(None, "Big Miss")] * 5
    result = resolve(listens(rows), NAMES, VOCAB)
    assert result.summary()["biggest_misses"][0] == "Big Miss"


def test_artists_outside_the_map_are_not_matched_by_name() -> None:
    # "Someone Else" has a name entry but is not in the map, so it cannot be used.
    result = resolve(listens([(None, "Someone Else")]), NAMES, VOCAB)
    assert result.matched_artists == 0
