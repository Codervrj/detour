/** Shared test data, shaped like the API responses. */

import type { ArtistMap, ArtistPoint, ListenerMap, MapReport } from "../api/types";

function artist(name: string, plays: number, x: number, y: number): ArtistPoint {
  return { artist_mbid: `mb-${name.toLowerCase().replace(/\s+/g, "-")}`, name, plays, x, y };
}

export const background: ArtistMap = {
  artists: [artist("Queen", 900, 0.3, 0.3), artist("Pixies", 400, 0.7, 0.6)],
  total_artists: 21951,
  showing: 2,
};

export const listenerMap: ListenerMap = {
  listener: "u1",
  coverage: {
    matched_artists: 638,
    unmatched_artists: 12,
    artist_coverage: 0.9815,
    play_coverage: 0.996,
    biggest_misses: ["Some Obscure Band"],
  },
  total_plays: 10788,
  artists: [
    artist("Pink Floyd", 820, 0.31, 0.29),
    artist("David Bowie", 540, 0.35, 0.33),
    artist("The Beatles", 410, 0.4, 0.45),
  ],
  islands: [
    {
      label: "Pink Floyd",
      size: 328,
      plays: 7415,
      isolation: 0.05,
      artists: [artist("Pink Floyd", 820, 0.31, 0.29), artist("Queen", 300, 0.3, 0.3)],
    },
    {
      label: "The Beatles",
      size: 170,
      plays: 2599,
      isolation: 0.23,
      artists: [artist("The Beatles", 410, 0.4, 0.45)],
    },
  ],
  edges: [artist("Pino Donaggio", 3, 0.85, 0.12)],
  frontier: [
    { artist_mbid: "mb-bee-gees", name: "Bee Gees", similarity: 0.702, x: 0.33, y: 0.31 },
    { artist_mbid: "mb-the-cars", name: "The Cars", similarity: 0.697, x: 0.36, y: 0.34 },
  ],
};

export const report: MapReport = {
  schema_version: 2,
  report_kind: "map",
  run_id: "testrun1",
  generated_at: "2026-10-03T06:00:00+00:00",
  split: "val",
  listeners_with_adoptions: 1694,
  vocabulary: 21951,
  k_values: [10, 20],
  config_hash: "abc123",
  git_commit: null,
  models: {
    random: { adoption_percentile: 0.4965, matched_percentile: 0.4955, median_rank: 10722, "hit@10": 0.0005, "hit@20": 0.0009 },
    popularity: { adoption_percentile: 0.5115, matched_percentile: 1.0, median_rank: 11138, "hit@10": 0.0004, "hit@20": 0.0007 },
    als: { adoption_percentile: 0.3935, matched_percentile: 0.3775, median_rank: 8042, "hit@10": 0.0183, "hit@20": 0.0291 },
    item2vec: { adoption_percentile: 0.2891, matched_percentile: 0.2285, median_rank: 5533, "hit@10": 0.0039, "hit@20": 0.0079 },
    ppmi_svd: { adoption_percentile: 0.1158, matched_percentile: 0.1876, median_rank: 1658, "hit@10": 0.0286, "hit@20": 0.0464 },
  },
  confidence_intervals: null,
};
