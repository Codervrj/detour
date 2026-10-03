/** Response shapes from the Detour API. Mirrors api/schemas.py. */

export interface ArtistPoint {
  artist_mbid: string;
  name: string;
  plays: number;
  x: number;
  y: number;
}

export interface Neighbour {
  artist_mbid: string;
  name: string;
  similarity: number;
  x: number;
  y: number;
}

export interface Coverage {
  matched_artists: number;
  unmatched_artists: number;
  artist_coverage: number;
  play_coverage: number;
  biggest_misses: string[];
}

export interface Island {
  label: string;
  size: number;
  plays: number;
  isolation: number;
  artists: ArtistPoint[];
}

export interface ListenerMap {
  listener: string;
  coverage: Coverage;
  total_plays: number;
  artists: ArtistPoint[];
  islands: Island[];
  edges: ArtistPoint[];
  frontier: Neighbour[];
}

export interface ArtistMap {
  artists: ArtistPoint[];
  total_artists: number;
  showing: number;
}

export interface ArtistNeighbours {
  artist_mbid: string;
  name: string;
  neighbours: Neighbour[];
}

export interface ModelScores {
  [metric: string]: number | null;
}

export interface MapReport {
  schema_version: number;
  report_kind: string;
  run_id: string;
  generated_at: string;
  split: string;
  listeners_with_adoptions: number;
  vocabulary: number;
  k_values: number[];
  config_hash: string;
  git_commit: string | null;
  models: Record<string, ModelScores>;
  confidence_intervals: unknown;
}
