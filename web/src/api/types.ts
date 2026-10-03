/** Response shapes from the Detour API. Mirrors api/schemas.py. */

export interface Listener {
  user_id: string;
  explorer_score: number;
  personalised_lambda: number;
  train_listens: number;
  top_artists: string[];
}

export interface TrackRecommendation {
  item_id: string;
  rank: number;
  relevance: number;
  novelty: number;
  is_anchor: boolean;
  artist_name: string;
  track_name: string;
  why: string;
}

export interface Recommendations {
  user_id: string;
  explorer_score: number;
  personalised_lambda: number;
  lambda_used: number;
  k: number;
  anchor_count: number;
  items: TrackRecommendation[];
}

export interface FrontierPoint {
  lambda: number;
  [metric: string]: number | null;
}

export interface ModelMetrics {
  [metric: string]: number | null | Record<string, unknown>;
}

export interface EvalReport {
  schema_version: number;
  run_id: string;
  generated_at: string;
  split: string;
  users_evaluated: number;
  cold_users: number;
  segment_sizes: Record<string, number>;
  fixed_lambda: number;
  k_values: number[];
  config_hash: string;
  git_commit: string | null;
  models: Record<string, ModelMetrics>;
  lambda_sweep: FrontierPoint[];
  confidence_intervals: unknown;
}
