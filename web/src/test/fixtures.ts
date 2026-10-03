/** Shared test data, shaped like the API responses. */

import type { EvalReport, Listener, Recommendations, TrackRecommendation } from "../api/types";

export const loyalist: Listener = {
  user_id: "user-001",
  explorer_score: 0.0,
  personalised_lambda: 0.1,
  train_listens: 320,
  top_artists: ["Artist 0007"],
};

export const explorer: Listener = {
  user_id: "user-049",
  explorer_score: 0.44,
  personalised_lambda: 0.41,
  train_listens: 410,
  top_artists: ["Artist 0082", "Artist 0020", "Artist 0053"],
};

export const listeners: Listener[] = [loyalist, explorer];

function track(rank: number, novelty: number, isAnchor: boolean): TrackRecommendation {
  return {
    item_id: `rec-${rank}`,
    rank,
    relevance: 1 - rank * 0.1,
    novelty,
    is_anchor: isAnchor,
    artist_name: `Artist ${rank}`,
    track_name: `Track ${rank}`,
    why: isAnchor ? "You already listen to this artist." : "A step out from Artist 0082.",
  };
}

export function recommendations(lambda: number): Recommendations {
  // Higher lambda shifts novelty up, mirroring the real re-ranker.
  const items = [track(1, 0.2 + lambda * 0.5, true), track(2, 0.3 + lambda * 0.5, false)];
  return {
    user_id: explorer.user_id,
    explorer_score: explorer.explorer_score,
    personalised_lambda: explorer.personalised_lambda,
    lambda_used: lambda,
    k: items.length,
    anchor_count: items.filter((item) => item.is_anchor).length,
    items,
  };
}

export const report: EvalReport = {
  schema_version: 1,
  run_id: "testrun1",
  generated_at: "2026-10-03T04:00:00+00:00",
  split: "val",
  users_evaluated: 50,
  cold_users: 0,
  segment_sizes: { loyalists: 17, middle: 16, explorers: 17 },
  fixed_lambda: 0.2,
  k_values: [10, 20],
  config_hash: "abc123",
  git_commit: null,
  models: {
    popularity: { "ndcg@20": 0.0352, "discovery_recall@20": 0.0596, "novelty@20": 1.18 },
    als: { "ndcg@20": 0.0512, "discovery_recall@20": 0.1186, "novelty@20": 2.24 },
    ours: {
      "ndcg@20": 0.0508,
      "discovery_recall@20": 0.0793,
      "novelty@20": 2.84,
      "familiarity_anchor_rate@20": 0.306,
      "catalogue_coverage@20": 0.4885,
      "segments@20": {
        loyalists: { users: 17, "ndcg@20": 0.045, "discovery_recall@20": 0.33 },
        middle: { users: 16, "ndcg@20": 0.058, "discovery_recall@20": 0.077 },
        explorers: { users: 17, "ndcg@20": 0.038, "discovery_recall@20": 0.117 },
        cold: { users: 0 },
      },
    },
  },
  lambda_sweep: [
    { lambda: 0.0, "ndcg@20": 0.0463, "novelty@20": 2.01, "discovery_recall@20": 0.1586 },
    { lambda: 0.5, "ndcg@20": 0.0395, "novelty@20": 3.07, "discovery_recall@20": 0.1296 },
    { lambda: 1.0, "ndcg@20": 0.0289, "novelty@20": 4.24, "discovery_recall@20": 0.037 },
  ],
  confidence_intervals: null,
};
