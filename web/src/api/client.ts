/** Thin fetch wrapper. The API serves a pre-computed map, so every call is a GET. */

import type { ArtistMap, ArtistNeighbours, ListenerMap, MapReport } from "./types";

const BASE = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

/** An API failure carrying the status, so screens can tell 404 from 503. */
export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function get<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`);
  } catch {
    throw new ApiError("Cannot reach the Detour API.", 0);
  }

  if (!response.ok) {
    const detail = await response
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined);
    throw new ApiError(detail ?? `Request failed (${response.status}).`, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  map: (limit = 3000) => get<ArtistMap>(`/map?limit=${limit}`),
  listeners: () => get<string[]>("/listeners"),
  listener: (id: string) => get<ListenerMap>(`/listener/${encodeURIComponent(id)}`),
  me: () => get<ListenerMap>("/me"),
  artist: (mbid: string, k = 12) =>
    get<ArtistNeighbours>(`/artist/${encodeURIComponent(mbid)}?k=${k}`),
  latestReport: () => get<{ report: MapReport }>("/report/latest").then((b) => b.report),
};
