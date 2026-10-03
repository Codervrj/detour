/** Thin fetch wrapper. The API serves pre-computed artefacts, so every call is a GET. */

import type { EvalReport, Listener, Recommendations } from "./types";

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
  listeners: () => get<Listener[]>("/listeners"),
  recommendations: (userId: string, lambda: number, k = 20) =>
    get<Recommendations>(
      `/recommendations/${encodeURIComponent(userId)}?lambda=${lambda}&k=${k}`,
    ),
  latestReport: () => get<{ report: EvalReport }>("/report/latest").then((b) => b.report),
};
