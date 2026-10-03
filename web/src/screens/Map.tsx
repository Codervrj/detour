/** Your map: where one listener sits in the world of recorded music.
 *
 * The map is the page. Everything else on it exists to explain what you are looking at,
 * and to give a non-visual route to the same information.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import type { ArtistPoint } from "../api/types";
import { ArtistMap } from "../components/ArtistMap";
import { ErrorState, LoadingRows } from "../components/States";
import "./Map.css";

export function MapScreen() {
  const [selected, setSelected] = useState<ArtistPoint | null>(null);

  const background = useQuery({ queryKey: ["map"], queryFn: () => api.map(4000) });
  const listeners = useQuery({ queryKey: ["listeners"], queryFn: api.listeners });

  const [chosen, setChosen] = useState<string | null>(null);
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
  const hasPersonal = me.isSuccess;
  const listenerId = chosen ?? listeners.data?.[0] ?? null;

  const other = useQuery({
    queryKey: ["listener", listenerId],
    queryFn: () => api.listener(listenerId as string),
    enabled: !hasPersonal && Boolean(listenerId),
  });

  const neighbours = useQuery({
    queryKey: ["artist", selected?.artist_mbid],
    queryFn: () => api.artist(selected?.artist_mbid as string),
    enabled: Boolean(selected),
  });

  const listener = hasPersonal ? me.data : other.data;
  const loading = background.isPending || (hasPersonal ? me.isPending : other.isPending);

  if (background.isError) {
    const error = background.error as ApiError;
    return (
      <div className="wrap">
        <h1>Your map</h1>
        <ErrorState title="No map to show yet" onRetry={() => background.refetch()}>
          {error.status === 0 ? (
            <>
              The API is not running. Start everything with <code>uv run python run.py demo</code>.
            </>
          ) : (
            <>
              {error.message} Build the map with <code>uv run python run.py pipeline</code>.
            </>
          )}
        </ErrorState>
      </div>
    );
  }

  return (
    <div className="wrap">
      <header>
        <h1>Where your taste sits</h1>
        <p className="lede">
          Every artist below was placed by how people listen, not by genre labels. Artists that
          share listeners end up near each other. Your own artists are lit up.
        </p>
      </header>

      {!hasPersonal && listeners.data && listeners.data.length > 0 && (
        <section className="section">
          <p className="notice">
            You have not imported your own history, so this is a real listener from the held-out
            data. To see yourself here, run{" "}
            <code>uv run python run.py lastfm --user YOURNAME</code>.
          </p>
          <label className="picker-label" htmlFor="listener-select">
            Listener
          </label>
          <select
            id="listener-select"
            className="select"
            value={listenerId ?? ""}
            onChange={(event) => {
              setChosen(event.target.value);
              setSelected(null);
            }}
          >
            {listeners.data.map((id) => (
              <option key={id} value={id}>
                {id.slice(0, 8)}
              </option>
            ))}
          </select>
        </section>
      )}

      <section className="section" aria-labelledby="map-heading">
        <h2 className="visually-hidden" id="map-heading">
          The map
        </h2>
        {loading ? (
          <LoadingRows rows={5} label="Loading the map" />
        ) : listener && background.data ? (
          <>
            <ArtistMap
              background={background.data.artists}
              mine={listener.artists}
              frontier={listener.frontier}
              onPick={setSelected}
              selected={selected?.artist_mbid ?? null}
            />
            <p className="map-foot">
              {background.data.showing.toLocaleString()} of{" "}
              {background.data.total_artists.toLocaleString()} artists shown.{" "}
              {listener.coverage.matched_artists.toLocaleString()} of your{" "}
              {(
                listener.coverage.matched_artists + listener.coverage.unmatched_artists
              ).toLocaleString()}{" "}
              artists are on the map, covering{" "}
              {(listener.coverage.play_coverage * 100).toFixed(0)}% of your plays.
            </p>
          </>
        ) : null}
      </section>

      {selected && (
        <section className="section" aria-labelledby="near-heading">
          <div className="section__head">
            <h2 id="near-heading">Next to {selected.name}</h2>
          </div>
          {neighbours.isPending ? (
            <LoadingRows rows={3} label="Loading neighbours" />
          ) : neighbours.data ? (
            <ol className="near">
              {neighbours.data.neighbours.map((n) => (
                <li key={n.artist_mbid}>
                  <span className="near__name">{n.name}</span>
                  <span className="near__bar" aria-hidden="true">
                    <span style={{ inlineSize: `${Math.max(2, n.similarity * 100)}%` }} />
                  </span>
                  <span className="near__score">{n.similarity.toFixed(2)}</span>
                </li>
              ))}
            </ol>
          ) : null}
        </section>
      )}

      {listener && (
        <section className="section" aria-labelledby="frontier-heading">
          <div className="section__head">
            <h2 id="frontier-heading">Closest to you, and you have never played them</h2>
          </div>
          <p className="note">
            Ranked by distance from your position on the map. This is the exact ranking the
            evaluation scored, so these are not a different list from the one that was measured.
          </p>
          <ol className="near">
            {listener.frontier.slice(0, 12).map((n) => (
              <li key={n.artist_mbid}>
                <span className="near__name">{n.name}</span>
                <span className="near__bar" aria-hidden="true">
                  <span
                    className="is-unknown"
                    style={{ inlineSize: `${Math.max(2, n.similarity * 100)}%` }}
                  />
                </span>
                <span className="near__score">{n.similarity.toFixed(2)}</span>
              </li>
            ))}
          </ol>
        </section>
      )}
    </div>
  );
}
