/** Islands: the distinct corners of one listener's taste, discovered rather than asserted.
 *
 * The clusters come from KMeans over the full-dimensional map, with the number of clusters
 * chosen by silhouette score. Each is named after its most played artist, because the data
 * contains no genre labels and inventing one would be a lie.
 */

import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import { ErrorState, LoadingRows } from "../components/States";
import "./Islands.css";

export function Islands() {
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
  const listeners = useQuery({ queryKey: ["listeners"], queryFn: api.listeners });
  const first = listeners.data?.[0];
  const other = useQuery({
    queryKey: ["listener", first],
    queryFn: () => api.listener(first as string),
    enabled: !me.isSuccess && Boolean(first),
  });

  const listener = me.isSuccess ? me.data : other.data;

  if (other.isError && !me.isSuccess) {
    const error = other.error as ApiError;
    return (
      <div className="wrap">
        <h1>Your islands</h1>
        <ErrorState title="Nothing to cluster yet" onRetry={() => other.refetch()}>
          {error.status === 0 ? (
            <>
              The API is not running. Start it with <code>uv run python run.py demo</code>.
            </>
          ) : (
            error.message
          )}
        </ErrorState>
      </div>
    );
  }

  if (!listener) {
    return (
      <div className="wrap">
        <h1>Your islands</h1>
        <LoadingRows rows={4} label="Finding islands" />
      </div>
    );
  }

  return (
    <div className="wrap">
      <header>
        <h1>The corners of your taste</h1>
        <p className="lede">
          Your artists, grouped by where they sit on the map. Nothing here was labelled with a
          genre; these groups were found in the listening data itself.
        </p>
      </header>

      {listener.islands.length === 0 ? (
        <div className="state">
          <h3>Not enough history to find islands</h3>
          <p>Clustering needs at least a handful of recognised artists.</p>
        </div>
      ) : (
        <ol className="islands">
          {listener.islands.map((island) => (
            <li key={island.label} className="island">
              <div className="island__head">
                <h2>{island.label}</h2>
                <p className="island__meta">
                  {island.size} artists, {island.plays.toLocaleString()} plays
                </p>
              </div>
              <p className="island__isolation">
                <span
                  className="island__gauge"
                  aria-hidden="true"
                  style={{ inlineSize: `${Math.min(100, island.isolation * 400)}%` }}
                />
                {island.isolation < 0.1
                  ? "Sits at the heart of your taste."
                  : island.isolation < 0.25
                    ? "Off to one side of your taste."
                    : "A long way from everything else you play."}
              </p>
              <ul className="island__artists">
                {island.artists.slice(0, 14).map((artist) => (
                  <li key={artist.artist_mbid}>
                    {artist.name} <span>{artist.plays.toLocaleString()}</span>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
      )}

      <section className="section" aria-labelledby="edges-heading">
        <div className="section__head">
          <h2 id="edges-heading">The edges of your taste</h2>
        </div>
        <p className="note">
          The artists you play that sit furthest from your centre. These are where your taste
          already touches territory you have otherwise not explored.
        </p>
        <ul className="edges">
          {listener.edges.slice(0, 12).map((artist) => (
            <li key={artist.artist_mbid}>
              {artist.name}{" "}
              <span>
                {artist.plays.toLocaleString()} {artist.plays === 1 ? "play" : "plays"}
              </span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
