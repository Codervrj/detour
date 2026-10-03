/** Listen: the working demo, and the first thing anyone sees.
 *
 * Pick a listener, move one control, watch the list re-rank. Nothing here is a marketing
 * hero; the control and its effect are the page.
 */

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import type { Listener } from "../api/types";
import { DiscoveryDial } from "../components/DiscoveryDial";
import { ListenerPicker, band } from "../components/ListenerPicker";
import { TrackList } from "../components/TrackList";
import { EmptyState, ErrorState, LoadingRows } from "../components/States";

const K = 20;
const PICKER_SLOTS = 8;

/** Spread a handful of listeners evenly across the explorer range.
 *
 * The full set is fifty, and the first dozen all score 0.00, so showing them in order
 * puts eight identical entries on screen and hides the range entirely. */
function spread(rows: Listener[], keep: string | null): Listener[] {
  if (rows.length <= PICKER_SLOTS) return rows;
  const step = (rows.length - 1) / (PICKER_SLOTS - 1);
  const picked = new Map<string, Listener>();
  for (let i = 0; i < PICKER_SLOTS; i += 1) {
    const row = rows[Math.round(i * step)];
    picked.set(row.user_id, row);
  }
  const selected = rows.find((row) => row.user_id === keep);
  if (selected) picked.set(selected.user_id, selected);
  return [...picked.values()].sort((a, b) => a.explorer_score - b.explorer_score);
}

export function Listen() {
  const [selected, setSelected] = useState<string | null>(null);
  const [lambda, setLambda] = useState<number | null>(null);

  const listeners = useQuery({ queryKey: ["listeners"], queryFn: api.listeners });

  // Start on a listener who actually roams, so the dial has something to show.
  useEffect(() => {
    if (selected || !listeners.data?.length) return;
    const rows = listeners.data;
    const pick = rows[Math.floor(rows.length * 0.8)] ?? rows[rows.length - 1];
    setSelected(pick.user_id);
    setLambda(pick.personalised_lambda);
  }, [listeners.data, selected]);

  const listener = listeners.data?.find((row) => row.user_id === selected) ?? null;
  const effectiveLambda = lambda ?? listener?.personalised_lambda ?? 0.3;

  const recs = useQuery({
    queryKey: ["recommendations", selected, effectiveLambda],
    queryFn: () => api.recommendations(selected as string, effectiveLambda, K),
    enabled: Boolean(selected),
    placeholderData: (previous) => previous,
  });

  function choose(userId: string) {
    setSelected(userId);
    const next = listeners.data?.find((row) => row.user_id === userId);
    setLambda(next ? next.personalised_lambda : null);
  }

  if (listeners.isError) {
    const error = listeners.error as ApiError;
    return (
      <div className="wrap">
        <h1>Listen</h1>
        <ErrorState title="No data to show yet" onRetry={() => listeners.refetch()}>
          {error.status === 0 ? (
            <>
              The API is not running. Start everything with <code>uv run python run.py demo</code>,
              then reload this page.
            </>
          ) : (
            <>
              {error.message} Build the artefacts with <code>uv run python run.py pipeline</code>.
            </>
          )}
        </ErrorState>
      </div>
    );
  }

  return (
    <div className="wrap">
      <header>
        <h1>Find something you didn&rsquo;t know you&rsquo;d love</h1>
        <p className="lede">
          Detour learns how adventurous each listener is, then decides how far past their
          habits to push. Move the dial and watch the list rearrange.
        </p>
      </header>

      <section className="section" aria-labelledby="pick-heading">
        <div className="section__head">
          <h2 id="pick-heading">Pick a listener</h2>
        </div>
        {listeners.isLoading ? (
          <LoadingRows rows={2} label="Loading listeners" />
        ) : listeners.data?.length ? (
          <ListenerPicker
            listeners={spread(listeners.data, selected)}
            selected={selected}
            onSelect={choose}
          />
        ) : (
          <EmptyState title="No listeners in the artefacts">
            The pipeline produced no scored listeners. Run{" "}
            <code>uv run python run.py pipeline</code> and reload.
          </EmptyState>
        )}

        {listener && (
          <p className="listener-summary">
            {listener.user_id} has {listener.train_listens.toLocaleString()} listens and{" "}
            {band(listener.explorer_score)}, scoring {listener.explorer_score.toFixed(2)} on the
            explorer scale. Most played:{" "}
            {listener.top_artists.length ? listener.top_artists.join(", ") : "not enough history"}.
          </p>
        )}
      </section>

      {listener && (
        <section className="section" aria-labelledby="dial-heading">
          <h2 className="visually-hidden" id="dial-heading">
            Discovery dial
          </h2>
          <DiscoveryDial
            value={effectiveLambda}
            personalised={listener.personalised_lambda}
            onChange={setLambda}
            onUseTheirSetting={() => setLambda(listener.personalised_lambda)}
          />
        </section>
      )}

      <section className="section" aria-labelledby="list-heading">
        <div className="section__head list-head">
          <h2 id="list-heading">What we&rsquo;d play next</h2>
          {recs.data && (
            <p className="list-head__count">
              {recs.data.anchor_count} of {recs.data.items.length} by artists they already play
            </p>
          )}
        </div>
        <div className="axis-key" aria-hidden="true">
          <span className="axis-key__scale">
            <span>familiar</span>
            <span>unknown</span>
          </span>
        </div>

        {recs.isError ? (
          <ErrorState title="Could not build this list" onRetry={() => recs.refetch()}>
            {(recs.error as ApiError).message}
          </ErrorState>
        ) : recs.isPending ? (
          <LoadingRows label="Building recommendations" />
        ) : recs.data?.items.length ? (
          <div className={recs.isFetching ? "is-restacking" : undefined}>
            <TrackList items={recs.data.items} />
          </div>
        ) : (
          <EmptyState title="No candidates for this listener">
            This listener has no unheard candidates in the artefacts. Pick another listener.
          </EmptyState>
        )}
      </section>
    </div>
  );
}
