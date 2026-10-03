/** Listener profile: where each listener sits on the explorer scale.
 *
 * The scale itself is the chart. Every listener is a tick on one axis, and the selected
 * listener is marked, so "how adventurous is this person" is answered by position rather
 * than by a number in a box.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import { band } from "../components/ListenerPicker";
import { ErrorState, LoadingRows } from "../components/States";
import "./Profile.css";

export function Profile() {
  const [selected, setSelected] = useState<string | null>(null);
  const listeners = useQuery({ queryKey: ["listeners"], queryFn: api.listeners });

  if (listeners.isPending) {
    return (
      <div className="wrap">
        <h1>Listeners</h1>
        <LoadingRows label="Loading listeners" />
      </div>
    );
  }

  if (listeners.isError) {
    const error = listeners.error as ApiError;
    return (
      <div className="wrap">
        <h1>Listeners</h1>
        <ErrorState title="No listeners to show" onRetry={() => listeners.refetch()}>
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

  const rows = listeners.data;
  const current = rows.find((row) => row.user_id === selected) ?? rows[0];
  const maxListens = Math.max(...rows.map((row) => row.train_listens));

  return (
    <div className="wrap">
      <header>
        <h1>Who is listening</h1>
        <p className="lede">
          Every listener sits somewhere between replaying what they know and roaming. That
          position is what sets their dial.
        </p>
      </header>

      <section className="section" aria-labelledby="scale-heading">
        <div className="section__head">
          <h2 id="scale-heading">The explorer scale</h2>
        </div>

        <div className="scale">
          <div className="scale__axis">
            <div className="scale__ticks">
            {rows.map((row) => (
              <button
                key={row.user_id}
                type="button"
                className={`scale__tick ${row.user_id === current.user_id ? "is-current" : ""}`}
                style={{ left: `${row.explorer_score * 100}%` }}
                onClick={() => setSelected(row.user_id)}
                aria-pressed={row.user_id === current.user_id}
              >
                <span className="visually-hidden">
                  {row.user_id}, explorer score {row.explorer_score.toFixed(2)}
                </span>
              </button>
            ))}
            </div>
          </div>
          <div className="scale__ends">
            <span>0.00 replays what they know</span>
            <span>1.00 always moving on</span>
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="detail-heading">
        <div className="section__head">
          <h2 id="detail-heading">{current.user_id}</h2>
        </div>

        <p className="profile__sentence">
          This listener {band(current.explorer_score)}, scoring{" "}
          <strong>{current.explorer_score.toFixed(2)}</strong> on the explorer scale across{" "}
          {current.train_listens.toLocaleString()} listens in the training period. Detour gives
          them a dial position of <strong>{current.personalised_lambda.toFixed(2)}</strong>.
        </p>

        {current.top_artists.length > 0 && (
          <div className="profile__artists">
            <h3>Most played</h3>
            <ol className="bars">
              {current.top_artists.map((artist, index) => (
                <li key={artist} className="bars__row">
                  <span className="bars__label">{artist}</span>
                  <span
                    className="bars__fill"
                    style={{ inlineSize: `${100 - index * 22}%` }}
                    aria-hidden="true"
                  />
                </li>
              ))}
            </ol>
          </div>
        )}
      </section>

      <section className="section" aria-labelledby="all-heading">
        <div className="section__head">
          <h2 id="all-heading">All listeners</h2>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <caption className="visually-hidden">Every sample listener</caption>
            <thead>
              <tr>
                <th scope="col">listener</th>
                <th scope="col">listens</th>
                <th scope="col">explorer score</th>
                <th scope="col">dial position</th>
                <th scope="col">habit</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.user_id}>
                  <th scope="row">
                    <button
                      type="button"
                      className="button button--quiet"
                      onClick={() => setSelected(row.user_id)}
                    >
                      {row.user_id}
                    </button>
                  </th>
                  <td>{row.train_listens.toLocaleString()}</td>
                  <td>{row.explorer_score.toFixed(2)}</td>
                  <td>{row.personalised_lambda.toFixed(2)}</td>
                  <td>{band(row.explorer_score)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="profile__foot">
          Longest history here is {maxListens.toLocaleString()} listens.
        </p>
      </section>
    </div>
  );
}
