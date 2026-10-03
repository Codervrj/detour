/** Results: does the map actually predict what people go on to listen to?
 *
 * Reports the numbers as measured, including the run where our first model choice lost.
 */

import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import { ErrorState, LoadingRows } from "../components/States";
import "./Results.css";

const LABELS: Record<string, string> = {
  random: "random",
  popularity: "popularity only",
  als: "ALS factors",
  item2vec: "item2vec",
  ppmi_svd: "PPMI + SVD (the map)",
};

// In a popularity-only space the matched negatives sit on top of the target by
// construction, so the control returns 1.0 and means nothing.
const NO_CONTROL = new Set(["popularity"]);

function num(value: number | null | undefined, digits = 4): string {
  return typeof value === "number" ? value.toFixed(digits) : "n/a";
}

export function Results() {
  const report = useQuery({ queryKey: ["report"], queryFn: api.latestReport });

  if (report.isPending) {
    return (
      <div className="wrap">
        <h1>Results</h1>
        <LoadingRows rows={5} label="Loading the latest report" />
      </div>
    );
  }

  if (report.isError) {
    const error = report.error as ApiError;
    return (
      <div className="wrap">
        <h1>Results</h1>
        <ErrorState title="No report yet" onRetry={() => report.refetch()}>
          {error.status === 0 ? (
            <>
              The API is not running. Start it with <code>uv run python run.py demo</code>.
            </>
          ) : (
            <>
              {error.message} Produce one with <code>uv run python run.py eval</code>.
            </>
          )}
        </ErrorState>
      </div>
    );
  }

  const data = report.data;
  const ours = data.models.ppmi_svd ?? {};
  const random = data.models.random ?? {};
  const percentile = ours.adoption_percentile;
  const matched = ours.matched_percentile;
  const chance = random.adoption_percentile;

  const ranked = Object.keys(LABELS)
    .filter((name) => data.models[name])
    .map((name) => ({ name, scores: data.models[name] }));
  const best = ranked.reduce<{ name: string; value: number } | null>((acc, row) => {
    const value = row.scores.adoption_percentile;
    if (typeof value !== "number") return acc;
    return !acc || value < acc.value ? { name: row.name, value } : acc;
  }, null);

  return (
    <div className="wrap">
      <header>
        <h1>Does the map work?</h1>
        <p className="lede">
          One run on the {data.split} split, {data.listeners_with_adoptions.toLocaleString()}{" "}
          listeners who adopted a new artist. These are the numbers as measured.
        </p>
      </header>

      <section className="section" aria-labelledby="test-heading">
        <div className="section__head">
          <h2 id="test-heading">The test</h2>
        </div>
        <p className="prose-p">
          Each listener is placed on the map using only their first nine months. Every artist is
          then ranked by distance from that position, and we check where the artists they really
          went on to play in the following months landed. A score of 0.5 means the map is
          worthless; lower is better.
        </p>
      </section>

      <section className="section" aria-labelledby="verdict-heading">
        <div className="section__head">
          <h2 id="verdict-heading">The answer</h2>
        </div>
        <div className="verdict">
          <p>
            The map scores <strong className="is-up">{num(percentile)}</strong>, against{" "}
            {num(chance)} for random placement. Real adoptions land far closer to a listener&rsquo;s
            territory than chance allows.
          </p>
          {typeof matched === "number" && (
            <p className={matched < 0.5 ? "verdict__ok" : "verdict__caveat"}>
              Popularity-matched control: <strong>{num(matched)}</strong>.{" "}
              {matched < 0.5
                ? "The result survives. Each real adoption was compared only against artists of similar global popularity, so this is not the model having quietly learned the charts."
                : "The result does not survive. The ranking is explained by popularity rather than taste, so the map has not been shown to work."}
            </p>
          )}
          {best && best.name !== "ppmi_svd" && (
            <p className="verdict__caveat">
              {LABELS[best.name]} scored better ({num(best.value)}). Reported as measured.
            </p>
          )}
        </div>
      </section>

      <section className="section" aria-labelledby="table-heading">
        <div className="section__head">
          <h2 id="table-heading">Every model</h2>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <caption className="visually-hidden">
              All baselines and the map, on the adoption question
            </caption>
            <thead>
              <tr>
                <th scope="col">model</th>
                <th scope="col">percentile</th>
                <th scope="col">matched control</th>
                <th scope="col">median rank</th>
                {data.k_values.map((k) => (
                  <th scope="col" key={k}>
                    hit@{k}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ranked.map(({ name, scores }) => (
                <tr key={name} className={name === "ppmi_svd" ? "is-ours" : undefined}>
                  <th scope="row">{LABELS[name]}</th>
                  <td>{num(scores.adoption_percentile)}</td>
                  <td>{NO_CONTROL.has(name) ? "not meaningful" : num(scores.matched_percentile)}</td>
                  <td>{num(scores.median_rank, 0)}</td>
                  {data.k_values.map((k) => (
                    <td key={k}>{num(scores[`hit@${k}`])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section" aria-labelledby="prov-heading">
        <div className="section__head">
          <h2 id="prov-heading">Where these numbers came from</h2>
        </div>
        <dl className="provenance">
          <dt>run</dt>
          <dd>{data.run_id}</dd>
          <dt>generated</dt>
          <dd>{new Date(data.generated_at).toLocaleString()}</dd>
          <dt>split</dt>
          <dd>{data.split}</dd>
          <dt>artists in the map</dt>
          <dd>{data.vocabulary.toLocaleString()}</dd>
          <dt>config hash</dt>
          <dd>{data.config_hash}</dd>
          <dt>git commit</dt>
          <dd>{data.git_commit ?? "not a git repository"}</dd>
          <dt>confidence intervals</dt>
          <dd>{data.confidence_intervals ? "included" : "not computed yet"}</dd>
        </dl>
      </section>
    </div>
  );
}
