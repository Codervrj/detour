/** Results: what the eval actually found, including where it went against us.
 *
 * Reads the latest report JSON. Numbers are shown as measured; the page does not hide a
 * baseline that beat us.
 */

import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "../api/client";
import { FrontierChart } from "../components/FrontierChart";
import { ErrorState, LoadingRows } from "../components/States";
import "./Results.css";

const MODEL_LABELS: Record<string, string> = {
  random: "random",
  popularity: "popularity",
  itemknn: "item-kNN",
  als: "ALS",
  als_mmr_fixed: "ALS with one global dial",
  ours: "ALS with a dial per listener",
};

const METRICS = [
  { key: "ndcg", label: "NDCG" },
  { key: "discovery_recall", label: "discovery recall" },
  { key: "novelty", label: "novelty" },
  { key: "familiarity_anchor_rate", label: "familiar share" },
  { key: "catalogue_coverage", label: "coverage" },
];

function num(value: unknown, digits = 4): string {
  return typeof value === "number" ? value.toFixed(digits) : "not applicable";
}

export function Results() {
  const report = useQuery({ queryKey: ["report"], queryFn: api.latestReport });

  if (report.isPending) {
    return (
      <div className="wrap">
        <h1>Results</h1>
        <LoadingRows label="Loading the latest eval report" />
      </div>
    );
  }

  if (report.isError) {
    const error = report.error as ApiError;
    return (
      <div className="wrap">
        <h1>Results</h1>
        <ErrorState title="No eval report yet" onRetry={() => report.refetch()}>
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
  const k = Math.max(...data.k_values);
  const ours = data.models.ours;
  const als = data.models.als;
  const popularity = data.models.popularity;

  const oursNdcg = ours?.[`ndcg@${k}`];
  const oursNovelty = ours?.[`novelty@${k}`];
  const oursDiscovery = ours?.[`discovery_recall@${k}`];
  const alsDiscovery = als?.[`discovery_recall@${k}`];
  const alsNdcg = als?.[`ndcg@${k}`];
  const popNdcg = popularity?.[`ndcg@${k}`];

  const discoveryDelta =
    typeof oursDiscovery === "number" && typeof alsDiscovery === "number"
      ? oursDiscovery - alsDiscovery
      : null;
  const ndcgDelta =
    typeof oursNdcg === "number" && typeof alsNdcg === "number" ? oursNdcg - alsNdcg : null;

  const segments = (ours?.[`segments@${k}`] ?? {}) as Record<string, Record<string, number>>;

  return (
    <div className="wrap">
      <header>
        <h1>Results</h1>
        <p className="lede">
          One run on the {data.split} split, {data.users_evaluated} listeners. These are the
          numbers as measured.
        </p>
      </header>

      <section className="section" aria-labelledby="verdict-heading">
        <div className="section__head">
          <h2 id="verdict-heading">Did it work?</h2>
        </div>
        <div className="verdict">
          <p>
            Against plain ALS, giving each listener their own dial position moved discovery
            recall by{" "}
            <strong className={discoveryDelta !== null && discoveryDelta < 0 ? "is-down" : "is-up"}>
              {discoveryDelta === null
                ? "an amount we cannot measure"
                : `${discoveryDelta >= 0 ? "+" : ""}${discoveryDelta.toFixed(4)}`}
            </strong>{" "}
            and NDCG by{" "}
            <strong className={ndcgDelta !== null && ndcgDelta < 0 ? "is-down" : "is-up"}>
              {ndcgDelta === null
                ? "an amount we cannot measure"
                : `${ndcgDelta >= 0 ? "+" : ""}${ndcgDelta.toFixed(4)}`}
            </strong>{" "}
            at K={k}.
          </p>
          {discoveryDelta !== null && discoveryDelta < 0 && (
            <p className="verdict__caveat">
              Discovery recall went down, so on this run the headline claim does not hold. The
              novelty term rewards globally rare tracks, which is not the same thing as the
              artists a listener actually went on to adopt. The lambda mapping has not yet been
              fitted on the validation split, which is the most likely cause.
            </p>
          )}
          <p>
            Against the popularity baseline on NDCG@{k}:{" "}
            {typeof oursNdcg === "number" && typeof popNdcg === "number"
              ? oursNdcg > popNdcg
                ? `ahead, ${num(oursNdcg)} against ${num(popNdcg)}.`
                : `behind, ${num(oursNdcg)} against ${num(popNdcg)}.`
              : "not measurable on this run."}
          </p>
        </div>
      </section>

      <section className="section" aria-labelledby="frontier-heading">
        <div className="section__head">
          <h2 id="frontier-heading">Accuracy against novelty</h2>
        </div>
        <FrontierChart
          sweep={data.lambda_sweep}
          ours={
            typeof oursNdcg === "number" && typeof oursNovelty === "number"
              ? { ndcg: oursNdcg, novelty: oursNovelty }
              : null
          }
          k={k}
        />
      </section>

      <section className="section" aria-labelledby="table-heading">
        <div className="section__head">
          <h2 id="table-heading">Every model at K={k}</h2>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <caption className="visually-hidden">
              All baselines and our model, every metric at K={k}
            </caption>
            <thead>
              <tr>
                <th scope="col">model</th>
                {METRICS.map((metric) => (
                  <th scope="col" key={metric.key}>
                    {metric.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.keys(MODEL_LABELS)
                .filter((name) => data.models[name])
                .map((name) => (
                  <tr key={name} className={name === "ours" ? "is-ours" : undefined}>
                    <th scope="row">{MODEL_LABELS[name]}</th>
                    {METRICS.map((metric) => (
                      <td key={metric.key}>{num(data.models[name][`${metric.key}@${k}`])}</td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section" aria-labelledby="segment-heading">
        <div className="section__head">
          <h2 id="segment-heading">Our model by how adventurous listeners are</h2>
        </div>
        <div className="table-scroll">
          <table className="data-table">
            <caption className="visually-hidden">
              Our model split by explorer tercile, and cold listeners separately
            </caption>
            <thead>
              <tr>
                <th scope="col">segment</th>
                <th scope="col">listeners</th>
                <th scope="col">NDCG</th>
                <th scope="col">discovery recall</th>
                <th scope="col">novelty</th>
                <th scope="col">familiar share</th>
              </tr>
            </thead>
            <tbody>
              {["loyalists", "middle", "explorers", "cold"].map((name) => (
                <tr key={name}>
                  <th scope="row">{name}</th>
                  <td>{segments[name]?.users ?? 0}</td>
                  <td>{num(segments[name]?.[`ndcg@${k}`])}</td>
                  <td>{num(segments[name]?.[`discovery_recall@${k}`])}</td>
                  <td>{num(segments[name]?.[`novelty@${k}`])}</td>
                  <td>{num(segments[name]?.[`familiarity_anchor_rate@${k}`])}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section" aria-labelledby="provenance-heading">
        <div className="section__head">
          <h2 id="provenance-heading">Where these numbers came from</h2>
        </div>
        <dl className="provenance">
          <dt>run</dt>
          <dd>{data.run_id}</dd>
          <dt>generated</dt>
          <dd>{new Date(data.generated_at).toLocaleString()}</dd>
          <dt>split</dt>
          <dd>{data.split}</dd>
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
