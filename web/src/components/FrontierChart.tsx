/** The accuracy-versus-novelty frontier: the headline result.
 *
 * Built from visx scales rather than a chart template, because the shape of this chart is
 * the argument. The fixed-lambda sweep draws a curve; our personalised model is a single
 * point that should sit above and to the right of it. Where it does not, the chart says so.
 */

import { useId } from "react";
import { scaleLinear } from "@visx/scale";
import type { FrontierPoint } from "../api/types";
import "./FrontierChart.css";

interface Props {
  sweep: FrontierPoint[];
  ours: { ndcg: number; novelty: number } | null;
  k: number;
}

const WIDTH = 720;
const HEIGHT = 420;
const PAD = { top: 24, right: 28, bottom: 56, left: 68 };

export function FrontierChart({ sweep, ours, k }: Props) {
  const titleId = useId();

  const points = sweep
    .map((row) => ({
      lambda: row.lambda,
      novelty: row[`novelty@${k}`] as number | null,
      ndcg: row[`ndcg@${k}`] as number | null,
    }))
    .filter((row): row is { lambda: number; novelty: number; ndcg: number } =>
      row.novelty !== null && row.ndcg !== null,
    );

  if (!points.length) {
    return (
      <div className="state">
        <h3>No frontier to draw</h3>
        <p>The report contains no lambda sweep. Re-run the eval to produce one.</p>
      </div>
    );
  }

  const noveltyValues = [...points.map((p) => p.novelty), ...(ours ? [ours.novelty] : [])];
  const ndcgValues = [...points.map((p) => p.ndcg), ...(ours ? [ours.ndcg] : [])];

  const x = scaleLinear({
    domain: [Math.min(...noveltyValues) * 0.95, Math.max(...noveltyValues) * 1.05],
    range: [PAD.left, WIDTH - PAD.right],
  });
  const y = scaleLinear({
    domain: [0, Math.max(...ndcgValues) * 1.15],
    range: [HEIGHT - PAD.bottom, PAD.top],
  });

  const path = points.map((p, i) => `${i ? "L" : "M"}${x(p.novelty)},${y(p.ndcg)}`).join(" ");

  return (
    <figure className="frontier">
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        className="frontier__svg"
        role="img"
        aria-labelledby={titleId}
      >
        <title id={titleId}>
          Accuracy against novelty at K equals {k}. The curve is the fixed-lambda sweep from{" "}
          {points[0].lambda.toFixed(2)} to {points[points.length - 1].lambda.toFixed(2)}.
          {ours
            ? ` Our personalised model reaches NDCG ${ours.ndcg.toFixed(4)} at novelty ${ours.novelty.toFixed(2)}.`
            : ""}
        </title>

        {y.ticks(5).map((tick) => (
          <g key={tick}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={y(tick)}
              y2={y(tick)}
              className="frontier__grid"
            />
            <text x={PAD.left - 10} y={y(tick)} className="frontier__tick frontier__tick--y">
              {tick.toFixed(3)}
            </text>
          </g>
        ))}

        {x.ticks(6).map((tick) => (
          <text
            key={tick}
            x={x(tick)}
            y={HEIGHT - PAD.bottom + 20}
            className="frontier__tick frontier__tick--x"
          >
            {tick.toFixed(1)}
          </text>
        ))}

        <line
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
          className="frontier__axis"
        />
        <line
          x1={PAD.left}
          x2={PAD.left}
          y1={PAD.top}
          y2={HEIGHT - PAD.bottom}
          className="frontier__axis"
        />

        <path d={path} className="frontier__curve" />

        {points.map((p) => (
          <g key={p.lambda}>
            <circle cx={x(p.novelty)} cy={y(p.ndcg)} r={4.5} className="frontier__dot" />
            <text x={x(p.novelty)} y={y(p.ndcg) - 12} className="frontier__dot-label">
              {p.lambda.toFixed(1)}
            </text>
          </g>
        ))}

        {ours && (
          <g>
            <line
              x1={x(ours.novelty)}
              x2={x(ours.novelty)}
              y1={y(ours.ndcg)}
              y2={HEIGHT - PAD.bottom}
              className="frontier__drop"
            />
            <circle cx={x(ours.novelty)} cy={y(ours.ndcg)} r={9} className="frontier__ours" />
            <text x={x(ours.novelty) + 18} y={y(ours.ndcg) + 4} className="frontier__ours-label">
              ours
            </text>
          </g>
        )}

        <text x={WIDTH / 2} y={HEIGHT - 12} className="frontier__axis-title">
          novelty at {k}, mean bits of surprise
        </text>
        <text
          transform={`translate(18 ${HEIGHT / 2}) rotate(-90)`}
          className="frontier__axis-title"
        >
          NDCG at {k}
        </text>
      </svg>

      <figcaption>
        Each dot is one fixed dial position applied to every listener. Further right is more
        novel; higher is more accurate. Our model gives each listener their own position, so it
        is a single point rather than a curve.
      </figcaption>
    </figure>
  );
}
