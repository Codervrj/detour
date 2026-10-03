/** The map of music, drawn on canvas.
 *
 * Twenty-two thousand artists will not survive as SVG DOM nodes, so this paints to a
 * canvas: the whole catalogue as a faint field, the listener's own artists on top, and
 * labels only where there is room for them.
 *
 * A canvas is invisible to a screen reader, so everything shown here is also available as
 * text on the page beneath it. The canvas is marked presentational rather than given a
 * misleading label.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { ArtistPoint, Neighbour } from "../api/types";
import "./ArtistMap.css";

interface Props {
  background: ArtistPoint[];
  mine: ArtistPoint[];
  frontier: Neighbour[];
  onPick: (artist: ArtistPoint) => void;
  selected: string | null;
}

const PAD = 28;
const LABEL_LIMIT = 28;

type Placed = { point: ArtistPoint; sx: number; sy: number; r: number };

function readToken(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

export function ArtistMap({ background, mine, frontier, onPick, selected }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const placedRef = useRef<Placed[]>([]);
  const [hovered, setHovered] = useState<ArtistPoint | null>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const ratio = window.devicePixelRatio || 1;
    const width = parent.clientWidth;
    const height = parent.clientHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const ink = readToken("--ink", "#16222b");
    const mid = readToken("--mid", "#6b7d87");
    const familiar = readToken("--familiar", "#0f6e72");
    const unknown = readToken("--unknown", "#c2365b");
    const surface = readToken("--surface", "#f7f9fa");

    const sx = (x: number) => PAD + x * (width - PAD * 2);
    const sy = (y: number) => PAD + (1 - y) * (height - PAD * 2);

    // The whole catalogue, faint: this is the shape of recorded music itself.
    ctx.fillStyle = mid;
    ctx.globalAlpha = 0.3;
    for (const point of background) {
      ctx.beginPath();
      ctx.arc(sx(point.x), sy(point.y), 1.4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    // Where this listener could go next.
    ctx.strokeStyle = unknown;
    ctx.lineWidth = 2;
    for (const point of frontier) {
      const px = sx(point.x);
      const py = sy(point.y);
      // A ring with a clear centre, so it reads as "not yours" against the solid dots.
      ctx.beginPath();
      ctx.arc(px, py, 6.5, 0, Math.PI * 2);
      ctx.fillStyle = surface;
      ctx.globalAlpha = 0.9;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.stroke();
    }
    ctx.lineWidth = 1;

    // The listener's own artists, sized by how much they play them.
    const maxPlays = Math.max(1, ...mine.map((a) => a.plays));
    const placed: Placed[] = [];
    for (const point of mine) {
      const r = 2.5 + 7 * Math.sqrt(point.plays / maxPlays);
      const px = sx(point.x);
      const py = sy(point.y);
      placed.push({ point, sx: px, sy: py, r });

      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = familiar;
      ctx.globalAlpha = point.artist_mbid === selected ? 1 : 0.8;
      ctx.fill();
      if (point.artist_mbid === selected) {
        ctx.strokeStyle = ink;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    placedRef.current = placed;

    // Label only the biggest, and only where nothing has been labelled already.
    const taken: Array<[number, number]> = [];
    const byPlays = [...placed].sort((a, b) => b.point.plays - a.point.plays);
    ctx.font = "500 12px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    let drawn = 0;
    for (const item of byPlays) {
      if (drawn >= LABEL_LIMIT) break;
      const clash = taken.some(
        ([tx, ty]) => Math.abs(tx - item.sx) < 76 && Math.abs(ty - item.sy) < 16,
      );
      if (clash) continue;
      taken.push([item.sx, item.sy]);
      drawn += 1;

      const label = item.point.name;
      const tx = item.sx + item.r + 5;
      const metrics = ctx.measureText(label);
      ctx.fillStyle = surface;
      ctx.globalAlpha = 0.82;
      ctx.fillRect(tx - 2, item.sy - 8, metrics.width + 4, 16);
      ctx.globalAlpha = 1;
      ctx.fillStyle = ink;
      ctx.fillText(label, tx, item.sy);
    }
  }, [background, mine, frontier, selected]);

  useEffect(() => {
    draw();
    const observer = new ResizeObserver(() => draw());
    const parent = canvasRef.current?.parentElement;
    if (parent) observer.observe(parent);
    return () => observer.disconnect();
  }, [draw]);

  function at(event: React.MouseEvent<HTMLCanvasElement>): ArtistPoint | null {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let best: Placed | null = null;
    let bestDistance = Infinity;
    for (const item of placedRef.current) {
      const distance = (item.sx - x) ** 2 + (item.sy - y) ** 2;
      if (distance < bestDistance && distance < (item.r + 6) ** 2) {
        best = item;
        bestDistance = distance;
      }
    }
    return best ? best.point : null;
  }

  return (
    <div className="artist-map">
      <canvas
        ref={canvasRef}
        role="presentation"
        onMouseMove={(event) => setHovered(at(event))}
        onMouseLeave={() => setHovered(null)}
        onClick={(event) => {
          const point = at(event);
          if (point) onPick(point);
        }}
      />
      {hovered && (
        <p className="artist-map__hover">
          {hovered.name}{" "}
          <span>
            {hovered.plays.toLocaleString()} {hovered.plays === 1 ? "play" : "plays"}
          </span>
        </p>
      )}
      <ul className="artist-map__key">
        <li>
          <span className="dot dot--mine" /> artists you play
        </li>
        <li>
          <span className="dot dot--frontier" /> nearest you have not played
        </li>
        <li>
          <span className="dot dot--all" /> everything else
        </li>
      </ul>
    </div>
  );
}
