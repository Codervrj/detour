/** The re-ranked list. Each row carries a marker on the same familiar-to-unknown axis
 * as the dial, so turning the dial makes the whole set visibly drift outward.
 *
 * The marker is decoration over data that is also written out in text, so nothing is
 * conveyed by position or colour alone.
 */

import { useState } from "react";
import type { TrackRecommendation } from "../api/types";
import "./TrackList.css";

interface Props {
  items: TrackRecommendation[];
}

export function TrackList({ items }: Props) {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <ol className="tracks" aria-label="Recommended tracks, most relevant first">
      {items.map((item) => {
        const isOpen = open === item.item_id;
        return (
          <li key={item.item_id} className="track">
            <span className="track__rank" aria-hidden="true">
              {item.rank}
            </span>

            <div className="track__body">
              <p className="track__title">{item.track_name}</p>
              <p className="track__artist">{item.artist_name}</p>
            </div>

            <div className="track__axis">
              <span
                className={`track__marker ${item.is_anchor ? "is-familiar" : "is-unknown"}`}
                style={{ left: `${(item.novelty * 100).toFixed(1)}%` }}
              />
              <span className="visually-hidden">
                {item.is_anchor ? "Familiar artist" : "New artist"}, novelty{" "}
                {item.novelty.toFixed(2)} of 1
              </span>
            </div>

            <div className="track__meta">
              <span className={`tag ${item.is_anchor ? "tag--familiar" : "tag--unknown"}`}>
                {item.is_anchor ? "you know them" : "new to you"}
              </span>
              <button
                type="button"
                className="track__why"
                aria-expanded={isOpen}
                onClick={() => setOpen(isOpen ? null : item.item_id)}
              >
                why this
              </button>
            </div>

            {isOpen && <p className="track__reason">{item.why}</p>}
          </li>
        );
      })}
    </ol>
  );
}
