/** Sample-listener picker. Ordered by explorer score so the spectrum reads left to
 * right, from people who replay what they know to people who roam. */

import type { Listener } from "../api/types";
import "./ListenerPicker.css";

interface Props {
  listeners: Listener[];
  selected: string | null;
  onSelect: (userId: string) => void;
}

/** Plain-language band for an explorer score, matching the eval's terciles. */
export function band(score: number): string {
  if (score < 0.18) return "stays close";
  if (score < 0.34) return "steps out";
  return "roams";
}

export function ListenerPicker({ listeners, selected, onSelect }: Props) {
  return (
    <div className="pickers" role="radiogroup" aria-label="Sample listeners">
      {listeners.map((listener) => (
        <div className="picker" key={listener.user_id}>
          <input
            type="radio"
            name="listener"
            id={`listener-${listener.user_id}`}
            value={listener.user_id}
            checked={selected === listener.user_id}
            onChange={() => onSelect(listener.user_id)}
          />
          <label className="picker__face" htmlFor={`listener-${listener.user_id}`}>
            <span className="picker__id">{listener.user_id}</span>
            <span className="picker__score">
              {band(listener.explorer_score)} {listener.explorer_score.toFixed(2)}
            </span>
          </label>
        </div>
      ))}
    </div>
  );
}
