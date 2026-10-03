/** The discovery dial: one control, on the same axis the track markers sit on.
 *
 * It is a real range input, so keyboard and screen-reader support come for free.
 * A ghost mark shows where this listener's own personalised lambda falls, which is
 * the whole point of the project: their setting is not the middle of the track.
 */

import "./DiscoveryDial.css";

interface Props {
  value: number;
  personalised: number;
  onChange: (value: number) => void;
  onUseTheirSetting: () => void;
  disabled?: boolean;
}

const PERCENT = (value: number) => `${(value * 100).toFixed(1)}%`;

export function DiscoveryDial({
  value,
  personalised,
  onChange,
  onUseTheirSetting,
  disabled = false,
}: Props) {
  const atTheirSetting = Math.abs(value - personalised) < 0.005;

  return (
    <div className="dial">
      <div className="dial__head">
        <label className="dial__label" htmlFor="discovery-dial">
          How far out should we go?
        </label>
        <output className="dial__value" htmlFor="discovery-dial">
          {value.toFixed(2)}
        </output>
      </div>

      <div className="dial__track">
        <input
          id="discovery-dial"
          className="dial__input"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={value}
          disabled={disabled}
          aria-describedby="dial-ends"
          aria-valuetext={`${value.toFixed(2)} of 1, where 0 keeps to familiar artists and 1 pushes furthest out`}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <span
          className="dial__ghost"
          style={{ left: PERCENT(personalised) }}
          aria-hidden="true"
        />
      </div>

      <div className="dial__ends" id="dial-ends">
        <span>familiar</span>
        <span>unknown</span>
      </div>

      <p className="dial__note">
        {atTheirSetting ? (
          <>This listener&rsquo;s own setting is {personalised.toFixed(2)}, from their listening history.</>
        ) : (
          <>
            Their own setting is {personalised.toFixed(2)}.{" "}
            <button type="button" className="button button--quiet" onClick={onUseTheirSetting}>
              Use their setting
            </button>
          </>
        )}
      </p>
    </div>
  );
}
