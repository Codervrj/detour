/** App shell: masthead, hash routing, footer.
 *
 * Hash routing keeps the build a static bundle with no server rewrites, which matters
 * because this is served by a dev server or opened from disk.
 */

import { useEffect, useState } from "react";
import { MapScreen } from "./screens/Map";
import { Islands } from "./screens/Islands";
import { Results } from "./screens/Results";
import { Method } from "./screens/Method";

const ROUTES = [
  { hash: "#/map", label: "Your map", element: <MapScreen /> },
  { hash: "#/islands", label: "Islands", element: <Islands /> },
  { hash: "#/results", label: "Results", element: <Results /> },
  { hash: "#/method", label: "Method", element: <Method /> },
] as const;

function currentHash(): string {
  const hash = window.location.hash;
  return ROUTES.some((route) => route.hash === hash) ? hash : ROUTES[0].hash;
}

export function App() {
  const [route, setRoute] = useState(currentHash);

  useEffect(() => {
    const onChange = () => setRoute(currentHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const active = ROUTES.find((candidate) => candidate.hash === route) ?? ROUTES[0];

  return (
    <div className="shell">
      <header className="masthead">
        <div className="wrap masthead__inner">
          <a className="wordmark" href={ROUTES[0].hash}>
            De<em>tour</em>
          </a>
          <nav className="nav" aria-label="Main">
            {ROUTES.map((item) => (
              <a
                key={item.hash}
                href={item.hash}
                aria-current={item.hash === active.hash ? "page" : undefined}
              >
                {item.label}
              </a>
            ))}
          </nav>
        </div>
      </header>

      <main>{active.element}</main>

      <footer className="page-foot">
        <div className="wrap">
          Built on the ListenBrainz data model from the MetaBrainz Foundation, with metadata from
          MusicBrainz. Listening histories from MLHD+, artist names from MusicBrainz.
        </div>
      </footer>
    </div>
  );
}
