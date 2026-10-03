/** Method: the data, the split, the metrics and the limitations, in plain language.
 *
 * This page also carries the attribution the licences require.
 */

import "./Method.css";

export function Method() {
  return (
    <div className="wrap">
      <header>
        <h1>How this works</h1>
        <p className="lede">
          No part of this is magic, and some of it does not work yet. Here is the whole method,
          including what it cannot do.
        </p>
      </header>

      <div className="prose">
        <section>
          <h2>The data</h2>
          <p>
            The design target is ListenBrainz listening history: real play events with a
            timestamp, a pseudonymous listener id and MusicBrainz identifiers for the recording
            and artist.
          </p>
          <p className="callout">
            What you are looking at right now is <strong>synthetic data</strong>, generated
            locally by <code>tests/fixtures/make_fixtures.py</code>. Fifty invented listeners
            across twelve months of 2024, with artist popularity on a Zipf curve and eight
            latent taste groups. No ListenBrainz data has been downloaded. The ingest code for
            the real thing is not written yet, so every number on this site describes invented
            listeners.
          </p>
        </section>

        <section>
          <h2>Getting from plays to a model</h2>
          <p>
            The same recording played twice by the same listener within thirty seconds counts
            once. A track is identified by its recording id, falling back to a normalised artist
            and title when that is missing.
          </p>
          <p>
            The split is by time, never at random: the first nine months train the model, month
            ten tunes it, and the last two months are held back. Filters on how active a
            listener is and how many people heard a track are computed from the training months
            alone, so nothing from the future decides who gets evaluated.
          </p>
        </section>

        <section>
          <h2>How adventurous is a listener?</h2>
          <p>
            For each month we take the share of that listener&rsquo;s plays that went to artists
            they had never heard before, smooth those monthly shares, and average them into one
            number between 0 and 1. A listener&rsquo;s first month is left out, because in month
            one every artist is new and counting it would make a creature of habit look like an
            adventurer.
          </p>
        </section>

        <section>
          <h2>Choosing what to play next</h2>
          <p>
            A collaborative filtering model proposes two hundred candidates per listener from
            what similar listeners played. Those candidates are then re-ordered by a score that
            trades three things off: how likely the listener is to want it, how rare it is, and
            how similar it already is to what we have picked. The dial sets the weight between
            the first two.
          </p>
          <p>
            Every list keeps a minimum number of tracks by artists the listener already plays, so
            turning the dial up never returns a list of total strangers. Each row carries a short
            reason it is there.
          </p>
        </section>

        <section>
          <h2>How we judge it</h2>
          <p>
            We ask whether a listener actually played the recommended tracks in the held-out
            period, and separately whether they went on to adopt artists that never appear in
            their history. Those are different questions, and a model can win one while losing
            the other. Alongside them we measure how rare the recommendations are, how varied,
            how much of the catalogue gets seen at all, and how unevenly attention is spread.
          </p>
          <p>
            Six approaches are compared on every run, from a random list and a straight
            popularity chart up to our own. The results page shows all of them.
          </p>
        </section>

        <section>
          <h2>What this cannot tell you</h2>
          <ul>
            <li>
              The listeners are invented. Numbers here say the plumbing works; they say nothing
              about real taste.
            </li>
            <li>
              Confidence intervals are not computed yet, so treat small differences between
              models as noise.
            </li>
            <li>
              The dial mapping from explorer score to weighting has not been fitted on the
              tuning month. It currently uses placeholder values, which is the leading
              explanation for why our discovery numbers trail plain collaborative filtering.
            </li>
            <li>
              Rarity is not the same as discovery. Rewarding globally obscure tracks pushes the
              list towards things nobody played, including this listener.
            </li>
            <li>
              Variety between tracks is measured with the collaborative model&rsquo;s own
              factors, which flatters it. A separate embedding would be a fairer judge.
            </li>
          </ul>
        </section>

        <section>
          <h2>Credits</h2>
          <p>
            Listening data model and format:{" "}
            <a href="https://listenbrainz.org/">ListenBrainz</a>, by the MetaBrainz Foundation,
            released under CC0.
          </p>
          <p>
            Artist, recording and tag metadata:{" "}
            <a href="https://musicbrainz.org/">MusicBrainz</a>, also MetaBrainz.
          </p>
          <p>
            The optional audio-similarity module, which is not built, would use the Free Music
            Archive dataset (Defferrard and others, ISMIR 2017; metadata CC BY 4.0).
          </p>
        </section>
      </div>
    </div>
  );
}
