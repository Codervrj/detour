/** Method: the data, the model, the test and the limitations, in plain language. */

import "./Method.css";

export function Method() {
  return (
    <div className="wrap">
      <header>
        <h1>How this works</h1>
        <p className="lede">
          No part of this is magic, and some of it does not work as well as I first expected.
          Here is the whole method, including what it cannot do.
        </p>
      </header>

      <div className="prose">
        <section>
          <h2>The data is real</h2>
          <p>
            Fourteen point eight million plays by 2,000 real Last.fm listeners across the whole
            of 2011, taken from MLHD+, a research dataset published by MetaBrainz that maps
            Last.fm listening histories onto MusicBrainz identifiers.
          </p>
          <p>
            Nobody is invented. Every artist is a real artist and every play happened. Artist
            names come from the MusicBrainz canonical dump, which covers 89% of the artists in
            the map; the rest are 2011-era identifiers that MusicBrainz has since merged away.
          </p>
        </section>

        <section>
          <h2>Building the map</h2>
          <p>
            Plays of the same artist by the same listener within thirty seconds count once. The
            split is by time, never at random: the first nine months train the model, month ten
            tunes it, and the last two months are held back. Which artists are eligible is
            decided from the training months alone, so nothing from the future leaks in.
          </p>
          <p>
            Then: count how many listeners each pair of artists shares, convert those counts to
            pointwise mutual information so a pair only counts as related when they co-occur more
            than chance would predict, and compress the result with a truncated SVD. Popularity
            cancels out in that middle step, which is why the map is not simply a chart.
          </p>
          <p>
            The map has 128 dimensions. The picture you see is flattened to two with UMAP for
            drawing only. <strong>Every number on this site is computed in the full 128
            dimensions</strong>, never on the flattened picture, because flattening distorts
            distance by construction.
          </p>
        </section>

        <section>
          <h2>Putting you on it</h2>
          <p>
            You were never in the training data. Your position is the average of the artists you
            play, weighted by the logarithm of your play counts so that one heavy rotation does
            not drown out everything else. That is all it takes: the model generalises to people
            it has never seen.
          </p>
        </section>

        <section>
          <h2>How it is judged</h2>
          <p>
            Place a listener using only their first nine months, rank every artist by distance
            from that position, and look at where the artists they actually went on to play
            landed. Chance is 0.5. The map scores 0.116.
          </p>
          <p className="callout">
            The trap in that test: popular artists get adopted more often <em>and</em> sit
            centrally on any map, so a model that learned nothing but the charts would still
            score well. So every real adoption is also compared only against artists of similar
            global popularity. The map survives that control, which is what makes the number
            worth anything.
          </p>
        </section>

        <section>
          <h2>What went wrong, and what I changed</h2>
          <p>
            The first model was item2vec, a neural embedding trained on listening sessions. It
            scored 0.289. A far simpler method with no neural training at all, counting
            co-listening and taking an SVD, scored 0.116 and found seven times as many artists in
            the top ten. So the simple method became the map and item2vec was demoted to a
            baseline.
          </p>
          <p>
            The likely reason is that the two learn different things. item2vec learns &ldquo;played
            in the same sitting&rdquo;, and with a median session of four tracks there is very
            little context to learn from. The counting method learns &ldquo;these artists share
            listeners across nine months&rdquo;, which is a much better guide to what somebody
            will adopt next.
          </p>
        </section>

        <section>
          <h2>What this cannot tell you</h2>
          <ul>
            <li>
              The listening data is from 2011. Nothing released since exists on this map at all.
            </li>
            <li>
              Confidence intervals are not computed yet, so treat small differences between models
              as noise. The gap between the top two is far too large to be noise.
            </li>
            <li>
              Artists with fewer than five listeners in the training year are not on the map, so
              the deepest part of the tail is missing.
            </li>
            <li>
              It knows nothing about how music sounds. Every relationship here comes from who
              listened to what.
            </li>
            <li>
              Two thousand listeners is a small sample of the 583,000 in the full dataset.
            </li>
          </ul>
        </section>

        <section>
          <h2>Credits</h2>
          <p>
            Listening histories: <a href="https://musicbrainz.org/doc/MLHD">MLHD+</a>, published by
            the MetaBrainz Foundation for research use, derived from Last.fm.
          </p>
          <p>
            Artist names and identifiers: <a href="https://musicbrainz.org/">MusicBrainz</a>, also
            MetaBrainz. The listening data model follows{" "}
            <a href="https://listenbrainz.org/">ListenBrainz</a> (CC0).
          </p>
          <p>Your own history, when you import it, comes from the Last.fm API and stays on your machine.</p>
        </section>
      </div>
    </div>
  );
}
