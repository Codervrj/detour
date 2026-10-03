"""Report schema and writer.

Writes `evals/reports/<date>_<run_id>.json` with all metrics, the config hash, the data
sidecar stats and the git commit, plus a `.md` summary. The frontend's results page reads
the latest JSON.

Keys for bootstrap CIs are present and null until that increment lands, so the frontend
contract does not have to change when they arrive.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORTS_DIR = Path("evals/reports")
MODEL_ORDER = ["random", "popularity", "itemknn", "als", "als_mmr_fixed", "ours"]
MODEL_LABELS = {
    "random": "random",
    "popularity": "popularity",
    "itemknn": "item-kNN",
    "als": "ALS",
    "als_mmr_fixed": "ALS + MMR (fixed lambda)",
    "ours": "ALS + personalised lambda (ours)",
}


def config_hash(config: dict[str, Any]) -> str:
    """A stable hash of the config, so a report can be tied to the settings that made it."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def git_commit() -> str | None:
    """The current commit, or None outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def sidecar_stats(processed_dir: str) -> dict[str, Any]:
    """Collect the pipeline sidecars so a report carries its own data provenance."""
    stats: dict[str, Any] = {}
    for path in sorted(Path(processed_dir).glob("*.stats.json")):
        stats[path.name.replace(".stats.json", "")] = json.loads(path.read_text(encoding="utf-8"))
    return stats


def build(results: dict[str, Any], config: dict[str, Any], config_path: str) -> dict[str, Any]:
    """Assemble the full report payload."""
    now = datetime.now(UTC)
    run_id = hashlib.sha256(f"{now.isoformat()}{config_hash(config)}".encode()).hexdigest()[:8]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": now.isoformat(),
        "config_path": config_path,
        "config_name": config.get("name"),
        "config_hash": config_hash(config),
        "config": config,
        "git_commit": git_commit(),
        "split": results["split"],
        "users_evaluated": results["users_evaluated"],
        "cold_users": results["cold_users"],
        "segment_sizes": results["segment_sizes"],
        "explorer_score_mean": results["explorer_score_mean"],
        "fixed_lambda": results["fixed_lambda"],
        "k_values": config["eval"]["k_values"],
        "models": results["models"],
        "lambda_sweep": results["lambda_sweep"],
        "confidence_intervals": None,
        "data_sidecars": sidecar_stats(config["data"]["processed_dir"]),
    }


def fmt(value: Any) -> str:
    """Format a metric for a Markdown table, marking the ones that do not apply."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def metric_table(payload: dict[str, Any], k: int) -> list[str]:
    """One Markdown table of every model against every metric at k."""
    columns = [
        f"ndcg@{k}",
        f"recall@{k}",
        f"discovery_recall@{k}",
        f"novelty@{k}",
        f"ild@{k}",
        f"serendipity@{k}",
        f"familiarity_anchor_rate@{k}",
        f"catalogue_coverage@{k}",
        f"exposure_gini@{k}",
    ]
    short = [c.replace(f"@{k}", "") for c in columns]
    lines = [f"### Metrics @{k}", "", "| model | " + " | ".join(short) + " |"]
    lines.append("|---|" + "---|" * len(columns))
    for name in MODEL_ORDER:
        model = payload["models"].get(name)
        if not model:
            continue
        row = " | ".join(fmt(model.get(c)) for c in columns)
        lines.append(f"| {MODEL_LABELS[name]} | {row} |")
    lines.append("")
    return lines


def frontier_table(payload: dict[str, Any]) -> list[str]:
    """The lambda sweep, which is the headline chart in the report and the UI."""
    k = max(payload["k_values"])
    lines = [
        "### Accuracy versus novelty frontier",
        "",
        f"Fixed lambda sweep at K={k}. Our personalised model sits as a point on the same axes.",
        "",
        f"| lambda | ndcg@{k} | novelty@{k} | discovery_recall@{k} |",
        "|---|---|---|---|",
    ]
    for point in payload["lambda_sweep"]:
        lines.append(
            f"| {point['lambda']:.2f} | {fmt(point[f'ndcg@{k}'])} | "
            f"{fmt(point[f'novelty@{k}'])} | {fmt(point[f'discovery_recall@{k}'])} |"
        )
    ours = payload["models"].get("ours", {})
    lines.append(
        f"| **ours** | **{fmt(ours.get(f'ndcg@{k}'))}** | "
        f"**{fmt(ours.get(f'novelty@{k}'))}** | **{fmt(ours.get(f'discovery_recall@{k}'))}** |"
    )
    lines.append("")
    return lines


def segment_table(payload: dict[str, Any]) -> list[str]:
    """Our model broken down by explorer tercile, plus cold-ish users."""
    k = max(payload["k_values"])
    segments = payload["models"].get("ours", {}).get(f"segments@{k}", {})
    lines = [
        f"### Our model by segment @{k}",
        "",
        f"| segment | users | ndcg@{k} | discovery_recall@{k} | novelty@{k} | anchor rate |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("loyalists", "middle", "explorers", "cold"):
        seg = segments.get(name, {})
        lines.append(
            f"| {name} | {seg.get('users', 0)} | {fmt(seg.get(f'ndcg@{k}'))} | "
            f"{fmt(seg.get(f'discovery_recall@{k}'))} | {fmt(seg.get(f'novelty@{k}'))} | "
            f"{fmt(seg.get(f'familiarity_anchor_rate@{k}'))} |"
        )
    lines.append("")
    return lines


def verdict(payload: dict[str, Any]) -> list[str]:
    """State plainly whether the project's claim held on this run."""
    k = max(payload["k_values"])
    ours = payload["models"].get("ours", {})
    als = payload["models"].get("als", {})
    popularity = payload["models"].get("popularity", {})

    lines = ["### Did the claim hold?", ""]
    claim = "better discovery without wrecking relevance"
    ours_ndcg, ours_disc = ours.get(f"ndcg@{k}"), ours.get(f"discovery_recall@{k}")
    als_ndcg, als_disc = als.get(f"ndcg@{k}"), als.get(f"discovery_recall@{k}")
    pop_ndcg = popularity.get(f"ndcg@{k}")

    lines.append(f'The claim under test is "{claim}".')
    lines.append("")
    if None in (ours_ndcg, ours_disc, als_ndcg, als_disc):
        lines.append("- Not enough data on this run to judge the claim.")
    else:
        disc_delta = ours_disc - als_disc
        ndcg_delta = ours_ndcg - als_ndcg
        lines.append(
            f"- Discovery recall@{k} versus plain ALS: {disc_delta:+.4f} "
            f"({fmt(als_disc)} to {fmt(ours_disc)})."
        )
        lines.append(
            f"- NDCG@{k} versus plain ALS: {ndcg_delta:+.4f} ({fmt(als_ndcg)} to {fmt(ours_ndcg)})."
        )
    if ours_ndcg is not None and pop_ndcg is not None:
        beat = "above" if ours_ndcg > pop_ndcg else "at or below"
        lines.append(f"- Against the popularity baseline on NDCG@{k}: {beat} it.")
    lines.append("")
    return lines


def to_markdown(payload: dict[str, Any]) -> str:
    """Render the Markdown summary."""
    lines = [
        f"# Detour eval report {payload['run_id']}",
        "",
        f"- generated: {payload['generated_at']}",
        f"- split: **{payload['split']}**",
        f"- config: `{payload['config_path']}` (hash `{payload['config_hash']}`)",
        f"- git commit: `{payload['git_commit'] or 'not a git repo'}`",
        f"- users evaluated: {payload['users_evaluated']} (cold-ish: {payload['cold_users']})",
        f"- segment sizes: {payload['segment_sizes']}",
        f"- fixed global lambda chosen on this split: {payload['fixed_lambda']:.2f}",
        "",
    ]
    lines += verdict(payload)
    for k in payload["k_values"]:
        lines += metric_table(payload, k)
    lines += frontier_table(payload)
    lines += segment_table(payload)
    lines += [
        "### Notes and limitations",
        "",
        "- Numbers come from synthetic fixture data unless the config points elsewhere.",
        "- Bootstrap confidence intervals are not computed yet; `confidence_intervals` is null.",
        "- ILD uses ALS item factors; item2vec embeddings are not wired in yet.",
        "- Metrics return n/a for users the metric does not apply to, rather than 0.",
        "",
    ]
    return "\n".join(lines)


def write(
    results: dict[str, Any], config: dict[str, Any], config_path: str = "evals/configs/smoke.yaml"
) -> str:
    """Write the JSON and Markdown report and return the JSON path."""
    payload = build(results, config, config_path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"{datetime.now(UTC).date().isoformat()}_{payload['run_id']}"
    json_path = REPORTS_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (REPORTS_DIR / f"{stem}.md").write_text(to_markdown(payload), encoding="utf-8")
    return str(json_path)


def latest() -> dict[str, Any] | None:
    """The most recent report JSON, or None if no eval has been run.

    Sorted by the generated_at inside each report, not by filename: the name ends in a
    random run id, so lexicographic order is not chronological.
    """
    payloads: list[dict[str, Any]] = []
    for path in REPORTS_DIR.glob("*.json"):
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    if not payloads:
        return None
    return max(payloads, key=lambda report: str(report.get("generated_at", "")))


# --- map evaluation --------------------------------------------------------------------

MAP_MODEL_LABELS = {
    "random": "random",
    "popularity": "popularity only",
    "als": "ALS factors",
    "item2vec": "item2vec",
    "ppmi_svd": "PPMI + SVD (ours)",
}

# In a popularity-only space the popularity-matched negatives are, by construction, at the
# same place as the target, so the control returns 1.0 and means nothing. Reporting it as a
# score would be misleading.
NO_MATCHED_CONTROL = {"popularity"}


def map_verdict(payload: dict[str, Any]) -> list[str]:
    """State plainly whether the map beat chance and, more importantly, popularity."""
    models = payload["models"]
    ours = models.get("ppmi_svd", {})
    popularity = models.get("popularity", {})

    lines = ["### Did the map learn anything?", ""]
    percentile = ours.get("adoption_percentile")
    matched = ours.get("matched_percentile")
    pop_percentile = popularity.get("adoption_percentile")

    if percentile is None:
        lines += ["- Nothing scorable on this run.", ""]
        return lines

    lines.append(
        f"- Adoption rank percentile for the map: **{percentile:.4f}** "
        f"(0.5 is chance, lower is better)."
    )
    if pop_percentile is not None:
        verdict = "beats" if percentile < pop_percentile else "does NOT beat"
        lines.append(
            f"- Against the popularity-only baseline ({pop_percentile:.4f}): **{verdict}** it."
        )
    if matched is not None:
        survived = "survives" if matched < 0.5 else "does NOT survive"
        lines.append(
            f"- Popularity-matched control: **{matched:.4f}**. The result **{survived}** "
            "the control, which compares each real adoption only against artists of "
            "similar global popularity."
        )
        if matched >= 0.5:
            lines.append(
                "  - This means the ranking is explained by popularity, not by taste. "
                "Whatever the headline number says, the map has not been shown to work."
            )
    lines.append("")
    return lines


def map_table(payload: dict[str, Any]) -> list[str]:
    """Every model against every adoption metric."""
    k_values = payload["k_values"]
    columns = ["adoption_percentile", "matched_percentile", "median_rank"] + [
        f"hit@{k}" for k in k_values
    ]
    header = ["percentile", "matched", "median rank"] + [f"hit@{k}" for k in k_values]

    lines = ["### Every model", "", "| model | " + " | ".join(header) + " |"]
    lines.append("|---|" + "---|" * len(columns))
    for name, label in MAP_MODEL_LABELS.items():
        model = payload["models"].get(name)
        if not model:
            continue
        row = " | ".join(
            "not meaningful"
            if column == "matched_percentile" and name in NO_MATCHED_CONTROL
            else fmt(model.get(column))
            for column in columns
        )
        lines.append(f"| {label} | {row} |")
    lines.append("")
    return lines


def build_map_report(
    results: dict[str, Any], config: dict[str, Any], config_path: str
) -> dict[str, Any]:
    """Assemble the map evaluation payload."""
    now = datetime.now(UTC)
    run_id = hashlib.sha256(f"{now.isoformat()}{config_hash(config)}".encode()).hexdigest()[:8]
    return {
        "schema_version": 2,
        "report_kind": "map",
        "run_id": run_id,
        "generated_at": now.isoformat(),
        "config_path": config_path,
        "config_hash": config_hash(config),
        "config": config,
        "git_commit": git_commit(),
        "split": results["split"],
        "listeners_with_adoptions": results["listeners_with_adoptions"],
        "vocabulary": results["vocabulary"],
        "k_values": results["k_values"],
        "models": results["models"],
        "confidence_intervals": None,
        "data_sidecars": sidecar_stats(config["data"]["processed_dir"]),
    }


def map_to_markdown(payload: dict[str, Any]) -> str:
    """Render the map report."""
    lines = [
        f"# Detour map report {payload['run_id']}",
        "",
        f"- generated: {payload['generated_at']}",
        f"- split: **{payload['split']}**",
        f"- config: `{payload['config_path']}` (hash `{payload['config_hash']}`)",
        f"- git commit: `{payload['git_commit'] or 'not a git repo'}`",
        f"- listeners with a real adoption: {payload['listeners_with_adoptions']:,}",
        f"- artists in the map: {payload['vocabulary']:,}",
        "",
        "The question: a listener is placed from their training history alone, every artist is",
        "ranked by distance from that position, and we check where the artists they really went",
        "on to play landed. 0.5 is chance.",
        "",
    ]
    lines += map_verdict(payload)
    lines += map_table(payload)
    lines += [
        "### Limitations",
        "",
        "- Confidence intervals are not computed yet, so small differences are not meaningful.",
        "- Listening data is from 2011; tastes and catalogues have moved on.",
        "- Artists outside the training vocabulary cannot be ranked and are excluded.",
        "",
    ]
    return "\n".join(lines)


def write_map_report(
    results: dict[str, Any], config: dict[str, Any], config_path: str = "evals/configs/main.yaml"
) -> str:
    """Write the map report JSON and Markdown, returning the JSON path."""
    payload = build_map_report(results, config, config_path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    stem = f"{datetime.now(UTC).date().isoformat()}_map_{payload['run_id']}"
    json_path = REPORTS_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (REPORTS_DIR / f"{stem}.md").write_text(map_to_markdown(payload), encoding="utf-8")
    return str(json_path)
