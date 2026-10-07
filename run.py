"""Run the project.

    uv run python run.py setup     # install Python and web dependencies
    uv run python run.py demo      # pipeline, then API and web together
    uv run python run.py --help    # everything else

`demo` is the one to reach for: from a fresh clone it builds the fixture pipeline and
serves the discovery dial. Nothing here needs credentials or network access, because the
pipeline runs on synthetic fixture data.

This file, not the Makefile, is the real entry point: `make` is not installed everywhere
the project has to run. The Makefile forwards here so both spellings work.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
ARTIFACTS = ROOT / "artifacts"
CONFIG = "evals/configs/main.yaml"
API_PORT = 8000
WEB_PORT = 5173

# implicit prints a warning and oversubscribes threads without this.
ENV = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "PYTHONIOENCODING": "utf-8"}

PIPELINE_ARTEFACTS = [
    ARTIFACTS / "artist_vectors.parquet",
    ARTIFACTS / "map_2d.parquet",
]
RAW_LISTENS = ROOT / "data" / "raw" / "listens.parquet"


def resolve(command: list[str]) -> list[str]:
    """Expand the executable to a full path, so Windows finds pnpm.CMD and friends."""
    found = shutil.which(command[0], path=ENV.get("PATH"))
    return [found, *command[1:]] if found else command


def run(command: list[str], check: bool = True) -> int:
    """Run a command in the project root, streaming its output."""
    print(f"\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(resolve(command), cwd=ROOT, env=ENV, shell=False)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode


def python(*args: str) -> None:
    """Run a module inside the project environment."""
    run(["uv", "run", "python", *args])


def has_web() -> bool:
    """Whether the frontend has been scaffolded yet."""
    return (WEB / "package.json").exists()


def pipeline_is_built() -> bool:
    """Whether the artefacts the API needs already exist."""
    return all(path.exists() for path in PIPELINE_ARTEFACTS)


def cmd_setup(_: argparse.Namespace) -> None:
    """Install Python dependencies, and web dependencies if the frontend exists."""
    run(["uv", "sync"])
    if has_web():
        run(["corepack", "enable", "pnpm"], check=False)
        run(["pnpm", "--dir", "web", "install"])
    else:
        print("\nweb/ not scaffolded yet. Create it with:")
        print("  uv run python scripts/create_project.py --with-web")


def cmd_fetch(args: argparse.Namespace) -> None:
    """Download real listening history from MLHD+, streamed and resumable."""
    python(
        "-m",
        "detour.ingest.mlhd_loader",
        "--users",
        str(args.users),
        "--from",
        args.start_month,
        "--months",
        "12",
    )


def cmd_lastfm(args: argparse.Namespace) -> None:
    """Import your own listening history from Last.fm."""
    if not args.user:
        print("Which Last.fm user? Pass --user YOURNAME")
        print("You also need LASTFM_API_KEY in .env (see .env.example).")
        sys.exit(1)
    python("-m", "detour.ingest.lastfm", "--user", args.user)


def cmd_pipeline(args: argparse.Namespace) -> None:
    """clean -> split -> map -> 2d projection."""
    if not RAW_LISTENS.exists():
        print("No listening data yet; fetching it first.")
        cmd_fetch(args)
    python("-m", "detour.clean.run", "--source", "data/raw/listens.parquet")
    python("-m", "detour.split", "--config", args.config)
    python("-m", "detour.models.artist_map", "--config", args.config)
    python("-m", "detour.models.project", "--config", args.config)
    print("\npipeline done. The map is in artifacts/")


def cmd_eval(args: argparse.Namespace) -> None:
    """Run the eval harness and write a report."""
    if not pipeline_is_built():
        print("No artefacts yet; building the pipeline first.")
        cmd_pipeline(args)
    python("-m", "detour.eval.map_harness", "--config", args.config, "--split", args.split)


def cmd_api(args: argparse.Namespace) -> None:
    """Serve the FastAPI app."""
    if not pipeline_is_built():
        print("No artefacts yet; building the pipeline first.")
        cmd_pipeline(args)
    run(["uv", "run", "uvicorn", "api.main:app", "--reload", "--port", str(API_PORT)])


def cmd_web(_: argparse.Namespace) -> None:
    """Serve the Vite dev server."""
    if not has_web():
        print("web/ not scaffolded yet. Create it with:")
        print("  uv run python scripts/create_project.py --with-web")
        sys.exit(1)
    run(["pnpm", "--dir", "web", "dev"])


def cmd_test(_: argparse.Namespace) -> None:
    """Run the test suite."""
    run(["uv", "run", "pytest"])


def cmd_lint(_: argparse.Namespace) -> None:
    """Lint, format-check and type-check."""
    run(["uv", "run", "ruff", "check", "."])
    run(["uv", "run", "ruff", "format", "--check", "."])
    run(["uv", "run", "mypy", "detour", "api"])


def spawn(command: list[str], label: str) -> subprocess.Popen[bytes]:
    """Start a long-running child process."""
    print(f"  {label}: {' '.join(command)}")
    return subprocess.Popen(resolve(command), cwd=ROOT, env=ENV, shell=False)


def cmd_demo(args: argparse.Namespace) -> None:
    """Build the pipeline if needed, then serve the API and the frontend together."""
    if not pipeline_is_built():
        cmd_pipeline(args)
    else:
        print("Artefacts already built. Delete artifacts/ to rebuild.")

    print("\nstarting:")
    children: list[subprocess.Popen[bytes]] = [
        spawn(
            ["uv", "run", "uvicorn", "api.main:app", "--reload", "--port", str(API_PORT)],
            "api",
        )
    ]
    if has_web():
        children.append(spawn(["pnpm", "--dir", "web", "dev"], "web"))
    else:
        print("  web: not scaffolded, skipping")

    print(f"\n  API   http://127.0.0.1:{API_PORT}/docs")
    if has_web():
        print(f"  app   http://127.0.0.1:{WEB_PORT}")
    print("\nCtrl-C to stop.")

    try:
        while all(child.poll() is None for child in children):
            time.sleep(0.4)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()


COMMANDS = {
    "setup": cmd_setup,
    "fetch": cmd_fetch,
    "lastfm": cmd_lastfm,
    "pipeline": cmd_pipeline,
    "eval": cmd_eval,
    "api": cmd_api,
    "web": cmd_web,
    "demo": cmd_demo,
    "test": cmd_test,
    "lint": cmd_lint,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=sorted(COMMANDS),
        help="what to run (default: demo)",
    )
    parser.add_argument("--config", default=CONFIG, help=f"eval config (default: {CONFIG})")
    parser.add_argument(
        "--split", default="val", choices=["val", "test"], help="eval split (default: val)"
    )
    parser.add_argument("--users", type=int, default=2000, help="listeners to fetch")
    parser.add_argument("--from", dest="start_month", default="2011-01", help="YYYY-MM")
    parser.add_argument("--user", default=None, help="your Last.fm username")
    args = parser.parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
