# make is not installed everywhere. run.py is the real entry point; these forward to it.
.PHONY: setup pipeline eval eval-smoke eval-final api web demo test lint

setup:
	uv run python run.py setup

pipeline:
	uv run python run.py pipeline

eval:
	uv run python run.py eval --config evals/configs/main.yaml

eval-smoke:
	uv run python run.py eval --config evals/configs/smoke.yaml

eval-final:
	uv run python run.py eval --config evals/configs/main.yaml --split test

api:
	uv run python run.py api

web:
	uv run python run.py web

demo:
	uv run python run.py demo

test:
	uv run pytest

lint:
	uv run ruff check . && uv run ruff format --check . && uv run mypy detour
