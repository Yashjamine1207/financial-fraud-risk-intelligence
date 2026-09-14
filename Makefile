.PHONY: install test lint format format-check check

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format:
	black .

format-check:
	black --check .

check:
	ruff check .
	black --check .
	pytest