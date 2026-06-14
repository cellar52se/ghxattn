.PHONY: help install dev lint type test smoke docker clean

help:
	@echo "install  install runtime package"
	@echo "dev      install with dev extras"
	@echo "lint     ruff + black + isort checks"
	@echo "type     mypy strict"
	@echo "test     pytest"
	@echo "smoke    two-step training smoke on synthetic cohort"
	@echo "docker   build the container image"
	@echo "clean    remove caches and build artifacts"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	mypy ghxattn

test:
	pytest -q

smoke:
	ghxattn weave experiment=_smoke

docker:
	docker build -t ghxattn:latest .

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
