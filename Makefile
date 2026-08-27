.PHONY: install test bench bench-live clean

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q

bench:
	python -m forgetting_bench.experiment

# Live mem0 / Letta -- skipped (not scored) unless SDKs + credentials are present.
bench-live:
	python -m forgetting_bench.experiment --incumbents

clean:
	rm -f results/*.png results/summary.*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
