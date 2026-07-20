.PHONY: install test bench clean

install:
	pip install -e ".[dev]"

test:
	pytest -q

bench:
	python -m forgetting_bench.experiment

clean:
	rm -f results/*.png results/summary.*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
