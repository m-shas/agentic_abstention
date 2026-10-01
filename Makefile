.PHONY: install install-torch test lint fmt typecheck serve init-db simulate \
        train-evaluator train-distiller extract-playbooks clean docker

PY  ?= python
PIP ?= $(PY) -m pip

install:
	$(PIP) install -e ".[dev]"

install-torch:
	$(PIP) install -e ".[dev,torch]"

test:
	$(PY) -m pytest -q

test-cov:
	$(PY) -m pytest --cov=convolve --cov-report=term-missing

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff check --fix src tests

typecheck:
	$(PY) -m mypy src

serve:
	$(PY) -m convolve.cli serve

init-db:
	$(PY) -m convolve.cli init-db

simulate:
	$(PY) -m convolve.cli simulate --n 200

train-evaluator:
	$(PY) -m convolve.cli train-evaluator

train-distiller:
	$(PY) -m convolve.cli train-distiller

extract-playbooks:
	$(PY) -m convolve.cli extract-playbooks

docker:
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
