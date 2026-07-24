PYTHON ?= .venv/bin/python

.PHONY: setup inspect search api test evaluate frontend

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r backend/requirements-dev.txt

inspect:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m backend.scripts.inspect_corpus

search:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m backend.scripts.search "$(QUERY)"

api:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m uvicorn backend.main_api:app --reload --port 8000

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q

evaluate:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m backend.scripts.evaluate

frontend:
	npm --prefix frontend run dev
