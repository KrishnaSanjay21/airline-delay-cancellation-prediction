.PHONY: install lint test pipeline dashboard

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests app.py

test:
	pytest

pipeline:
	airline-risk run-all

dashboard:
	streamlit run app.py

