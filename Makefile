RAW ?= data/raw/Online Retail.xlsx

.PHONY: install data run report test app all

install:
	pip install -e ".[app,dev]"

data:
	python scripts/fetch_data.py

run:
	python -m pricesim.pipeline --raw "$(RAW)"

report:
	python -m pricesim.report

test:
	pytest -q

app:
	streamlit run app/streamlit_app.py

all: data run report test
