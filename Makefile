.PHONY: install install-dev lint format typecheck test docker-build docker-run clean synthetic-data preprocess train evaluate app

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check src tests app scripts

format:
	black src tests app scripts
	ruff check --fix src tests app scripts

typecheck:
	mypy src

test:
	pytest

docker-build:
	docker build -t hydrosense:latest .

docker-run:
	docker compose up --build

synthetic-data:
	python scripts/generate_synthetic_dataset.py

preprocess:
	python -m src.preprocessing.run \
		--input_dir data/synthetic \
		--output_dir data/processed \
		--sr 16000 --segment_length 10.0 --overlap 0.5

train:
	python -m src.training.train \
		--model hydrosense_base --representation mel \
		--folds 2 --epochs 5 --batch_size 8 \
		--output_dir runs/hydrosense_base_mel

evaluate:
	python -m src.evaluation.evaluate \
		--checkpoint runs/hydrosense_base_mel/best.ckpt \
		--test_split data/splits/test.csv \
		--output_dir results/hydrosense_base_mel

app:
	streamlit run app/streamlit_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
