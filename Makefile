.PHONY: install run test lint clean docker-build docker-run build-exe

# ── Development ────────────────────────────────────────────────────

install:
	pip install -r requirements.txt
	pip install scikit-learn>=1.3.0 joblib>=1.3.0 psutil>=5.9.0
	pip install arabic-reshaper>=2.1.0 python-bidi>=0.4.0

install-all:
	pip install -e ".[all]"

run:
	python src/main.py

test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/

format:
	black src/ tests/

typecheck:
	python -m mypy src/ --ignore-missing-imports || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info

# ── Docker ─────────────────────────────────────────────────────────

docker-build:
	docker compose build

docker-run:
	docker compose up

docker-training:
	docker compose --profile training up

# ── Packaging ──────────────────────────────────────────────────────

build-exe:
	pyinstaller --name "ArSL Overlay" \
		--onefile \
		--windowed \
		--add-data "config:config" \
		--add-data "src:src" \
		--hidden-import PySide6.QtCharts \
		--hidden-import mediapipe \
		src/main.py

build-exe-dir:
	pyinstaller --name "ArSL Overlay" \
		--onedir \
		--windowed \
		--add-data "config:config" \
		--add-data "src:src" \
		--hidden-import PySide6.QtCharts \
		--hidden-import mediapipe \
		src/main.py

# ── Training ───────────────────────────────────────────────────────

train-static:
	python -m src.models.train_static

train-dynamic:
	python -m src.models.train_dynamic

# ── Dataset ────────────────────────────────────────────────────────

dataset-stats:
	python -c "from src.infrastructure.dataset.organizer import DatasetOrganizer; o=DatasetOrganizer(); print(o.class_statistics('iraqi'))"
