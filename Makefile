.PHONY: help install install-dev test test-cov lint typecheck format check clean build docker-build docker-run docker-stop docs

# Default target
help:
	@echo "pdf-ingest Development Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make install-pre   Install pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run tests"
	@echo "  make test-cov      Run tests with coverage"
	@echo "  make test-watch    Run tests in watch mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run ruff linter"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make format        Format code with ruff"
	@echo "  make check         Run all checks (lint, typecheck, test)"
	@echo ""
	@echo "Building:"
	@echo "  make build         Build package"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  Build Docker images"
	@echo "  make docker-run    Run docker-compose"
	@echo "  make docker-stop   Stop docker-compose"
	@echo "  make docker-logs   View docker logs"
	@echo ""
	@echo "Development:"
	@echo "  make run-cli       Run CLI help"
	@echo "  make run-web       Run Streamlit web UI"
	@echo "  make run-api       Run FastAPI server"
	@echo "  make run-pipeline  Run ingestion pipeline demo"

# Setup
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-pre:
	pre-commit install

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=pdf_ingest --cov-report=term-missing --cov-report=html

test-watch:
	pytest-watch tests/ -- -v

# Code Quality
lint:
	ruff check .

typecheck:
	mypy src/

format:
	ruff format .

check: lint typecheck test

# Building
build: clean
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .mypy_cache/ htmlcov/ .coverage coverage.xml bandit-report.json

# Docker
docker-build:
	docker-compose build

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart: docker-stop docker-run

# Development
run-cli:
	pdf-ingest --help

run-web:
	streamlit run web/app.py

run-api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-pipeline:
	python -m pdf_ingest.cli ingest documents/sample.pdf --output ./data/test_vs

# Demo with sample data
demo:
	@echo "Running demo pipeline..."
	pdf-ingest ingest documents/sample.pdf
	pdf-ingest chat "What is this document about?"
	pdf-ingest search "test"
	pdf-ingest summarize

# Code generation
generate-diagrams:
	python scripts/make_diagrams.py

# Security
security-scan:
	bandit -r src/ -f json -o bandit-report.json
	safety check

# Release
version-patch:
	bump2version patch

version-minor:
	bump2version minor

version-major:
	bump2version major

release: check build
	@echo "Ready to release. Tag with: git tag v\$(cat pyproject.toml | grep version | head -1 | cut -d'\"' -f2)"