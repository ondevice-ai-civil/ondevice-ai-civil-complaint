# Makefile for GovOn Project

.PHONY: help install dev-install lint format test docker-build docker-run clean

PYTHON := python3
PIP := $(PYTHON) -m pip

help:
	@echo "Available commands:"
	@echo "  install      : Install dependencies"
	@echo "  dev-install  : Install development dependencies"
	@echo "  lint         : Run linting checks (black, isort, flake8, mypy)"
	@echo "  format       : Format code (black, isort)"
	@echo "  test         : Run tests (pytest)"
	@echo "  docker-build : Build docker image"
	@echo "  docker-run   : Run docker container"
	@echo "  clean        : Remove cache files"

install:
	$(PIP) install .

dev-install:
	$(PIP) install .[dev]

lint:
	black --check .
	isort --check .
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	mypy .

format:
	black .
	isort .

test:
	pytest --cov=src --cov-report=term-missing

docker-build:
	docker build -t govon-backend .

docker-run:
	docker compose up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
