#* Variables
SHELL := /usr/bin/env bash
PYTHON := python

# Determine OS.
ifeq ($(OS),Windows_NT)
    OS := windows
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS := linux
    endif
    ifeq ($(UNAME_S),Darwin)
        OS := macos
    endif
endif

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org | python3 -

#* Installation
.PHONY: install
install:
	poetry lock -n && poetry export --without-hashes > requirements.txt
	poetry install -n
	-poetry run mypy --install-types --non-interactive ./

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

.PHONY: rclone-install
rclone-install:
ifeq ($(OS),windows)
	@echo "This command is not supported on Windows. Please download rclone from https://rclone.org/downloads/"
else
	sudo -v ; curl https://rclone.org/install.sh | sudo bash
endif

.PHONY: download-all-fragments
download-all-fragments:
	./scripts/download-fragments.sh 1 2 3

.PHONY: download-all-scrolls
download-all-scrolls:
	./scripts/download-scroll-surface-vols.sh 1 2 PHerc1667 PHerc0332

.PHONY: download-monster-segment
download-monster-segment:
	./scripts/download-monster-segment-surface-vols.sh recto verso

#* Formatters
.PHONY: codestyle
codestyle:
	poetry run isort --settings-path pyproject.toml ./
	poetry run black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	poetry run pytest -c pyproject.toml tests/ --cov-report=html --cov=vesuvius_challenge_rnd

test-unit:
	poetry run pytest -m "not fragment_data and not scroll_data" -c pyproject.toml tests/ --cov-report=html --cov=vesuvius_challenge_rnd

.PHONY: check-codestyle
check-codestyle:
	poetry run isort --diff --check-only --settings-path pyproject.toml ./
	poetry run black --diff --check --config pyproject.toml ./

.PHONY: mypy
mypy:
	poetry run mypy --config-file pyproject.toml ./

.PHONY: check-safety
check-safety:
	poetry check

.PHONY: lint
lint: test-unit check-codestyle check-safety

#* Docker
.PHONY: frag-ink-det-gpu-build
frag-ink-det-gpu-build:
	docker build -t frag-ink-det-gpu -f docker/fragment-ink-detection-gpu/Dockerfile .

.PHONY: frag-ink-det-gpu-run
frag-ink-det-gpu-run:
	docker run -it --rm --gpus all -e WANDB_DOCKER=frag-ink-det-gpu frag-ink-det-gpu

.PHONY: scroll-ink-det-gpu-build
scroll-ink-det-gpu-build:
	docker build -t scroll-ink-det-gpu -f docker/scroll-ink-detection-gpu/Dockerfile .

.PHONY: scroll-ink-det-gpu-run
scroll-ink-det-gpu-run:
	docker run -it --rm --gpus all -e WANDB_DOCKER=scroll-ink-det-gpu scroll-ink-det-gpu
