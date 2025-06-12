# -*- mode: make; coding: utf-8 -*-
#!make
#
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# All rights reserved.
# https://github.com/btschwertfeger
#

UV := uv
PYTHON := python
PYTEST := $(UV) run pytest
TESTS := tests
PYTEST_OPTS := -vv --junit-xml=pytest.xml
PYTEST_COV_OPTS := $(PYTEST_OPTS) --cov=cmethods --cov-report=xml:coverage.xml --cov-report=term

## ======= M A K E F I L E - T A R G E T S =====================================
## help           Show this help message
##
.PHONY: help
help:
	@grep "^##" Makefile | sed -e "s/##//"

## ======= B U I L D I N G =====================================================
## build		Builds the package
##
.PHONY: build
build:
	$(PYTHON) -m build .

## rebuild 	Rebuild the package
##
.PHONY: rebuild
rebuild: clean build

## ======= I N S T A L L A T I O N =============================================
## install	Install the package
##
.PHONY: install
install: check-uv
	$(UV) pip install .

## dev		Installs the package in edit mode
##
.PHONY: dev
dev:
	@git lfs install
	$(UV) pip install -e ".[dev,test,jupyter,examples]" -r doc/requirements.txt

## ======= T E S T I N G =======================================================
## test		Run the unit tests
##
.PHONY: test
test:
	$(PYTHON) -m pytest $(PYTEST_OPTS) $(TESTS)

.PHONY: test
tests: test

## retest 	Rerun tests that failed before
##
.PHONY: retest
retest:
	$(PYTHON) -m pytest $(PYTEST_OPTS) --lf $(TESTS)

## wip  	Run tests marked as wip
##
.PHONY: wip
wip:
	$(PYTHON) -m pytest $(PYTEST_OPTS) -m "wip" $(TESTS)

## coverage       Run all tests and generate the coverage report
##
.PHONY: coverage
coverage:
	$(PYTEST) $(PYTEST_COV_OPTS) $(TEST_DIR)

## doc		Build the documentation
##
.PHONY: doc
doc:
	cd doc && make html

## doctest		Run the documentation tests
##
.PHONY: doctest
doctest:
	cd doc && make doctest

## ======= M I S C E L A N I O U S =============================================
## pre-commit	Run the pre-commit targets
##
.PHONY: pre-commit
pre-commit:
	@pre-commit run -a

## ruff 	Run ruff without fix
.PHONY: ruff
ruff:
	ruff check --preview .

## ruff-fix 	Run ruff with fix
##
.PHONY: ruff-fix
ruff-fix:
	ruff check --fix --preview .
	ruff format .

## changelog		Create the changelog
##
.PHONY: changelog
changelog:
	docker run -it --rm \
		-v $(PWD):/usr/local/src/your-app/ \
		githubchangeloggenerator/github-changelog-generator \
		-u btschwertfeger \
		-p python-cmethods \
		-t $(GHTOKEN)  \
		--breaking-labels Breaking \
		--enhancement-labels Feature

## clean		Clean the workspace
##
.PHONY: clean
clean:
	rm -rf .mypy_cache .pytest_cache .cache \
		build/ dist/ python_cmethods.egg-info \
		docs/_build \
		examples/.ipynb_checkpoints .ipynb_checkpoints \
		doc/_build

	rm -f .coverage cmethods/_version.py

	find tests -name "__pycache__" | xargs rm -rf
	find cmethods -name "__pycache__" | xargs rm -rf
	find examples -name "__pycache__" | xargs rm -rf

## check-uv       Check if uv is installed
##
.PHONY: check-uv
check-uv:
	@if ! command -v $(UV) >/dev/null; then \
		echo "Error: uv is not installed. Please visit https://github.com/astral-sh/uv for installation instructions."; \
		exit 1; \
	fi
