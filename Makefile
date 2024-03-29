#!make
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

VENV := venv
PYTHON := $(VENV)/bin/python3
TESTS := tests
PYTEST_OPTS := -vv

.PHONY: help
help:
	@grep "^##" Makefile | sed -e "s/##//"

## build		Builds python-cmethods
##
.PHONY: build
build:
	$(PYTHON) -m pip wheel -w dist --no-deps .

## dev		Installs the package in edit mode
##
.PHONY: dev
dev:
	@git lfs install
	$(PYTHON) -m pip install -e ".[dev]"

## install		Install the package
##
.PHONY: install
install:
	$(PYTHON) -m pip install .

## test		Run the unit tests
##
.PHONY: test
test:
	$(PYTHON) -m pytest $(PYTEST_OPTS) $(TESTS)

.PHONY: tests
tests: test

## wip  	Run tests marked as wip
##
.PHONY: wip
wip:
	$(PYTHON) -m pytest $(PYTEST_OPTS) -m "wip" $(TESTS)

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

## pre-commit		Pre-Commit
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

## changelog		Create the changelog
##
.PHONY: changelog
changelog:
	docker run -it --rm \
		-v "$(PWD)":/usr/local/src/your-app/ \
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

# for file in `ls .github/workflows`; do sed -i '' 's/actions\/checkout@v4/actions\/checkout@1e31de5234b9f8995739874a8ce0492dc87873e2  # v4.0.0/g' .github/workflows/$file; done
# for file in `ls .github/workflows`; do sed -i '' 's/actions\/setup-python@v5/actions\/setup-python@0a5c61591373683505ea898e09a3ea4f39ef2b9c  # v5.0.0/g' .github/workflows/$file; done
