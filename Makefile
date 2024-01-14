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
	cd docs && make html

## doctest		Run the documentation tests
##
.PHONY: doctest
doctest:
	cd docs && make doctest

## pre-commit		Pre-Commit
##
.PHONY: pre-commit
pre-commit:
	@pre-commit run -a

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
	rm -rf .pytest_cache .cache \
		build/ dist/ python_cmethods.egg-info \
		docs/_build \
		examples/.ipynb_checkpoints .ipynb_checkpoints \
		.mypy_cache .pytest_cache

	rm -f .coverage cmethods/_version.py

	find tests -name "__pycache__" | xargs rm -rf
	find cmethods -name "__pycache__" | xargs rm -rf
	find examples -name "__pycache__" | xargs rm -rf
