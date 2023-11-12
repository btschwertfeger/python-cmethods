#!make
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger

VENV := venv
GLOBAL_PYTHON := $(shell which python3)
PYTHON := $(VENV)/bin/python3
TESTS := tests
PYTEST_OPTS := -vv

.PHONY := build dev install test tests doc doctest pre-commit changelog clean

## build		Builds the python-kraken-sdk
##
build:
	$(PYTHON) -m pip wheel -w dist --no-deps .

## dev		Installs the package in edit mode
##
dev:
	$(PYTHON) -m pip install -e ".[dev]"

## install		Install the package
##
install:
	$(PYTHON) -m pip install .

## test		Run the unit tests
##
test:
	$(PYTHON) -m pytest $(PYTEST_OPTS) $(TESTS)
tests: test

## wip  	Run tests marked as wip
##
wip:
	$(PYTHON) -m pytest $(PYTEST_OPTS)  -m "wip" $(TESTS)

## doc		Build the documentation
##
doc:
	cd docs && make html

## doctest		Run the documentation tests
##
doctest:
	cd docs && make doctest

## pre-commit		Pre-Commit
pre-commit:
	@pre-commit run -a

## changelog		Create the changelog
##
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
clean:
	rm -rf .pytest_cache \
		build/ dist/ python_cmethods.egg-info \
		docs/_build \
		examples/.ipynb_checkpoints .ipynb_checkpoints \
		.mypy_cache .pytest_cache

	rm -f .coverage cmethods/_version.py

	find tests -name "__pycache__" | xargs rm -rf
	find cmethods -name "__pycache__" | xargs rm -rf
	find examples -name "__pycache__" | xargs rm -rf
