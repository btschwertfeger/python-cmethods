# -*- mode: make; coding: utf-8 -*-
#!make
#
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# All rights reserved.
# https://github.com/btschwertfeger
#

VENV := venv
PYTHON := python
TESTS := tests
PYTEST_OPTS := -vv
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: help
help:
	@grep "^##" Makefile | sed -e "s/##//"

## build		Builds python-cmethods
##
.PHONY: build
build:
	$(PYTHON) -m build .

## dev		Installs the package in edit mode
##
.PHONY: dev
dev:
	@git lfs install
	$(PYTHON) -m pip install -e ".[dev,test,jupyter,examples]" -r doc/requirements.txt

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
	ruff format .

## changelog		Create the changelog
##
.PHONY: changelog
changelog:
	docker run -it --rm \
		-v $(ROOT_DIR):/usr/local/src/your-app/ \
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
