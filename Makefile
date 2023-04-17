#!make
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# Github: https://github.com/btschwertfeger

VENV := venv
GLOBAL_PYTHON := $(shell which python3)
PYTHON := $(VENV)/bin/python3

.PHONY := build dev install test test_upload live_upload clean

##		Builds the python-kraken-sdk
##
build:
	$(PYTHON) -m pip wheel -w dist --no-deps .

##		Installs the package in edit mode
##
dev:
	$(PYTHON) -m pip install -e ".[dev]"

##		Install the package
##
install:
	$(PYTHON) -m pip install .

##		Run the unittests
##
test:
	$(PYTHON) -m pytest tests/

##		Build the documentation
##
doc:
	cd docs && make html

##		Run the documentation tests
##
doctest:
	cd docs && make doctest

##		Pre-Commit
pre-commit:
	@pre-commit run -a

## 		Create the changelog
##
changelog:
	docker run -it --rm \
		-v "$(pwd)":/usr/local/src/python-cmethods \
		githubchangeloggenerator/github-changelog-generator \
		-u btschwertfeger \
		-p python-cmethods \
		-t $(GHTOKEN)  \
		--breaking-labels Breaking \
		--enhancement-labels Feature

##		Clean the workspace
##
clean:
	rm -rf .pytest_cache build/ dist/ python_cmethods.egg-info docs/_build examples/.ipynb_checkpoints
	rm -f .coverage cmethods/_version.py

	find tests -name "__pycache__" | xargs rm -rf
	find cmethods -name "__pycache__" | xargs rm -rf
	find examples -name "__pycache__" | xargs rm -rf
