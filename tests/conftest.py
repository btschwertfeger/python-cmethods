#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""Module providing fixtures for testing."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

import pytest
import xarray as xr
from dask.distributed import LocalCluster

from cmethods import CMethods

from .helper import get_datasets

FIXTURE_DIR: str = os.path.join(os.path.dirname(__file__), "fixture")


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """


def pytest_unconfigure(config):
    """
    called before test process is exited.
    """


@pytest.fixture(scope="session")
def dask_cluster() -> Any:
    # Create a Dask LocalCluster
    cluster = LocalCluster()

    # Create a Dask Client connected to the LocalCluster
    client = cluster.get_client()

    # Yield the client, making it available for the tests
    yield client

    # After the tests are done, close the cluster
    client.close()


@pytest.fixture()
def datasets() -> dict:
    obsh_add, obsp_add, simh_add, simp_add = get_datasets(kind="+")
    obsh_mult, obsp_mult, simh_mult, simp_mult = get_datasets(kind="*")

    return {
        "+": {
            "obsh": obsh_add["+"],
            "obsp": obsp_add["+"],
            "simh": simh_add["+"],
            "simp": simp_add["+"],
        },
        "*": {
            "obsh": obsh_mult["*"],
            "obsp": obsp_mult["*"],
            "simh": simh_mult["*"],
            "simp": simp_mult["*"],
        },
    }


@pytest.fixture()
def cm() -> CMethods:
    return CMethods()


@lru_cache(maxsize=None)
@pytest.fixture()
def datasets_from_zarr() -> dict:
    return {
        "+": {
            "obsh": xr.open_zarr(
                os.path.join(FIXTURE_DIR, "temperature_obsh.zarr")
            ).chunk({"time": -1}),
            "obsp": xr.open_zarr(
                os.path.join(FIXTURE_DIR, "temperature_obsp.zarr")
            ).chunk({"time": -1}),
            "simh": xr.open_zarr(
                os.path.join(FIXTURE_DIR, "temperature_simh.zarr")
            ).chunk({"time": -1}),
            "simp": xr.open_zarr(
                os.path.join(FIXTURE_DIR, "temperature_simp.zarr")
            ).chunk({"time": -1}),
        },
        "*": {
            "obsh": xr.open_zarr(
                os.path.join(FIXTURE_DIR, "precipitation_obsh.zarr")
            ).chunk({"time": -1}),
            "obsp": xr.open_zarr(
                os.path.join(FIXTURE_DIR, "precipitation_obsp.zarr")
            ).chunk({"time": -1}),
            "simh": xr.open_zarr(
                os.path.join(FIXTURE_DIR, "precipitation_simh.zarr")
            ).chunk({"time": -1}),
            "simp": xr.open_zarr(
                os.path.join(FIXTURE_DIR, "precipitation_simp.zarr")
            ).chunk({"time": -1}),
        },
    }
