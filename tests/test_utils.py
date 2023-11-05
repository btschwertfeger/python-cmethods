#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitGub: https://github.com/btschwertfeger
#

"""
Module to to test utility functions for the CMethods package

Data types are ignored for simplicity.
"""

import numpy as np
import pytest
import xarray as xr

from cmethods import CMethods as cm

# --------------------------------------------------------------------------
# test for nan values


def test_quantile_mapping_single_nan() -> None:
    obs, simh, simp = list(np.arange(10)), list(np.arange(10)), list(np.arange(10))
    obs[0] = np.nan
    expected = np.array([0.0, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.0])

    res = cm.__quantile_mapping(obs=obs, simh=simh, simp=simp, n_quantiles=5)
    assert np.allclose(res, expected)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
def test_quantile_mapping_all_nan() -> None:
    obs, simh, simp = (
        list(np.full(10, np.nan)),
        list(np.arange(10)),
        list(np.arange(10)),
    )
    res = cm.__quantile_mapping(obs=obs, simh=simh, simp=simp, n_quantiles=5)
    assert np.allclose(res, simp)


def test_quantile_delta_mapping_single_nan() -> None:
    obs, simh, simp = list(np.arange(10)), list(np.arange(10)), list(np.arange(10))
    obs[0] = np.nan
    expected = np.array([0.0, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.0])

    res = cm.__quantile_delta_mapping(obs=obs, simh=simh, simp=simp, n_quantiles=5)
    assert np.allclose(res, expected)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
def test_quantile_delta_mapping_all_nan() -> None:
    obs, simh, simp = (
        list(np.full(10, np.nan)),
        list(np.arange(10)),
        list(np.arange(10)),
    )
    res = cm.__quantile_delta_mapping(obs=obs, simh=simh, simp=simp, n_quantiles=5)
    assert np.allclose(res, simp)


# --------------------------------------------------------------------------
# test utils


def test_get_available_methods() -> None:
    assert cm.get_available_methods() == [
        "linear_scaling",
        "variance_scaling",
        "delta_method",
        "quantile_mapping",
        "detrended_quantile_mapping",
        "quantile_delta_mapping",
    ]


def test_nan_or_equal() -> None:
    assert cm.nan_or_equal(0, 0)
    assert cm.nan_or_equal(np.NaN, np.NaN)
    assert not cm.nan_or_equal(0, 1)


def test_get_pdf() -> None:
    assert (cm.get_pdf(np.arange(10), [0, 5, 11]) == np.array((5, 5))).all()


def test_get_adjusted_scaling_factor() -> None:
    assert cm.get_adjusted_scaling_factor(10, 5) == 5
    assert cm.get_adjusted_scaling_factor(10, 11) == 10
    assert cm.get_adjusted_scaling_factor(-10, -11) == -10
    assert cm.get_adjusted_scaling_factor(-11, -10) == -10


def test_ensure_devidable() -> None:
    assert np.array_equal(
        cm.ensure_devidable(np.array((1, 2, 3, 4, 5, 0)), np.array((0, 1, 0, 2, 3, 0))),
        np.array((10, 2, 30, 2, 5 / 3, 0)),
    )


# --------------------------------------------------------------------------
# test type checking related functions
# For most of them only one part of the check is tested, since other tests
# are already covering the functionality and correctness of functions using
# valid values.


def test_type_check() -> None:
    """
    Checks the correctness of the type checking function when the types are
    correct. No error should occur.
    """

    cm.check_types(obs=[], simh=[], simp=[])


def test_type_check_failing() -> None:
    """
    Checks the correctness of the type checking function when the inputs do not
    have the correct type.
    """

    phrase: str = "must be type list, np.ndarray or xarray.core.dataarray.DataArray"
    with pytest.raises(TypeError, match=f"'obs' {phrase}"):
        cm.check_types(obs=1, simh=[], simp=[])

    with pytest.raises(TypeError, match=f"'simh' {phrase}"):
        cm.check_types(obs=[], simh=1, simp=[])

    with pytest.raises(TypeError, match=f"'simp' {phrase}"):
        cm.check_types(obs=[], simh=[], simp=1)


def test_quantile_mapping_type_check_n_quantiles_failing() -> None:
    """n_quantiles must by type int"""
    with pytest.raises(TypeError, match="'n_quantiles' must be type int"):
        cm.__quantile_delta_mapping(obs=[], simh=[], simp=[], n_quantiles="100")


def test_detrended_quantile_mapping_type_check_n_quantiles_failing() -> None:
    """n_quantiles must by type int"""
    with pytest.raises(TypeError, match="'n_quantiles' must be type int"):
        cm.detrended_quantile_mapping(obs=[], simh=[], simp=[], n_quantiles="100")


def test_detrended_quantile_mapping_type_check_simp_failing() -> None:
    """simp must be type xarray.core.dataarray.DataArray"""
    with pytest.raises(
        TypeError, match="'simp' must be type xarray.core.dataarray.DataArray"
    ):
        cm.detrended_quantile_mapping(obs=[], simh=[], simp=[], n_quantiles=100)


def test_quantile_delta_mapping_type_check_n_quantiles_failing() -> None:
    """n_quantiles must by type int"""
    with pytest.raises(TypeError, match="'n_quantiles' must be type int"):
        cm.__quantile_delta_mapping(obs=[], simh=[], simp=[], n_quantiles="100")


def test_grouped_correction_worng_type_failing() -> None:
    """simp must be type xarray.core.dataarray.DataArray"""
    with pytest.raises(
        TypeError, match="'simp' must be type xarray.core.dataarray.DataArray"
    ):
        cm.__linear_scaling(
            obs=[], simh=[], simp=[], n_quantiles=100, group="time.month"
        )


@pytest.mark.wip()
def test_adjust_3d_type_checking_failing() -> None:
    """
    Checks for all types that are expected to be passed to the adjust_3d method
    """
    # Create the DataArray
    data: xr.core.dataarray.DataArray = xr.DataArray(
        [10, 20, 30, 40, 50], dims=["time"]
    )
    with pytest.raises(
        TypeError, match="'obs' must be type xarray.core.dataarray.DataArray"
    ):
        cm.adjust_3d(
            method="linear_scaling",
            obs=[],
            simh=data,
            simp=data,
            n_quantiles=100,
            group="time.month",
        )
    with pytest.raises(
        TypeError, match="'simh' must be type xarray.core.dataarray.DataArray"
    ):
        cm.adjust_3d(
            method="linear_scaling",
            obs=data,
            simh=[],
            simp=data,
            n_quantiles=100,
            group="time.month",
        )

    with pytest.raises(
        TypeError, match="'simp' must be type xarray.core.dataarray.DataArray"
    ):
        cm.adjust_3d(
            method="linear_scaling",
            obs=data,
            simh=data,
            simp=[],
            n_quantiles=100,
            group="time.month",
        )

    with pytest.raises(TypeError, match="'n_quantiles' must be type int"):
        cm.adjust_3d(
            method="linear_scaling",
            obs=data,
            simh=data,
            simp=data,
            n_quantiles="100",
            group="time.month",
        )

    with pytest.raises(TypeError, match="'n_jobs' must be type int"):
        cm.adjust_3d(
            method="linear_scaling",
            obs=data,
            simh=data,
            simp=data,
            n_quantiles=100,
            group="time.month",
            n_jobs="1",
        )
