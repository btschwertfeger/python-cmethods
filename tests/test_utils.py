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

from cmethods import CMethods as cm

# --------------------------------------------------------------------------
# test nan values


def test_quantile_mapping_single_nan() -> None:
    obs, simh, simp = list(np.arange(10)), list(np.arange(10)), list(np.arange(10))
    obs[0] = np.nan
    expected = np.array([0.0, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.0])

    res = cm.quantile_mapping(obs=obs, simh=simh, simp=simp, n_quantiles=5)
    assert np.allclose(res, expected)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
def test_quantile_mapping_all_nan() -> None:
    obs, simh, simp = (
        list(np.full(10, np.nan)),
        list(np.arange(10)),
        list(np.arange(10)),
    )
    res = cm.quantile_mapping(obs=obs, simh=simh, simp=simp, n_quantiles=5)
    assert np.allclose(res, simp)


def test_quantile_delta_mapping_single_nan() -> None:
    obs, simh, simp = list(np.arange(10)), list(np.arange(10)), list(np.arange(10))
    obs[0] = np.nan
    expected = np.array([0.0, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.0])

    res = cm.quantile_delta_mapping(obs=obs, simh=simh, simp=simp, n_quantiles=5)
    assert np.allclose(res, expected)


@pytest.mark.filterwarnings("ignore:All-NaN slice encountered")
def test_quantile_delta_mapping_all_nan() -> None:
    obs, simh, simp = (
        list(np.full(10, np.nan)),
        list(np.arange(10)),
        list(np.arange(10)),
    )
    res = cm.quantile_delta_mapping(obs=obs, simh=simh, simp=simp, n_quantiles=5)
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
