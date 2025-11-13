# -*- mode: python; coding: utf-8 -*-
#
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# All rights reserved.
# https://github.com/btschwertfeger
#

"""Module implementing even more tests"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pytest
import xarray as xr

from cmethods import adjust
from cmethods.core import _add_cmethods_metadata
from cmethods.distribution import (
    detrended_quantile_mapping,
    quantile_delta_mapping,
    quantile_mapping,
)
from cmethods.scaling import delta_method, linear_scaling, variance_scaling


def test_not_implemented_errors(
    datasets: dict,
    caplog: Any,
) -> None:
    caplog.set_level(logging.INFO)

    with (
        pytest.raises(
            NotImplementedError,
            match=re.escape(r"kind='/' not available for linear_scaling."),
        ),
        pytest.warns(UserWarning, match="Do not call linear_scaling"),
    ):
        linear_scaling(obs=[], simh=[], simp=[], kind="/")

    with (
        pytest.raises(
            NotImplementedError,
            match=re.escape(r"kind='/' not available for variance_scaling."),
        ),
        pytest.warns(UserWarning, match="Do not call variance_scaling"),
    ):
        variance_scaling(obs=[], simh=[], simp=[], kind="/")

    with (
        pytest.raises(
            NotImplementedError,
            match=re.escape(r"kind='/' not available for delta_method. "),
        ),
        pytest.warns(UserWarning, match="Do not call delta_method"),
    ):
        delta_method(obs=[], simh=[], simp=[], kind="/")

    with (
        pytest.raises(
            NotImplementedError,
            match=re.escape(r"kind='/' for quantile_mapping is not available."),
        ),
        pytest.warns(UserWarning, match="Do not call quantile_mapping"),
    ):
        quantile_mapping(
            obs=np.array(datasets["+"]["obsh"][:, 0, 0]),
            simh=np.array(datasets["+"]["simh"][:, 0, 0]),
            simp=np.array(datasets["+"]["simp"][:, 0, 0]),
            kind="/",
            n_quantiles=100,
        )
    with pytest.raises(
        NotImplementedError,
        match=re.escape(r"kind='/' for detrended_quantile_mapping is not available."),
    ):
        detrended_quantile_mapping(
            obs=np.array(datasets["+"]["obsh"][:, 0, 0]),
            simh=np.array(datasets["+"]["simh"][:, 0, 0]),
            simp=np.array(datasets["+"]["simp"][:, 0, 0]),
            kind="/",
            n_quantiles=100,
        )

    with (
        pytest.raises(
            NotImplementedError,
            match=re.escape(r"kind='/' not available for quantile_delta_mapping."),
        ),
        pytest.warns(UserWarning, match="Do not call quantile_delta_mapping"),
    ):
        quantile_delta_mapping(
            obs=np.array(datasets["+"]["obsh"][:, 0, 0]),
            simh=np.array(datasets["+"]["simh"][:, 0, 0]),
            simp=np.array(datasets["+"]["simp"][:, 0, 0]),
            kind="/",
            n_quantiles=100,
        )


def test_adjust_failing_dqm(datasets: dict) -> None:
    with pytest.raises(
        ValueError,
        match=r"This function is not available for detrended quantile mapping. "
        "Please use cmethods.CMethods.detrended_quantile_mapping",
    ):
        adjust(
            method="detrended_quantile_mapping",
            obs=datasets["+"]["obsh"][:, 0, 0],
            simh=datasets["+"]["simh"][:, 0, 0],
            simp=datasets["+"]["simp"][:, 0, 0],
            kind="/",
            n_quantiles=100,
        )


def test_adjust_failing_no_group_for_distribution(datasets: dict) -> None:
    with pytest.raises(
        ValueError,
        match=r"Can't use group for distribution based methods.",
    ):
        adjust(
            method="quantile_mapping",
            obs=datasets["+"]["obsh"][:, 0, 0],
            simh=datasets["+"]["simh"][:, 0, 0],
            simp=datasets["+"]["simp"][:, 0, 0],
            kind="/",
            n_quantiles=100,
            group="time.month",
        )


def test_add_cmethods_metadata_with_dataarray() -> None:
    """Test that _add_cmethods_metadata adds correct attributes to a DataArray"""
    data = xr.DataArray(
        np.array([1, 2, 3, 4, 5]),
        dims=["time"],
        coords={"time": np.arange(5)},
    )

    result = _add_cmethods_metadata(
        data,
        method="linear_scaling",
        kind="+",
        n_quantiles=100,
        group="time.month",
    )

    assert "cmethods_version" in result.attrs
    assert "cmethods_method" in result.attrs
    assert "cmethods_timestamp" in result.attrs
    assert "cmethods_source" in result.attrs

    assert result.attrs["cmethods_method"] == "linear_scaling"
    assert result.attrs["cmethods_kind"] == "+"
    assert result.attrs["cmethods_n_quantiles"] == "100"
    assert result.attrs["cmethods_group"] == "time.month"
    assert result.attrs["cmethods_source"] == "https://github.com/btschwertfeger/python-cmethods"
    assert "UTC" in result.attrs["cmethods_timestamp"]


def test_add_cmethods_metadata_with_dataset() -> None:
    """Test that _add_cmethods_metadata adds correct attributes to a Dataset"""
    data = xr.Dataset(
        {
            "temperature": xr.DataArray(
                np.array([1, 2, 3, 4, 5]),
                dims=["time"],
                coords={"time": np.arange(5)},
            ),
        },
    )

    result = _add_cmethods_metadata(data, method="quantile_mapping")

    assert "cmethods_version" in result.attrs
    assert "cmethods_method" in result.attrs
    assert "cmethods_timestamp" in result.attrs
    assert "cmethods_source" in result.attrs
    assert result.attrs["cmethods_method"] == "quantile_mapping"


def test_add_cmethods_metadata_optional_params() -> None:
    """Test that _add_cmethods_metadata handles optional parameters correctly"""
    data = xr.DataArray(
        np.array([1, 2, 3]),
        dims=["time"],
        coords={"time": np.arange(3)},
    )

    result = _add_cmethods_metadata(data, method="variance_scaling")

    assert "cmethods_method" in result.attrs
    assert result.attrs["cmethods_method"] == "variance_scaling"
    assert "cmethods_kind" not in result.attrs
    assert "cmethods_n_quantiles" not in result.attrs
    assert "cmethods_group" not in result.attrs
