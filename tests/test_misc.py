#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""Module implementing even more tests"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pytest

from cmethods import adjust
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
