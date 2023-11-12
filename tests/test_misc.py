#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""Module implementing even more tests"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from cmethods import CMethods

import numpy as np

from cmethods.utils import UnknownMethodError


def test_not_implemented_errors(cm: CMethods, datasets: dict) -> None:
    with pytest.raises(
        NotImplementedError,
        match=re.escape(r"kind='/' not available for linear_scaling."),
    ):
        cm._CMethods__linear_scaling(obs=[], simh=[], simp=[], kind="/")

    with pytest.raises(
        NotImplementedError,
        match=re.escape(r"kind='/' not available for variance_scaling."),
    ):
        cm._CMethods__variance_scaling(obs=[], simh=[], simp=[], kind="/")

    with pytest.raises(
        NotImplementedError,
        match=re.escape(r"kind='/' not available for delta_method. "),
    ):
        cm._CMethods__delta_method(obs=[], simh=[], simp=[], kind="/")

    with pytest.raises(
        NotImplementedError,
        match=re.escape(r"kind='/' for quantile_mapping is not available."),
    ):
        cm._CMethods__quantile_mapping(
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
        cm.detrended_quantile_mapping(
            obs=np.array(datasets["+"]["obsh"][:, 0, 0]),
            simh=np.array(datasets["+"]["simh"][:, 0, 0]),
            simp=np.array(datasets["+"]["simp"][:, 0, 0]),
            kind="/",
            n_quantiles=100,
        )

    with pytest.raises(
        NotImplementedError,
        match=re.escape(r"kind='/' not available for quantile_delta_mapping."),
    ):
        cm._CMethods__quantile_delta_mapping(
            obs=np.array(datasets["+"]["obsh"][:, 0, 0]),
            simh=np.array(datasets["+"]["simh"][:, 0, 0]),
            simp=np.array(datasets["+"]["simp"][:, 0, 0]),
            kind="/",
            n_quantiles=100,
        )


def test_adjust_failing_dqm(cm: CMethods, datasets: dict) -> None:
    with pytest.raises(ValueError):
        cm.adjust(
            method="detrended_quantile_mapping",
            obs=datasets["+"]["obsh"][:, 0, 0],
            simh=datasets["+"]["simh"][:, 0, 0],
            simp=datasets["+"]["simp"][:, 0, 0],
            kind="/",
            n_quantiles=100,
        )


def test_adjust_failing_no_group_for_distribution(cm: CMethods, datasets: dict) -> None:
    with pytest.raises(ValueError):
        cm.adjust(
            method="quantile_mapping",
            obs=datasets["+"]["obsh"][:, 0, 0],
            simh=datasets["+"]["simh"][:, 0, 0],
            simp=datasets["+"]["simp"][:, 0, 0],
            kind="/",
            n_quantiles=100,
            group="time.month",
        )


def test_get_function_unknown_method(cm: CMethods) -> None:
    with pytest.raises(UnknownMethodError):
        cm._CMethods__get_function("not-existing")
