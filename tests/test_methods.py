#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""
Module implementing the unit tests for all implemented bias correction
techniques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cmethods.types import NPData_t, XRData_t

if TYPE_CHECKING:
    from cmethods import CMethods

import pytest

from .helper import is_1d_rmse_better, is_3d_rmse_better

GROUP: str = "time.month"
N_QUANTILES: int = 100


@pytest.mark.parametrize(
    "method,kind",
    [
        ("linear_scaling", "+"),
        ("variance_scaling", "+"),
        ("delta_method", "+"),
        ("linear_scaling", "*"),
        ("delta_method", "*"),
    ],
)
def test_1d_scaling(
    cm: CMethods,
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)

    # grouped
    result = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    "method,kind",
    [
        ("linear_scaling", "+"),
        ("variance_scaling", "+"),
        ("delta_method", "+"),
        ("linear_scaling", "*"),
        ("delta_method", "*"),
    ],
)
def test_3d_scaling(
    cm: CMethods,
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    # not grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)

    # grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    "method,kind",
    [
        ("quantile_mapping", "+"),
        ("quantile_delta_mapping", "+"),
        ("quantile_mapping", "*"),
        ("quantile_delta_mapping", "*"),
    ],
)
def test_1d_distribution(
    cm: CMethods,
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    "method,kind",
    [
        ("quantile_mapping", "+"),
        ("quantile_delta_mapping", "+"),
        ("quantile_mapping", "*"),
        ("quantile_delta_mapping", "*"),
    ],
)
def test_3d_distribution(
    cm: CMethods,
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_detrended_quantile_mapping_add_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.detrended_quantile_mapping(
        obs=obsh, simh=simh, simp=simp, kind=kind, n_quantiles=N_QUANTILES
    )
    assert isinstance(result, NPData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_detrended_quantile_mapping_mult_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.detrended_quantile_mapping(
        obs=obsh, simh=simh, simp=simp, kind=kind, n_quantiles=N_QUANTILES
    )
    assert isinstance(result, NPData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)
