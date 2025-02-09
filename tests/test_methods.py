# -*- mode: python; coding: utf-8 -*-
#
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# All rights reserved.
# https://github.com/btschwertfeger
#

"""
Module implementing the unit tests for all implemented bias correction
techniques.
"""

from __future__ import annotations

import pytest

from cmethods import adjust
from cmethods.distribution import detrended_quantile_mapping
from cmethods.types import NPData_t, XRData_t

from .helper import is_1d_rmse_better, is_3d_rmse_better

GROUP: str = "time.month"
N_QUANTILES: int = 100


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("linear_scaling", "+"),
        ("linear_scaling", "*"),
        ("variance_scaling", "+"),
        ("delta_method", "+"),
        ("delta_method", "*"),
    ],
)
def test_1d_scaling(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = adjust(method=method, obs=obsh, simh=simh, simp=simp, kind=kind)
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)

    # grouped
    result = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        group=GROUP,
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("linear_scaling", "+"),
        ("linear_scaling", "*"),
        ("variance_scaling", "+"),
        ("delta_method", "+"),
        ("delta_method", "*"),
    ],
)
def test_3d_scaling(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    # not grouped
    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)

    # grouped
    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        group=GROUP,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("linear_scaling", "+"),
        ("linear_scaling", "*"),
        ("variance_scaling", "+"),
    ],
)
def test_3d_scaling_different_time_span(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]
    simh = simh.sel(time=slice(simh.time[1], None)).rename({"time": "t_simh"})

    time_names = {"obs": "time", "simh": "t_simh", "simp": "time"}

    # not grouped
    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        input_core_dims=time_names,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)

    # grouped
    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        group={"obs": "time.month", "simh": "t_simh.month", "simp": "time.month"},
        input_core_dims=time_names,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("quantile_mapping", "+"),
        ("quantile_mapping", "*"),
        ("quantile_delta_mapping", "+"),
        ("quantile_delta_mapping", "*"),
    ],
)
def test_1d_distribution(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    result: XRData_t = adjust(
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
    ("method", "kind"),
    [
        ("quantile_mapping", "+"),
        ("quantile_mapping", "*"),
        ("quantile_delta_mapping", "+"),
        ("quantile_delta_mapping", "*"),
    ],
)
def test_3d_distribution(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("quantile_mapping", "+"),
        ("quantile_mapping", "*"),
        ("quantile_delta_mapping", "+"),
        ("quantile_delta_mapping", "*"),
    ],
)
def test_3d_distribution_different_time_span(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    simh = simh.sel(time=slice(simh.time[1], None)).rename({"time": "t_simh"})
    time_names = {"obs": "time", "simh": "t_simh", "simp": "time"}

    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
        input_core_dims=time_names,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_1d_detrended_quantile_mapping_add(datasets: dict) -> None:
    kind: str = "+"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = detrended_quantile_mapping(
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )
    assert isinstance(result, NPData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_1d_detrended_quantile_mapping_mult(datasets: dict) -> None:
    kind: str = "*"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = detrended_quantile_mapping(
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )
    assert isinstance(result, NPData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)
