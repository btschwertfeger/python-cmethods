# -*- mode: python; coding: utf-8 -*-
#
# Copyright (C) 2024 Benjamin Thomas Schwertfeger
# All rights reserved.
# https://github.com/btschwertfeger
#

"""
Module implementing the unit tests that check if the input data sets can have
different shapes.

TODO: Remove the copy-paste stuff here. That could be done way simpler.
"""

from __future__ import annotations

import pytest

from cmethods import adjust
from cmethods.types import XRData_t

from .helper import is_1d_rmse_better

pytestmark = [pytest.mark.flaky]

N_QUANTILES: int = 100


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("linear_scaling", "+"),
        ("linear_scaling", "*"),
        ("variance_scaling", "+"),
    ],
)
def test_1d_scaling_obs_shorter(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:7300, 0, 0].rename({"time": "t_time"})  # 20/30 years
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        input_core_dims={"obs": "t_time", "simh": "time", "simp": "time"},
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)

    # grouped
    result = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        group={"obs": "t_time.month", "simh": "time.month", "simp": "time.month"},
        input_core_dims={"obs": "t_time", "simh": "time", "simp": "time"},
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("linear_scaling", "+"),
        ("linear_scaling", "*"),
        ("delta_method", "+"),
        ("delta_method", "*"),
        ("variance_scaling", "+"),
    ],
)
def test_1d_scaling_simh_shorter(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:7300, 0, 0].rename({"time": "t_time"})  # 20/30 years
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        input_core_dims={"obs": "time", "simh": "t_time", "simp": "time"},
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)

    # grouped
    result = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        group={"obs": "time.month", "simh": "t_time.month", "simp": "time.month"},
        input_core_dims={"obs": "time", "simh": "t_time", "simp": "time"},
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("linear_scaling", "+"),
        ("linear_scaling", "*"),
        ("variance_scaling", "+"),
    ],
)
def test_1d_scaling_simp_shorter(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:7300, 0, 0].rename({"time": "t_time"})  # 20/30 years
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:7300, 0, 0].rename({"time": "t_time"})  # 20/30 years

    # not group
    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        input_core_dims={"obs": "time", "simh": "time", "simp": "t_time"},
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)

    # grouped
    result = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        group={"obs": "time.month", "simh": "time.month", "simp": "t_time.month"},
        input_core_dims={"obs": "time", "simh": "time", "simp": "t_time"},
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("quantile_mapping", "+"),
        ("quantile_mapping", "*"),
        ("quantile_delta_mapping", "+"),
        ("quantile_delta_mapping", "*"),
    ],
)
def test_1d_distribution_obs_shorter(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:7300, 0, 0].rename({"time": "t_time"})  # 20/30 years
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
        input_core_dims={"obs": "t_time", "simh": "time", "simp": "time"},
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
def test_1d_distribution_simh_shorter(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:7300, 0, 0].rename({"time": "t_time"})  # 20/30 years
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
        input_core_dims={"obs": "time", "simh": "t_time", "simp": "time"},
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
def test_1d_distribution_simp_shorter(
    datasets: dict,
    method: str,
    kind: str,
) -> None:
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:7300, 0, 0].rename({"time": "t_time"})  # 20/30 years
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:7300, 0, 0].rename({"time": "t_time"})  # 20/30 years

    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
        input_core_dims={"obs": "time", "simh": "time", "simp": "t_time"},
    )

    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)
