#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

from typing import Any

import pytest
import xarray as xr

from cmethods import adjust
from cmethods.types import XRData_t

from .helper import is_3d_rmse_better

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
def test_3d_scaling_zarr(
    datasets_from_zarr: xr.Dataset,
    method: str,
    kind: str,
    dask_cluster: Any,
) -> None:
    variable: str = "tas" if kind == "+" else "pr"
    obsh: xr.DataArray = datasets_from_zarr[kind]["obsh"][variable]
    obsp: xr.DataArray = datasets_from_zarr[kind]["obsp"][variable]
    simh: xr.DataArray = datasets_from_zarr[kind]["simh"][variable]
    simp: xr.DataArray = datasets_from_zarr[kind]["simp"][variable]

    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
    )
    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[variable], obsp=obsp, simp=simp)

    # grouped
    result = adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )
    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[variable], obsp=obsp, simp=simp)


@pytest.mark.parametrize(
    "method,kind",
    [
        ("quantile_mapping", "+"),
        ("quantile_delta_mapping", "+"),
        ("quantile_mapping", "*"),
        ("quantile_delta_mapping", "*"),
    ],
)
def test_3d_distribution_zarr(
    datasets_from_zarr: dict,
    method: str,
    kind: str,
    dask_cluster: Any,
) -> None:
    variable: str = "tas" if kind == "+" else "pr"
    obsh: XRData_t = datasets_from_zarr[kind]["obsh"][variable]
    obsp: XRData_t = datasets_from_zarr[kind]["obsp"][variable]
    simh: XRData_t = datasets_from_zarr[kind]["simh"][variable]
    simp: XRData_t = datasets_from_zarr[kind]["simp"][variable]

    result: XRData_t = adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[variable], obsp=obsp, simp=simp)
