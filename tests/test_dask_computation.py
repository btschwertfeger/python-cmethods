#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

import xarray as xr

from cmethods.types import NPData_t, XRData_t
import pytest

from cmethods import CMethods

from .helper import is_1d_rmse_better, is_3d_rmse_better

from dask.distributed import LocalCluster

GROUP: str = "time.month"


@pytest.mark.wip()
def test_linear_scaling_add_zarr(
    cm: CMethods, temperature_data_zarr: xr.Dataset, variable: str
) -> None:
    method: str = "linear_scaling"
    kind: str = "+"
    obsh: xr.DataArray = temperature_data_zarr["obsh"]
    obsp: xr.DataArray = temperature_data_zarr["obsp"]
    simh: xr.DataArray = temperature_data_zarr["simh"]
    simp: xr.DataArray = temperature_data_zarr["simp"]

    cluster = LocalCluster()
    client = cluster.get_client()

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
