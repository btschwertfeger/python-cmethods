#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""Module providing utility functions for testing."""

from __future__ import annotations

from functools import cache
from typing import List

import numpy as np
import xarray as xr
from sklearn.metrics import mean_squared_error


def is_1d_rmse_better(result, obsp, simp) -> bool:
    return np.sqrt(mean_squared_error(result, obsp)) < np.sqrt(
        mean_squared_error(simp, obsp),
    )


def is_3d_rmse_better(result, obsp, simp) -> bool:
    result_reshaped = result.stack(z=("lat", "lon"))
    obsp_reshaped = obsp.stack(z=("lat", "lon"))
    simp_reshaped = simp.stack(z=("lat", "lon"))

    # Compute RMSE
    rmse_values_old = np.sqrt(
        mean_squared_error(simp_reshaped, obsp_reshaped, multioutput="raw_values"),
    )
    rmse_values_new = np.sqrt(
        mean_squared_error(result_reshaped, obsp_reshaped, multioutput="raw_values"),
    )
    # Convert the flattened array back to the original grid shape
    rmse_values_old_ds = xr.DataArray(
        rmse_values_old.reshape(obsp.lat.size, obsp.lon.size),
        coords={"lat": obsp.lat, "lon": obsp.lon},
        dims=["lat", "lon"],
    )
    rmse_values_new_ds = xr.DataArray(
        rmse_values_new.reshape(obsp.lat.size, obsp.lon.size),
        coords={"lat": obsp.lat, "lon": obsp.lon},
        dims=["lat", "lon"],
    )
    return (rmse_values_new_ds < rmse_values_old_ds).all()


@cache
def get_datasets(kind: str) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset, xr.Dataset]:
    historical_time = xr.cftime_range(
        "1971-01-01",
        "2000-12-31",
        freq="D",
        calendar="noleap",
    )
    future_time = xr.cftime_range(
        "2001-01-01",
        "2030-12-31",
        freq="D",
        calendar="noleap",
    )
    latitudes = np.arange(23, 27, 1)

    def get_hist_temp_for_lat(lat: int) -> List[float]:
        """Returns a fake interval time series by latitude value"""
        return 273.15 - (
            lat * np.cos(2 * np.pi * historical_time.dayofyear / 365)
            + 2 * np.random.random_sample((historical_time.size,))
            + 273.15
            + 0.1 * (historical_time - historical_time[0]).days / 365
        )

    def get_fake_hist_precipitation_data() -> List[float]:
        """Returns ratio based fake time series"""
        pr = (
            np.cos(2 * np.pi * historical_time.dayofyear / 365)
            * np.cos(2 * np.pi * historical_time.dayofyear / 365)
            * np.random.random_sample((historical_time.size,))
        )

        pr *= 0.0004 / pr.max()  # scaling
        years = 30
        days_without_rain_per_year = 239

        c = days_without_rain_per_year * years  # avoid rain every day
        pr.ravel()[np.random.choice(pr.size, c, replace=False)] = 0
        return pr

    def get_dataset(data, time, kind: str) -> xr.Dataset:
        """Returns a data set by data and time"""
        return (
            xr.DataArray(
                data,
                dims=("lon", "lat", "time"),
                coords={"time": time, "lat": latitudes, "lon": [0, 1, 3]},
            )
            .transpose("time", "lat", "lon")
            .to_dataset(name=kind)
        )

    if kind == "+":  # noqa: PLR2004
        some_data = [get_hist_temp_for_lat(val) for val in latitudes]
        data = np.array(
            [
                np.array(some_data),
                np.array(some_data) + 0.5,
                np.array(some_data) + 1,
            ],
        )
        obsh = get_dataset(data, historical_time, kind=kind)
        obsp = get_dataset(data + 1, historical_time, kind=kind)
        simh = get_dataset(data - 2, historical_time, kind=kind)
        simp = get_dataset(data - 1, future_time, kind=kind)

    else:  # precipitation
        some_data = [get_fake_hist_precipitation_data() for _ in latitudes]
        data = np.array(
            [some_data, np.array(some_data) + np.random.rand(), np.array(some_data)],
        )
        obsh = get_dataset(data, historical_time, kind=kind)
        obsp = get_dataset(data * 1.02, historical_time, kind=kind)
        simh = get_dataset(data * 0.98, historical_time, kind=kind)
        simp = get_dataset(data * 0.09, future_time, kind=kind)

    return obsh, obsp, simh, simp
