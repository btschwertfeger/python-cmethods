#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""
Module providing the main function that is used to apply the implemented bias
correction techniques.
"""

from __future__ import annotations

from typing import Callable, Dict

import xarray as xr

from cmethods.distribution import quantile_delta_mapping as __quantile_delta_mapping
from cmethods.distribution import quantile_mapping as __quantile_mapping
from cmethods.scaling import delta_method as __delta_method
from cmethods.scaling import linear_scaling as __linear_scaling
from cmethods.scaling import variance_scaling as __variance_scaling
from cmethods.types import XRData
from cmethods.utils import UnknownMethodError, check_xr_types

__METHODS_FUNC__: Dict[str, Callable] = {
    "linear_scaling": __linear_scaling,
    "variance_scaling": __variance_scaling,
    "delta_method": __delta_method,
    "quantile_mapping": __quantile_mapping,
    "quantile_delta_mapping": __quantile_delta_mapping,
}


def apply_ufunc(
    method: str,
    obs: XRData,
    simh: XRData,
    simp: XRData,
    **kwargs: dict,
) -> XRData:
    """
    Internal function used to apply the bias correction technique to the
    passed input data.
    """
    check_xr_types(obs=obs, simh=simh, simp=simp)
    if method not in __METHODS_FUNC__:
        raise UnknownMethodError(method, __METHODS_FUNC__.keys())

    result: XRData = xr.apply_ufunc(
        __METHODS_FUNC__[method],
        obs,
        simh,
        # Need to spoof a fake time axis since 'time' coord on full dataset is different
        # than 'time' coord on training dataset.
        simp.rename({"time": "t2"}),
        dask="parallelized",
        vectorize=True,
        # This will vectorize over the time dimension, so will submit each grid cell
        # independently
        input_core_dims=[["time"], ["time"], ["t2"]],
        # Need to denote that the final output dataset will be labeled with the
        # spoofed time coordinate
        output_core_dims=[["t2"]],
        kwargs=dict(kwargs),
    )

    # Rename to proper coordinate name.
    result = result.rename({"t2": "time"})

    # ufunc will put the core dimension to the end (time), so want to preserve original
    # order where time is commonly first.
    return result.transpose(*obs.dims)


__all__ = []
