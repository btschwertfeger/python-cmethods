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

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import xarray as xr

from cmethods.distribution import quantile_delta_mapping as __quantile_delta_mapping
from cmethods.distribution import quantile_mapping as __quantile_mapping
from cmethods.scaling import delta_method as __delta_method
from cmethods.scaling import linear_scaling as __linear_scaling
from cmethods.scaling import variance_scaling as __variance_scaling
from cmethods.static import SCALING_METHODS
from cmethods.utils import UnknownMethodError, check_xr_types

if TYPE_CHECKING:
    from cmethods.types import XRData

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


def adjust(
    method: str,
    obs: XRData,
    simh: XRData,
    simp: XRData,
    **kwargs,
) -> XRData:
    """
    Function to apply a bias correction technique on single and multidimensional
    data sets. For more information please refer to the method specific
    requirements and execution examples.

    See https://python-cmethods.readthedocs.io/en/latest/src/methods.html

    :param method: Technique to apply
    :type method: str
    :param obs: The reference/observational data set
    :type obs: XRData
    :param simh: The modeled data of the control period
    :type simh: XRData
    :param simp: The modeled data of the period to adjust
    :type simp: XRData
    :param kwargs: Any other method-specific parameter (like
        ``n_quantiles`` and ``kind``)
    :type kwargs: dict
    :return: The bias corrected/adjusted data set
    :rtype: XRData
    """
    kwargs["adjust_called"] = True
    check_xr_types(obs=obs, simh=simh, simp=simp)

    if method == "detrended_quantile_mapping":  # noqa: PLR2004
        raise ValueError(
            "This function is not available for detrended quantile mapping."
            " Please use cmethods.CMethods.detrended_quantile_mapping",
        )

    # No grouped correction | distribution-based technique
    # NOTE: This is disabled since using groups like "time.month" will lead
    #       to unrealistic monthly transitions. If such behavior is wanted,
    #       mock this function or apply ``CMethods.__apply_ufunc` directly
    #       on your data sets.
    if kwargs.get("group", None) is None:
        return apply_ufunc(method, obs, simh, simp, **kwargs).to_dataset()

    if method not in SCALING_METHODS:
        raise ValueError(
            "Can't use group for distribution based methods.",  # except for DQM
        )

    # Grouped correction | scaling-based technique
    group: str = kwargs["group"]
    del kwargs["group"]

    obs_g: List[Tuple[int, XRData]] = list(obs.groupby(group))
    simh_g: List[Tuple[int, XRData]] = list(simh.groupby(group))
    simp_g: List[Tuple[int, XRData]] = list(simp.groupby(group))

    result: Optional[XRData] = None
    for index in range(len(list(obs_g))):
        obs_gds: XRData = obs_g[index][1]
        simh_gds: XRData = simh_g[index][1]
        simp_gds: XRData = simp_g[index][1]

        monthly_result = apply_ufunc(
            method,
            obs_gds,
            simh_gds,
            simp_gds,
            **kwargs,
        )

        result = (
            monthly_result if result is None else xr.merge([result, monthly_result])
        )

    return result


__all__ = ["adjust"]
