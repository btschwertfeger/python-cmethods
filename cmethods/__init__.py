#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

r"""
    Module providing the a method named "adjust" to apply different bias
    correction techniques to time-series climate data.

    Some variables used in this package:

    T = Temperatures ($T$)
    X = Some climate variable ($X$)
    h = historical
    p = scenario; future; predicted
    obs = observed data ($T_{obs,h}$)
    simh = modeled data with same time period as obs ($T_{sim,h}$)
    simp = data to correct (predicted simulated data) ($T_{sim,p}$)
    F = Cumulative Distribution Function
    \mu = mean
    \sigma = standard deviation
    i = index
    _{m} = long-term monthly interval
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import xarray as xr

from cmethods.core import apply_ufunc
from cmethods.static import SCALING_METHODS
from cmethods.types import XRData
from cmethods.utils import check_xr_types

__author__ = "Benjamin Thomas Schwertfeger"
__copyright__ = __author__
__email__ = "contact@b-schwertfeger.de"
__link__ = "https://github.com/btschwertfeger"
__github__ = "https://github.com/btschwertfeger/python-cmethods"


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

    if method == "detrended_quantile_mapping":
        raise ValueError(
            "This function is not available for detrended quantile mapping."
            " Please use cmethods.CMethods.detrended_quantile_mapping"
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
            "Can't use group for distribution based methods."  # except for DQM
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

        if result is None:
            result = monthly_result
        else:
            result = xr.merge([result, monthly_result])

    return result


__all__ = ["adjust"]
