#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""
Module providing functions for distribution-based bias adjustments. Functions are not
intended to used directly - but as part of the adjustment procedure triggered by
:func:``cmethods.adjust``.
"""

from typing import Any, Final

import numpy as np
import xarray as xr

from cmethods.static import ADDITIVE, MAX_SCALING_FACTOR, MULTIPLICATIVE
from cmethods.types import NPData
from cmethods.utils import (
    check_adjust_called,
    check_np_types,
    ensure_dividable,
    get_cdf,
    get_inverse_of_cdf,
    nan_or_equal,
)


# ? -----========= Q U A N T I L E - M A P P I N G =========------
def quantile_mapping(
    obs: NPData,
    simh: NPData,
    simp: NPData,
    n_quantiles: int,
    kind: str = "+",
    **kwargs: Any,
) -> np.ndarray:
    r"""
    **Do not call this function directly, please use :func:`cmethods.adjust`**
    See https://python-cmethods.readthedocs.io/en/latest/src/methods.html#quantile-mapping
    """
    check_adjust_called(
        function_name="quantile_mapping",
        adjust_called=kwargs.get("adjust_called", None),
    )
    check_np_types(obs=obs, simh=simh, simp=simp)

    if not isinstance(n_quantiles, int):
        raise TypeError("'n_quantiles' must be type int")

    obs, simh, simp = np.array(obs), np.array(simh), np.array(simp)

    global_max = max(np.nanmax(obs), np.nanmax(simh))
    global_min = min(np.nanmin(obs), np.nanmin(simh))

    if nan_or_equal(value1=global_max, value2=global_min):
        return simp

    wide = abs(global_max - global_min) / n_quantiles
    xbins = np.arange(global_min, global_max + wide, wide)

    cdf_obs = get_cdf(obs, xbins)
    cdf_simh = get_cdf(simh, xbins)

    if kind in ADDITIVE:
        epsilon = np.interp(simp, xbins, cdf_simh)  # Eq. 1
        return get_inverse_of_cdf(cdf_obs, epsilon, xbins)  # Eq. 1

    if kind in MULTIPLICATIVE:
        epsilon = np.interp(  # Eq. 2
            simp,
            xbins,
            cdf_simh,
            left=kwargs.get("val_min", 0.0),
            right=kwargs.get("val_max", None),
        )
        return get_inverse_of_cdf(cdf_obs, epsilon, xbins)  # Eq. 2

    raise NotImplementedError(
        f"{kind=} for quantile_mapping is not available. Use '+' or '*' instead.",
    )


# ? -----========= D E T R E N D E D - Q U A N T I L E - M A P P I N G =========------


def detrended_quantile_mapping(
    obs: xr.core.dataarray.DataArray,
    simh: xr.core.dataarray.DataArray,
    simp: xr.core.dataarray.DataArray,
    n_quantiles: int,
    kind: str = "+",
    **kwargs: Any,
) -> NPData:
    r"""
    See https://python-cmethods.readthedocs.io/en/latest/src/methods.html#detrended_quantile_mapping

    This function can only be applied to 1-dimensional data.
    """

    # TODO: this function should also benefit from ufunc -- but how? # pylint: disable=fixme

    if kind not in MULTIPLICATIVE and kind not in ADDITIVE:
        raise NotImplementedError(
            f"{kind=} for detrended_quantile_mapping is not available. Use '+' or '*' instead.",
        )

    if not isinstance(n_quantiles, int):
        raise TypeError("'n_quantiles' must be type int")
    if not isinstance(simp, xr.core.dataarray.DataArray):
        raise TypeError("'simp' must be type xarray.core.dataarray.DataArray")

    obs, simh = np.array(obs), np.array(simh)

    global_max = max(np.nanmax(obs), np.nanmax(simh))
    global_min = min(np.nanmin(obs), np.nanmin(simh))

    if nan_or_equal(value1=global_max, value2=global_min):
        return np.array(simp.values)

    wide = abs(global_max - global_min) / n_quantiles
    xbins = np.arange(global_min, global_max + wide, wide)

    cdf_obs = get_cdf(obs, xbins)
    cdf_simh = get_cdf(simh, xbins)

    # detrended => shift mean of $X_{sim,p}$ to range of $X_{sim,h}$ to adjust extremes
    res = np.zeros(len(simp.values))
    max_scaling_factor: Final[float] = kwargs.get(
        "max_scaling_factor",
        MAX_SCALING_FACTOR,
    )
    for indices in simp.groupby("time.month").groups.values():
        # detrended by long-term month
        m_simh, m_simp = [], []
        for index in indices:
            m_simh.append(simh[index])
            m_simp.append(simp[index])

        m_simh = np.array(m_simh)
        m_simp = np.array(m_simp)
        m_simh_mean = np.nanmean(m_simh)
        m_simp_mean = np.nanmean(m_simp)

        if kind in ADDITIVE:
            epsilon = np.interp(
                m_simp - m_simp_mean + m_simh_mean,
                xbins,
                cdf_simh,
            )  # Eq. 1
            X = (
                get_inverse_of_cdf(cdf_obs, epsilon, xbins) + m_simp_mean - m_simh_mean
            )  # Eq. 1

        else:  # kind in cls.MULTIPLICATIVE:
            epsilon = np.interp(  # Eq. 2
                ensure_dividable(
                    (m_simh_mean * m_simp),
                    m_simp_mean,
                    max_scaling_factor=max_scaling_factor,
                ),
                xbins,
                cdf_simh,
                left=kwargs.get("val_min", 0.0),
                right=kwargs.get("val_max", None),
            )
            X = np.interp(epsilon, cdf_obs, xbins) * ensure_dividable(
                m_simp_mean,
                m_simh_mean,
                max_scaling_factor=max_scaling_factor,
            )  # Eq. 2
        for i, index in enumerate(indices):
            res[index] = X[i]
    return res


# ? -----========= E M P I R I C A L - Q U A N T I L E - M A P P I N G =========------
# def __empirical_quantile_mapping(
#     self: CMethods,
#     obs: NPData,
#     simh: NPData,
#     simp: NPData,
#     n_quantiles: int = 100,
#     extrapolate: Optional[str] = None,
#     **kwargs: Any,
# ) -> NPData:
#     """
#     Method to adjust 1-dimensional climate data by empirical quantile mapping
#     """
#     raise NotImplementedError(
#         "Not implemented; please have a look at:
# https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py "
#     )

# ? -----========= Q U A N T I L E - D E L T A - M A P P I N G =========------


def quantile_delta_mapping(
    obs: NPData,
    simh: NPData,
    simp: NPData,
    n_quantiles: int,
    kind: str = "+",
    **kwargs: Any,
) -> NPData:
    r"""
    **Do not call this function directly, please use :func:`cmethods.adjust`**

    See https://python-cmethods.readthedocs.io/en/latest/src/methods.html#quantile-delta-mapping
    """
    check_adjust_called(
        function_name="quantile_delta_mapping",
        adjust_called=kwargs.get("adjust_called", None),
    )
    check_np_types(obs=obs, simh=simh, simp=simp)

    if not isinstance(n_quantiles, int):
        raise TypeError("'n_quantiles' must be type int")

    if kind in ADDITIVE:
        obs, simh, simp = (
            np.array(obs),
            np.array(simh),
            np.array(simp),
        )  # to achieve higher accuracy
        global_max = kwargs.get("global_max", max(np.nanmax(obs), np.nanmax(simh)))
        global_min = kwargs.get("global_min", min(np.nanmin(obs), np.nanmin(simh)))

        if nan_or_equal(value1=global_max, value2=global_min):
            return simp

        wide = abs(global_max - global_min) / n_quantiles
        xbins = np.arange(global_min, global_max + wide, wide)

        cdf_obs = get_cdf(obs, xbins)
        cdf_simh = get_cdf(simh, xbins)
        cdf_simp = get_cdf(simp, xbins)

        # calculate exact CDF values of $F_{sim,p}[T_{sim,p}(t)]$
        epsilon = np.interp(simp, xbins, cdf_simp)  # Eq. 1.1
        QDM1 = get_inverse_of_cdf(cdf_obs, epsilon, xbins)  # Eq. 1.2
        delta = simp - get_inverse_of_cdf(cdf_simh, epsilon, xbins)  # Eq. 1.3
        return QDM1 + delta  # Eq. 1.4

    if kind in MULTIPLICATIVE:
        obs, simh, simp = np.array(obs), np.array(simh), np.array(simp)
        global_max = kwargs.get("global_max", max(np.nanmax(obs), np.nanmax(simh)))
        global_min = kwargs.get("global_min", 0.0)
        if nan_or_equal(value1=global_max, value2=global_min):
            return simp

        wide = global_max / n_quantiles
        xbins = np.arange(global_min, global_max + wide, wide)

        cdf_obs = get_cdf(obs, xbins)
        cdf_simh = get_cdf(simh, xbins)
        cdf_simp = get_cdf(simp, xbins)

        epsilon = np.interp(simp, xbins, cdf_simp)  # Eq. 1.1
        QDM1 = get_inverse_of_cdf(cdf_obs, epsilon, xbins)  # Eq. 1.2

        delta = ensure_dividable(  # Eq. 2.3
            simp,
            get_inverse_of_cdf(cdf_simh, epsilon, xbins),
            max_scaling_factor=kwargs.get(
                "max_scaling_scaling",
                MAX_SCALING_FACTOR,
            ),
        )
        return QDM1 * delta  # Eq. 2.4
    raise NotImplementedError(
        f"{kind=} not available for quantile_delta_mapping. Use '+' or '*' instead.",
    )


__all__ = ["detrended_quantile_mapping"]
