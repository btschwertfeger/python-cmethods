#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

r"""
    Module to apply different bias correction techniques to time-series climate data

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

from typing import Any, Callable, Optional

import numpy as np
import xarray as xr

__author__ = "Benjamin Thomas Schwertfeger"
__copyright__ = __author__
__email__ = "contact@b-schwertfeger.de"
__link__ = "https://github.com/btschwertfeger"
__github__ = "https://github.com/btschwertfeger/python-cmethods"


from cmethods.static import (
    ADDITIVE,
    DISTRIBUTION_METHODS,
    METHODS,
    MULTIPLICATIVE,
    SCALING_METHODS,
)
from cmethods.types import NPData, XRData
from cmethods.utils import (
    UnknownMethodError,
    check_np_types,
    check_xr_types,
    ensure_devidable,
    get_adjusted_scaling_factor,
    get_cdf,
    get_inverse_of_cdf,
    nan_or_equal,
)


class CMethods:
    """
    The CMethods class serves a collection of bias correction procedures to
    adjust time-series of climate data.

    The following bias correction techniques are available:

    *Scaling-based techniques*:
        * Linear Scaling
        * Variance Scaling
        * Delta (change) Method
    *Distribution-based techniques*:
        * Quantile Mapping
        * Detrended Quantile Mapping
        * Quantile Delta Mapping

    To execute one of those techniques (except for detrended quantile mapping),
    the :func:`CMethods.adjust` func can be used. Please refer to this function
    and the method specific characteristics described in the documentation
    https://python-cmethods.readthedocs.io/en/latest/.

    Except for the Variance Scaling all methods can be applied on both,
    stochastic and non-stochastic variables. The Variance Scaling can only be
    applied on stochastic climate variables.

    - Non-stochastic climate variables are those that can be predicted with
      relative certainty based on factors such as location, elevation, and
      season. Examples of non-stochastic climate variables include air
      temperature, air pressure, and solar radiation.
    - Stochastic climate variables, on the other hand, are those that exhibit a
      high degree of variability and unpredictability, making them difficult to
      forecast accurately. Precipitation is an example of a stochastic climate
      variable because it can vary greatly in timing, intensity, and location
      due to complex atmospheric and meteorological processes.
    """

    MAX_SCALING_FACTOR: int | float = 10

    # ? -----========= L I N E A R - S C A L I N G =========------
    def __linear_scaling(
        self: CMethods,
        obs: NPData,
        simh: NPData,
        simp: NPData,
        kind: str = "+",
        **kwargs: Any,
    ) -> NPData:
        r"""
        **Do not call this function directly, please use :func:`cmethods.CMethods.adjust`**

        See https://python-cmethods.readthedocs.io/en/latest/src/methods.html#linear-scaling
        """
        check_np_types(obs=obs, simh=simh, simp=simp)
        if kind in ADDITIVE:
            return np.array(simp) + (np.nanmean(obs) - np.nanmean(simh))  # Eq. 1
        if kind in MULTIPLICATIVE:
            adj_scaling_factor = get_adjusted_scaling_factor(
                ensure_devidable(
                    np.nanmean(obs), np.nanmean(simh), self.MAX_SCALING_FACTOR
                ),
                kwargs.get("max_scaling_factor", self.MAX_SCALING_FACTOR),
            )
            return np.array(simp) * adj_scaling_factor  # Eq. 2
        raise NotImplementedError(
            f"{kind=} not available for linear_scaling. Use '+' or '*' instead."
        )

    # ? -----========= V A R I A N C E - S C A L I N G =========------

    def __variance_scaling(
        self: CMethods,
        obs: NPData,
        simh: NPData,
        simp: NPData,
        kind: str = "+",
        **kwargs: Any,
    ) -> NPData:
        r"""
        **Do not call this function directly, please use :func:`cmethods.CMethods.adjust`**

        See https://python-cmethods.readthedocs.io/en/latest/src/methods.html#variance-scaling
        """
        check_np_types(obs=obs, simh=simp, simp=simp)
        if kind in ADDITIVE:
            LS_simh = self.__linear_scaling(obs, simh, simh)  # Eq. 1
            LS_simp = self.__linear_scaling(obs, simh, simp)  # Eq. 2

            VS_1_simh = LS_simh - np.nanmean(LS_simh)  # Eq. 3
            VS_1_simp = LS_simp - np.nanmean(LS_simp)  # Eq. 4

            adj_scaling_factor = get_adjusted_scaling_factor(
                ensure_devidable(
                    np.std(np.array(obs)), np.std(VS_1_simh), self.MAX_SCALING_FACTOR
                ),
                kwargs.get("max_scaling_factor", self.MAX_SCALING_FACTOR),
            )

            VS_2_simp = VS_1_simp * adj_scaling_factor  # Eq. 5
            return VS_2_simp + np.nanmean(LS_simp)  # Eq. 6

        raise NotImplementedError(
            f"{kind=} not available for variance_scaling. Use '+' instead."
        )

    # ? -----========= D E L T A - M E T H O D =========------
    def __delta_method(
        self: CMethods,
        obs: NPData,
        simh: NPData,
        simp: NPData,
        kind: str = "+",
        **kwargs: Any,
    ) -> NPData:
        r"""
        **Do not call this function directly, please use :func:`cmethods.CMethods.adjust`**
        See https://python-cmethods.readthedocs.io/en/latest/src/methods.html#delta-method
        """
        check_np_types(obs=obs, simh=simh, simp=simp)

        if kind in ADDITIVE:
            return np.array(obs) + (np.nanmean(simp) - np.nanmean(simh))  # Eq. 1
        if kind in MULTIPLICATIVE:
            adj_scaling_factor = get_adjusted_scaling_factor(
                ensure_devidable(
                    np.nanmean(simp), np.nanmean(simh), self.MAX_SCALING_FACTOR
                ),
                kwargs.get("max_scaling_factor", self.MAX_SCALING_FACTOR),
            )
            return np.array(obs) * adj_scaling_factor  # Eq. 2
        raise NotImplementedError(
            f"{kind=} not available for delta_method. Use '+' or '*' instead."
        )

    # ? -----========= Q U A N T I L E - M A P P I N G =========------
    def __quantile_mapping(
        self: CMethods,
        obs: NPData,
        simh: NPData,
        simp: NPData,
        n_quantiles: int,
        kind: str = "+",
        **kwargs: Any,
    ) -> np.ndarray:
        r"""
        **Do not call this function directly, please use :func:`cmethods.CMethods.adjust`**
        See https://python-cmethods.readthedocs.io/en/latest/src/methods.html#quantile-mapping
        """
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
            f"{kind=} for quantile_mapping is not available. Use '+' or '*' instead."
        )

    # ? -----========= D E T R E N D E D - Q U A N T I L E - M A P P I N G =========------

    def detrended_quantile_mapping(
        self: CMethods,
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
                f"{kind=} for detrended_quantile_mapping is not available. Use '+' or '*' instead."
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
                    m_simp - m_simp_mean + m_simh_mean, xbins, cdf_simh
                )  # Eq. 1
                X = (
                    get_inverse_of_cdf(cdf_obs, epsilon, xbins)
                    + m_simp_mean
                    - m_simh_mean
                )  # Eq. 1

            else:  # kind in cls.MULTIPLICATIVE:
                epsilon = np.interp(  # Eq. 2
                    ensure_devidable(
                        (m_simh_mean * m_simp),
                        m_simp_mean,
                        max_scaling_factor=self.MAX_SCALING_FACTOR,
                    ),
                    xbins,
                    cdf_simh,
                    left=kwargs.get("val_min", 0.0),
                    right=kwargs.get("val_max", None),
                )
                X = np.interp(epsilon, cdf_obs, xbins) * ensure_devidable(
                    m_simp_mean, m_simh_mean, max_scaling_factor=self.MAX_SCALING_FACTOR
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
    #         "Not implemented; please have a look at: https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py "
    #     )

    # ? -----========= Q U A N T I L E - D E L T A - M A P P I N G =========------

    def __quantile_delta_mapping(
        self: CMethods,
        obs: NPData,
        simh: NPData,
        simp: NPData,
        n_quantiles: int,
        kind: str = "+",
        **kwargs: Any,
    ) -> NPData:
        r"""
        **Do not call this function directly, please use :func:`cmethods.CMethods.adjust`**

        See https://python-cmethods.readthedocs.io/en/latest/src/methods.html#quantile-delta-mapping
        """
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

            delta = ensure_devidable(  # Eq. 2.3
                simp,
                get_inverse_of_cdf(cdf_simh, epsilon, xbins),
                max_scaling_factor=self.MAX_SCALING_FACTOR,
            )
            return QDM1 * delta  # Eq. 2.4
        raise NotImplementedError(
            f"{kind=} not available for quantile_delta_mapping. Use '+' or '*' instead."
        )

    def adjust(
        self: CMethods, method: str, obs: XRData, simh: XRData, simp: XRData, **kwargs
    ) -> XRData:
        """
        Function to apply a bias correction technique on 1-and 3-dimensional
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
        :param **kwargs: Any other method-specific parameter (like
            ``n_quantiles`` and ``kind``)
        :type **kwargs: dict
        :return: The bias corrected/adjusted data set
        :rtype: XRData
        """
        check_xr_types(obs=obs, simh=simh, simp=simp)

        if method == "detrended_quantile_mapping":
            raise ValueError(
                "This function is not available for detrended quantile mapping."
                " Please use cmethods.CMethods.detrended_quantile_mapping"
            )

        # No grouped correction | distribution-based technique
        if kwargs.get("group", None) is None:
            return self.__apply_ufunc(method, obs, simh, simp, **kwargs).to_dataset()

        if method not in SCALING_METHODS:
            raise ValueError(
                "Can't use group for distribution based methods"  # except for DQM
            )

        # Grouped correction | scaling-based technique
        group: str = kwargs["group"]
        del kwargs["group"]

        obs_g: list[tuple[int, XRData]] = list(obs.groupby(group))
        simh_g: list[tuple[int, XRData]] = list(simh.groupby(group))
        simp_g: list[tuple[int, XRData]] = list(simp.groupby(group))

        result: Optional[XRData] = None
        for index in range(len(list(obs_g))):
            obs_gds: XRData = obs_g[index][1]
            simh_gds: XRData = simh_g[index][1]
            simp_gds: XRData = simp_g[index][1]

            monthly_result = self.__apply_ufunc(
                method, obs_gds, simh_gds, simp_gds, **kwargs
            )

            if result is None:
                result = monthly_result
            else:
                result = xr.merge([result, monthly_result])

        return result

    def __get_function(self: CMethods, method: str) -> Callable:
        """
        Returns the bias correction function corresponding to the ``method`` name.

        :param method: The method name to get the function for
        :type method: str
        :raises UnknownMethodError: If the function is not implemented
        :return: The function of the corresponding method
        :rtype: function
        """
        if method == "linear_scaling":
            return self.__linear_scaling
        if method == "variance_scaling":
            return self.__variance_scaling
        if method == "delta_method":
            return self.__delta_method
        if method == "quantile_mapping":
            return self.__quantile_mapping
        if method == "quantile_delta_mapping":
            return self.__quantile_delta_mapping
        raise UnknownMethodError(method, METHODS)

    def __apply_ufunc(
        self: CMethods,
        method: str,
        obs: XRData,
        simh: XRData,
        simp: XRData,
        **kwargs: dict,
    ) -> XRData:
        check_xr_types(obs=obs, simh=simh, simp=simp)

        result: XRData = xr.apply_ufunc(
            self.__get_function(method),
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

    def get_available_methods(self: CMethods) -> list[str]:
        """
        Function to return the available adjustment methods of the CMethods class.

        :return: List of available bias correction methods
        :rtype: list[str]

        .. code-block:: python
            :linenos:
            :caption: Example: Get available methods

            >>> from cmethods import CMethods as cm

            >>> cm.get_available_methods()
            [
                "linear_scaling", "variance_scaling", "delta_method",
                "quantile_mapping", "detrended_quantile_mapping", "quantile_delta_mapping"
            ]
        """
        return METHODS
