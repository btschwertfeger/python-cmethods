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

from typing import Any, List, Optional, Union

import numpy as np
import xarray as xr

__author__ = "Benjamin Thomas Schwertfeger"
__copyright__ = __author__
__email__ = "contact@b-schwertfeger.de"
__link__ = "https://github.com/btschwertfeger"
__github__ = "https://github.com/btschwertfeger/python-cmethods"

from cmethods.static import ADDITIVE, METHODS, MULTIPLICATIVE
from cmethods.utils import (
    UnknownMethodError,
    check_types,
    ensure_devidable,
    get_adjusted_scaling_factor,
    get_cdf,
    get_inverse_of_cdf,
    nan_or_equal,
)


class CMethods2:
    MAX_SCALING_FACTOR: Union[int, float] = 10

    # ? -----========= L I N E A R - S C A L I N G =========------
    @classmethod
    def linear_scaling(
        cls,
        obs,
        simh,
        simp,
        kind: str = "+",
        **kwargs: Any,
    ) -> np.ndarray:
        r"""
        The Linear Scaling bias correction technique can be applied on stochastic and
        non-stochastic climate variables to minimize deviations in the mean values
        between predicted and observed time-series of past and future time periods.

        Since the multiplicative scaling can result in very high scaling factors,
        a maximum scaling factor of 10 is set. This can be changed by passing
        another value to the optional `max_scaling_factor` argument.

        This method must be applied on a 1-dimensional data set i.e., there is only one
        time-series passed for each of ``obs``, ``simh``, and ``simp``.

        The Linear Scaling bias correction technique implemented here is based on the
        method described in the equations of Teutschbein, Claudia and Seibert, Jan (2012)
        *"Bias correction of regional climate model simulations for hydrological climate-change
        impact studies: Review and evaluation of different methods"*
        (https://doi.org/10.1016/j.jhydrol.2012.05.052). In the following the equations
        for both additive and multiplicative Linear Scaling are shown:

        **Additive**:

            In Linear Scaling, the long-term monthly mean (:math:`\mu_m`) of the modeled data :math:`X_{sim,h}` is subtracted
            from the long-term monthly mean of the reference data :math:`X_{obs,h}` at time step :math:`i`.
            This difference in month-dependent long-term mean is than added to the value of time step :math:`i`,
            in the time-series that is to be adjusted (:math:`X_{sim,p}`).

            .. math::

                X^{*LS}_{sim,p}(i) = X_{sim,p}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

        **Multiplicative**:

            The multiplicative Linear Scaling differs from the additive variant in such way, that the changes are not computed
            in absolute but in relative values.

            .. math::

                X^{*LS}_{sim,h}(i) = X_{sim,h}(i) \cdot \left[\frac{\mu_{m}(X_{obs,h}(i))}{\mu_{m}(X_{sim,h}(i))}\right]

        :param obs: The reference data set of the control period
            (in most cases the observational data)
        :type obs: xr.core.dataarray.DataArray
        :param simh: The modeled data of the control period
        :type simh: xr.core.dataarray.DataArray
        :param simp: The modeled data of the scenario period (this is the data set
            on which the bias correction takes action)
        :type simp: xr.core.dataarray.DataArray
        :param kind: The kind of the correction, additive for non-stochastic and multiplicative
            for stochastic climate variables, defaults to ``+``
        :type kind: str, optional
        :raises NotImplementedError: If the kind is not in (``+``, ``*``, ``add``, ``mult``)
        :return: The bias-corrected time series
        :rtype: np.ndarray

        .. code-block:: python
            :linenos:
            :caption: Example: Linear Scaling

            >>> import xarray as xr
            >>> from cmethods import CMethods as cm

            >>> # Note: The data sets must contain the dimension "time"
            >>> #       for the respective variable.
            >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
            >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
            >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
            >>> variable = "tas" # temperatures

            >>> ls_adjusted = cm.linear_scaling(
            ...     obs=obs[variable],
            ...     simh=simh[variable],
            ...     simp=simp[variable],
            ...     kind="+"
            ... )
        """
        if kind in ADDITIVE:
            return np.array(simp) + (np.nanmean(obs) - np.nanmean(simh))  # Eq. 1
        if kind in MULTIPLICATIVE:
            adj_scaling_factor = get_adjusted_scaling_factor(
                ensure_devidable(
                    np.nanmean(obs), np.nanmean(simh), cls.MAX_SCALING_FACTOR
                ),
                kwargs.get("max_scaling_factor", cls.MAX_SCALING_FACTOR),
            )
            return np.array(simp) * adj_scaling_factor  # Eq. 2
        raise NotImplementedError(
            f"{kind} not available for linear_scaling. Use '+' or '*' instead."
        )

    # ? -----========= V A R I A N C E - S C A L I N G =========------
    @classmethod
    def variance_scaling(
        cls,
        obs,
        simh,
        simp,
        kind: str = "+",
        **kwargs: Any,
    ) -> np.ndarray:
        r"""
        The Variance Scaling bias correction technique can be applied only on non-stochastic
        climate variables to minimize deviations in the mean and variance
        between predicted and observed time-series of past and future time periods.

        This method must be applied on a 1-dimensional data set i.e., there is only one
        time-series passed for each of ``obs``, ``simh``, and ``simp``.

        The Variance Scaling bias correction technique implemented here is based on the
        method described in the equations of Teutschbein, Claudia and Seibert, Jan (2012)
        *"Bias correction of regional climate model simulations for hydrological climate-change
        impact studies: Review and evaluation of different methods"*
        (https://doi.org/10.1016/j.jhydrol.2012.05.052). In the following the equations
        of the Variance Scaling approach are shown:

        **(1)** First, the modeled data of the control and scenario period must be bias-corrected using
        the additive linear scaling technique. This adjusts the deviation in the mean.

        .. math::

            X^{*LS}_{sim,h}(i) = X_{sim,h}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

            X^{*LS}_{sim,p}(i) = X_{sim,p}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

        **(2)** In the second step, the time-series are shifted to a zero mean. This enables the adjustment
        of the standard deviation in the following step.

        .. math::

            X^{VS(1)}_{sim,h}(i) = X^{*LS}_{sim,h}(i) - \mu_{m}(X^{*LS}_{sim,h}(i))

            X^{VS(1)}_{sim,p}(i) = X^{*LS}_{sim,p}(i) - \mu_{m}(X^{*LS}_{sim,p}(i))

        **(3)** Now the standard deviation (so variance too) can be scaled.

        .. math::

            X^{VS(2)}_{sim,p}(i) = X^{VS(1)}_{sim,p}(i) \cdot \left[\frac{\sigma_{m}(X_{obs,h}(i))}{\sigma_{m}(X^{VS(1)}_{sim,h}(i))}\right]

        **(4)** Finally the previously removed mean is shifted back

        .. math::

            X^{*VS}_{sim,p}(i) = X^{VS(2)}_{sim,p}(i) + \mu_{m}(X^{*LS}_{sim,p}(i))

        :param obs: The reference data set of the control period
            (in most cases the observational data)
        :type obs: xr.core.dataarray.DataArray
        :param simh: The modeled data of the control period
        :type simh: xr.core.dataarray.DataArray
        :param simp: The modeled data of the scenario period (this is the data set
            on which the bias correction takes action)
        :type simp: xr.core.dataarray.DataArray
        :param kind: The kind of the correction, additive for non-stochastic climate variables
            no other kind is available so far, defaults to ``+``
        :type kind: str, optional
        :raises NotImplementedError: If the kind is not in (``+``, ``add``)
        :return: The bias-corrected time series
        :rtype: np.ndarray

        .. code-block:: python
            :linenos:
            :caption: Example: Variance Scaling

            >>> import xarray as xr
            >>> from cmethods import CMethods as cm

            >>> # Note: The data sets must contain the dimension "time"
            >>> #       for the respective variable.
            >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
            >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
            >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
            >>> variable = "tas" # temperatures

            >>> vs_adjusted = cm.variance_scaling(
            ...     obs=obs[variable],
            ...     simh=simh[variable],
            ...     simp=simp[variable],
            ... )
        """
        if kind in ADDITIVE:
            LS_simh = cls.linear_scaling(obs, simh, simh)  # Eq. 1
            LS_simp = cls.linear_scaling(obs, simh, simp)  # Eq. 2

            VS_1_simh = LS_simh - np.nanmean(LS_simh)  # Eq. 3
            VS_1_simp = LS_simp - np.nanmean(LS_simp)  # Eq. 4

            adj_scaling_factor = get_adjusted_scaling_factor(
                ensure_devidable(
                    np.std(np.array(obs)), np.std(VS_1_simh), cls.MAX_SCALING_FACTOR
                ),
                kwargs.get("max_scaling_factor", cls.MAX_SCALING_FACTOR),
            )

            VS_2_simp = VS_1_simp * adj_scaling_factor  # Eq. 5
            return VS_2_simp + np.nanmean(LS_simp)  # Eq. 6

        raise NotImplementedError(
            f"{kind} not available for variance_scaling. Use '+' instead."
        )

    # ? -----========= D E L T A - M E T H O D =========------
    @classmethod
    def delta_method(
        cls,
        obs: xr.core.dataarray.DataArray,
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        kind: str = "+",
        **kwargs: Any,
    ) -> np.ndarray:
        r"""
        The Delta Method bias correction technique can be applied on stochastic and
        non-stochastic climate variables to minimize deviations in the mean values
        between predicted and observed time-series of past and future time periods.

        Since the multiplicative scaling can result in very high scaling factors,
        a maximum scaling factor of 10 is set. This can be changed by passing
        another value to the optional `max_scaling_factor` argument.

        This method must be applied on a 1-dimensional data set i.e., there is only one
        time-series passed for each of ``obs``, ``simh``, and ``simp``.

        The Delta Method bias correction technique implemented here is based on the
        method described in the equations of Beyer, R. and Krapp, M. and Manica, A. (2020)
        *"An empirical evaluation of bias correction methods for paleoclimate simulations"*
        (https://doi.org/10.5194/cp-16-1493-2020). In the following the equations
        for both additive and multiplicative Delta Method are shown:

        **Additive**:

            The Delta Method looks like the Linear Scaling method but the important difference is, that the Delta method
            uses the change between the modeled data instead of the difference between the modeled and reference data of the control
            period. This means that the long-term monthly mean (:math:`\mu_m`) of the modeled data of the control period :math:`T_{sim,h}`
            is subtracted from the long-term monthly mean of the modeled data from the scenario period :math:`T_{sim,p}` at time step :math:`i`.
            This change in month-dependent long-term mean is than added to the long-term monthly mean for time step :math:`i`,
            in the time-series that represents the reference data of the control period (:math:`T_{obs,h}`).

            .. math::

                X^{*DM}_{sim,p}(i) = X_{obs,h}(i) + \mu_{m}(X_{sim,p}(i)) - \mu_{m}(X_{sim,h}(i))

        **Multiplicative**:

            The multiplicative variant behaves like the additive, but with the difference that the change is computed using the relative change
            instead of the absolute change.

            .. math::

                X^{*DM}_{sim,p}(i) = X_{obs,h}(i) \cdot \left[\frac{ \mu_{m}(X_{sim,p}(i)) }{ \mu_{m}(X_{sim,h}(i))}\right]

        :param obs: The reference data set of the control period
            (in most cases the observational data)
        :type obs: xr.core.dataarray.DataArray
        :param simh: The modeled data of the control period
        :type simh: xr.core.dataarray.DataArray
        :param simp: The modeled data of the scenario period (this is the data set
            on which the bias correction takes action)
        :type simp: xr.core.dataarray.DataArray
        :param kind: The kind of the correction, additive for non-stochastic and multiplicative
            for stochastic climate variables, defaults to ``+``
        :type kind: str, optional
        :raises NotImplementedError: If the kind is not in (``+``, ``*``, ``add``, ``mult``)
        :return: The bias-corrected time series
        :rtype: np.ndarray

        .. code-block:: python
            :linenos:
            :caption: Example: Delta Method

            >>> import xarray as xr
            >>> from cmethods import CMethods as cm

            >>> # Note: The data sets must contain the dimension "time"
            >>> #       for the respective variable.
            >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
            >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
            >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
            >>> variable = "tas" # temperatures

            >>> dm_adjusted = cm.delta_method(
            ...     obs=obs[variable],
            ...     simh=simh[variable],
            ...     simp=simp[variable],
            ... )
        """
        if kind in ADDITIVE:
            return np.array(obs) + (np.nanmean(simp) - np.nanmean(simh))  # Eq. 1
        if kind in MULTIPLICATIVE:
            adj_scaling_factor = get_adjusted_scaling_factor(
                ensure_devidable(
                    np.nanmean(simp), np.nanmean(simh), cls.MAX_SCALING_FACTOR
                ),
                kwargs.get("max_scaling_factor", cls.MAX_SCALING_FACTOR),
            )
            return np.array(obs) * adj_scaling_factor  # Eq. 2
        raise NotImplementedError(
            f"{kind} not available for delta_method. Use '+' or '*' instead."
        )

    # ? -----========= Q U A N T I L E - M A P P I N G =========------
    @classmethod
    def quantile_mapping(
        cls,
        obs,
        simh,
        simp,
        n_quantiles: int,
        kind: str = "+",
        **kwargs: Any,
    ) -> np.ndarray:
        r"""
        The Quantile Mapping bias correction technique can be used to minimize distributional
        biases between modeled and observed time-series climate data. Its interval-independent
        behavior ensures that the whole time series is taken into account to redistribute
        its values, based on the distributions of the modeled and observed/reference data of the
        control period.

        This method must be applied on a 1-dimensional data set i.e., there is only one
        time-series passed for each of ``obs``, ``simh``, and ``simp``.

        The Quantile Mapping technique implemented here is based on the equations of
        Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock (2015) *"Bias Correction of GCM
        Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles
        and Extremes?"* (https://doi.org/10.1175/JCLI-D-14-00754.1).

        The regular Quantile Mapping is bounded to the value range of the modeled data
        of the control period. To avoid this, the Detrended Quantile Mapping can be used.

        In the following the equations of Alex J. Cannon (2015) are shown and explained:

        **Additive**:

            .. math::

                X^{*QM}_{sim,p}(i) = F^{-1}_{obs,h} \left\{F_{sim,h}\left[X_{sim,p}(i)\right]\right\}


            The additive quantile mapping procedure consists of inserting the value to be
            adjusted (:math:`X_{sim,p}(i)`) into the cumulative distribution function
            of the modeled data of the control period (:math:`F_{sim,h}`). This determines,
            in which quantile the value to be adjusted can be found in the modeled data of the control period
            The following images show this by using :math:`T` for temperatures.

            .. figure:: ../_static/images/qm-cdf-plot-1.png
                :width: 600
                :align: center
                :alt: Determination of the quantile value

                Fig 1: Inserting :math:`X_{sim,p}(i)` into :math:`F_{sim,h}` to determine the quantile value

            This value, which of course lies between 0 and 1, is subsequently inserted
            into the inverse cumulative distribution function of the reference data of the control period to
            determine the bias-corrected value at time step :math:`i`.

            .. figure:: ../_static/images/qm-cdf-plot-2.png
                :width: 600
                :align: center
                :alt: Determination of the QM bias-corrected value

                Fig 1: Inserting the quantile value into :math:`F^{-1}_{obs,h}` to determine the bias-corrected value for time step :math:`i`

        **Multiplicative**:

            .. math::

                X^{*QM}_{sim,p}(i) = F^{-1}_{obs,h}\Biggl\{F_{sim,h}\left[\frac{\mu{X_{sim,h}} \cdot \mu{X_{sim,p}(i)}}{\mu{X_{sim,p}(i)}}\right]\Biggr\}\frac{\mu{X_{sim,p}(i)}}{\mu{X_{sim,h}}}


        :param obs: The reference data set of the control period
            (in most cases the observational data)
        :type obs: xr.core.dataarray.DataArray
        :param simh: The modeled data of the control period
        :type simh: xr.core.dataarray.DataArray
        :param simp: The modeled data of the scenario period (this is the data set
            on which the bias correction takes action)
        :type simp: xr.core.dataarray.DataArray
        :param n_quantiles: Number of quantiles to respect/use
        :type n_quantiles: int
        :param kind: The kind of the correction, additive for non-stochastic and multiplicative
            for stochastic climate variables, defaults to ``+``
        :type kind: str, optional
        :param val_min: Lower boundary for interpolation (only if ``kind="*"``, default: ``0.0``)
        :type val_min: float, optional
        :param val_max: Upper boundary for interpolation (only if ``kind="*"``, default: ``None``)
        :type val_max: float, optional
        :raises NotImplementedError: If the kind is not in (``+``, ``*``, ``add``, ``mult``)
        :return: The bias-corrected time series
        :rtype: np.ndarray

        .. code-block:: python
            :linenos:
            :caption: Example: Quantile Mapping

            >>> import xarray as xr
            >>> from cmethods import CMethods as cm

            >>> # Note: The data sets must contain the dimension "time"
            >>> #       for the respective variable.
            >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
            >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
            >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
            >>> variable = "tas" # temperatures

            >>> qm_adjusted = cm.quantile_mapping(
            ...     obs=obs[variable],
            ...     simh=simh[variable],
            ...     simp=simp[variable],
            ...     n_quantiles=250
            ... )
        """
        check_types(obs=obs, simh=simh, simp=simp)
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
            f"{kind} for quantile_mapping is not available. Use '+' or '*' instead."
        )

    # @classmethod
    # def detrended_quantile_mapping(
    #     cls,
    #     obs,
    #     simh,
    #     simp,
    #     n_quantiles: int,
    #     kind: str = "+",
    #     **kwargs: Any,
    # ) -> np.ndarray:
    #     r"""
    #     The Detrended Quantile Mapping bias correction technique can be used to minimize distributional
    #     biases between modeled and observed time-series climate data like the regular Quantile Mapping.
    #     Detrending means, that the values of :math:`X_{sim,p}` are shifted to the value range of
    #     :math:`X_{sim,h}` before the regular Quantile Mapping is applied.
    #     After the Quantile Mapping was applied, the mean is shifted back. Since it does not make sens
    #     to take the whole mean to rescale the data, the month-dependent long-term mean is used.

    #     This method must be applied on a 1-dimensional data set i.e., there is only one
    #     time-series passed for each of ``obs``, ``simh``, and ``simp``. Also this method requires
    #     that the time series are groupable by ``time.month``.

    #     The Detrended Quantile Mapping technique implemented here is based on the equations of
    #     Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock (2015) *"Bias Correction of GCM
    #     Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles
    #     and Extremes?"* (https://doi.org/10.1175/JCLI-D-14-00754.1).

    #     In the following the equations of Alex J. Cannon (2015) are shown (without detrending; see QM
    #     for explanations):

    #     **Additive**:

    #         .. math::

    #             X^{*QM}_{sim,p}(i) = F^{-1}_{obs,h} \left\{F_{sim,h}\left[X_{sim,p}(i)\right]\right\}

    #     **Multiplicative**:

    #         .. math::

    #             X^{*QM}_{sim,p}(i) = F^{-1}_{obs,h}\Biggl\{F_{sim,h}\left[\frac{\mu{X_{sim,h}} \cdot \mu{X_{sim,p}(i)}}{\mu{X_{sim,p}(i)}}\right]\Biggr\}\frac{\mu{X_{sim,p}(i)}}{\mu{X_{sim,h}}}

    #     :param obs: The reference data set of the control period
    #         (in most cases the observational data)
    #     :type obs: xr.core.dataarray.DataArray
    #     :param simh: The modeled data of the control period
    #     :type simh: xr.core.dataarray.DataArray
    #     :param simp: The modeled data of the scenario period (this is the data set
    #         on which the bias correction takes action)
    #     :type simp: xr.core.dataarray.DataArray
    #     :param n_quantiles: Number of quantiles to respect/use
    #     :type n_quantiles: int
    #     :param kind: The kind of the correction, additive for non-stochastic and multiplicative
    #         for stochastic climate variables, defaults to ``+``
    #     :type kind: str, optional
    #     :param val_min: Lower boundary for interpolation (only if ``kind="*"``, default: ``0.0``)
    #     :type val_min: float, optional
    #     :param val_max: Upper boundary for interpolation (only if ``kind="*"``, default: ``None``)
    #     :type val_max: float, optional
    #     :raises NotImplementedError: If the kind is not in (``+``, ``*``, ``add``, ``mult``)
    #     :return: The bias-corrected time series
    #     :rtype: np.ndarray

    #     .. code-block:: python
    #         :linenos:
    #         :caption: Example: Quantile Mapping

    #         >>> import xarray as xr
    #         >>> from cmethods import CMethods as cm

    #         >>> # Note: The data sets must contain the dimension "time"
    #         >>> #       for the respective variable.
    #         >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
    #         >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
    #         >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
    #         >>> variable = "tas" # temperatures

    #         >>> qm_adjusted = cm.detrended_quantile_mapping(
    #         ...     obs=obs[variable],
    #         ...     simh=simh[variable],
    #         ...     simp=simp[variable],
    #         ...     n_quantiles=250
    #         ... )
    #     """
    #     if kind not in MULTIPLICATIVE and kind not in ADDITIVE:
    #         raise NotImplementedError(
    #             f"{kind} for detrended_quantile_mapping is not available. Use '+' or '*' instead."
    #         )

    #     check_types(obs=obs, simh=simh, simp=simp)
    #     if not isinstance(n_quantiles, int):
    #         raise TypeError("'n_quantiles' must be type int")
    #     if not isinstance(simp, xr.core.dataarray.DataArray):
    #         raise TypeError("'simp' must be type xarray.core.dataarray.DataArray")

    #     obs, simh = np.array(obs), np.array(simh)

    #     global_max = max(np.nanmax(obs), np.nanmax(simh))
    #     global_min = min(np.nanmin(obs), np.nanmin(simh))

    #     if nan_or_equal(value1=global_max, value2=global_min):
    #         return np.array(simp.values)

    #     wide = abs(global_max - global_min) / n_quantiles
    #     xbins = np.arange(global_min, global_max + wide, wide)

    #     cdf_obs = get_cdf(obs, xbins)
    #     cdf_simh = get_cdf(simh, xbins)

    #     # detrended => shift mean of $X_{sim,p}$ to range of $X_{sim,h}$ to adjust extremes
    #     res = np.zeros(len(simp.values))
    #     for indices in simp.groupby("time.month").groups.values():
    #         # detrended by long-term month
    #         m_simh, m_simp = [], []
    #         for index in indices:
    #             m_simh.append(simh[index])
    #             m_simp.append(simp[index])

    #         m_simh = np.array(m_simh)
    #         m_simp = np.array(m_simp)
    #         m_simh_mean = np.nanmean(m_simh)
    #         m_simp_mean = np.nanmean(m_simp)

    #         if kind in ADDITIVE:
    #             epsilon = np.interp(
    #                 m_simp - m_simp_mean + m_simh_mean, xbins, cdf_simh
    #             )  # Eq. 1
    #             X = (
    #                 get_inverse_of_cdf(cdf_obs, epsilon, xbins)
    #                 + m_simp_mean
    #                 - m_simh_mean
    #             )  # Eq. 1

    #         else:  # kind in MULTIPLICATIVE:
    #             epsilon = np.interp(  # Eq. 2
    #                 ensure_devidable(
    #                     (m_simh_mean * m_simp),
    #                     m_simp_mean,
    #                     max_scaling_factor=cls.MAX_SCALING_FACTOR,
    #                 ),
    #                 xbins,
    #                 cdf_simh,
    #                 left=kwargs.get("val_min", 0.0),
    #                 right=kwargs.get("val_max", None),
    #             )
    #             X = np.interp(epsilon, cdf_obs, xbins) * ensure_devidable(
    #                 m_simp_mean,
    #                 m_simh_mean,
    #                 max_scaling_factor=cls.MAX_SCALING_FACTOR,
    #             )  # Eq. 2
    #         for i, index in enumerate(indices):
    #             res[index] = X[i]
    #     return res

    # ? -----========= E M P I R I C A L - Q U A N T I L E - M A P P I N G =========------
    @classmethod
    def empirical_quantile_mapping(
        cls,
        obs,
        simh,
        simp,
        n_quantiles: int = 100,
        extrapolate: Optional[str] = None,
        **kwargs: Any,
    ) -> xr.core.dataarray.DataArray:
        """
        Method to adjust 1-dimensional climate data by empirical quantile mapping

        :param obs: The reference data set of the control period
            (in most cases the observational data)
        :type obs: xr.core.dataarray.DataArray
        :param simh: The modeled data of the control period
        :type simh: xr.core.dataarray.DataArray
        :param simp: The modeled data of the scenario period (this is the data set
            on which the bias correction takes action)
        :type simp: xr.core.dataarray.DataArray
        :param n_quantiles: Number of quantiles to respect/use, defaults to ``100``
        :type n_quantiles: int, optional
        :type kind: str, optional
        :param extrapolate: Bounded range or extrapolate, defaults to ``None``
        :type extrapolate: str | None
        :return: The bias-corrected time series
        :rtype: xr.core.dataarray.DataArray
        :raises NotImplementedError: This method is not implemented
        """
        raise NotImplementedError(
            "Not implemented; please have a look at: https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py "
        )

    # ? -----========= Q U A N T I L E - D E L T A - M A P P I N G =========------
    @classmethod
    def quantile_delta_mapping(
        cls,
        obs,
        simh,
        simp,
        n_quantiles: int,
        kind: str = "+",
        **kwargs: Any,
    ) -> np.ndarray:
        r"""
        The Quantile Delta Mapping bias correction technique can be used to minimize distributional
        biases between modeled and observed time-series climate data. Its interval-independent
        behavior ensures that the whole time series is taken into account to redistribute
        its values, based on the distributions of the modeled and observed/reference data of the
        control period. In contrast to the regular Quantile Mapping (:func:`cmethods.CMethods.quantile_mapping`)
        the Quantile Delta Mapping also takes the change between the modeled data of the control and scenario
        period into account.

        This method must be applied on a 1-dimensional data set i.e., there is only one
        time-series passed for each of ``obs``, ``simh``, and ``simp``.

        The Quantile Delta Mapping technique implemented here is based on the equations of
        Tong, Y., Gao, X., Han, Z. et al. (2021) *"Bias correction of temperature and precipitation
        over China for RCM simulations using the QM and QDM methods"*. Clim Dyn 57, 1425-1443
        (https://doi.org/10.1007/s00382-020-05447-4). In the following the additive and multiplicative
        variant are shown.

        **Additive**:

            **(1.1)** In the first step the quantile value of the time step :math:`i` to adjust is stored in
            :math:`\varepsilon(i)`.

            .. math::

                \varepsilon(i) = F_{sim,p}\left[X_{sim,p}(i)\right], \hspace{1em} \varepsilon(i)\in\{0,1\}

            **(1.2)** The bias corrected value at time step :math:`i` is now determined by inserting the
            quantile value into the inverse cummulative distribution function of the reference data of the control
            period. This results in a bias corrected value for time step :math:`i` but still without taking the
            change in modeled data into account.

            .. math::

                X^{QDM(1)}_{sim,p}(i) = F^{-1}_{obs,h}\left[\varepsilon(i)\right]

            **(1.3)** The :math:`\Delta(i)` represents the absolute change in quantiles between the modeled value
            in the control and scenario period.

            .. math::

                 \Delta(i) & = F^{-1}_{sim,p}\left[\varepsilon(i)\right] - F^{-1}_{sim,h}\left[\varepsilon(i)\right] \\[1pt]
                           & = X_{sim,p}(i) - F^{-1}_{sim,h}\left\{F^{}_{sim,p}\left[X_{sim,p}(i)\right]\right\}

            **(1.4)** Finally the previously calculated change can be added to the bias-corrected value.

            .. math::

                X^{*QDM}_{sim,p}(i) = X^{QDM(1)}_{sim,p}(i) + \Delta(i)

        **Multiplicative**:

            The first two steps of the multiplicative Quantile Delta Mapping bias correction technique are the
            same as for the additive variant.

            **(2.3)** The :math:`\Delta(i)` in the multiplicative Quantile Delta Mapping is calulated like the
            additive variant, but using the relative than the absolute change.

                .. math::

                    \Delta(i) & = \frac{ F^{-1}_{sim,p}\left[\varepsilon(i)\right] }{ F^{-1}_{sim,h}\left[\varepsilon(i)\right] } \\[1pt]
                              & = \frac{ X_{sim,p}(i) }{ F^{-1}_{sim,h}\left\{F_{sim,p}\left[X_{sim,p}(i)\right]\right\} }

            **(2.4)** The relative change between the modeled data of the control and scenario period is than
            multiplicated with the bias-corrected value (see **1.2**).

                .. math::

                    X^{*QDM}_{sim,p}(i) = X^{QDM(1)}_{sim,p}(i) \cdot \Delta(i)

        :param obs: The reference data set of the control period
            (in most cases the observational data)
        :type obs: xr.core.dataarray.DataArray
        :param simh: The modeled data of the control period
        :type simh: xr.core.dataarray.DataArray
        :param simp: The modeled data of the scenario period (this is the data set
            on which the bias correction takes action)
        :type simp: xr.core.dataarray.DataArray
        :param n_quantiles: Number of quantiles to respect/use
        :type n_quantiles: int
        :param kind: The kind of the correction, additive for non-stochastic and multiplicative
            for non-stochastic climate variables, defaults to ``+``
        :type kind: str, optional
        :raises NotImplementedError: If the kind is not in (``+``, ``*``, ``add``, ``mult``)
        :return: The bias-corrected time series
        :rtype: np.ndarray

        .. code-block:: python
            :linenos:
            :caption: Example: Quantile Delta Mapping

            >>> import xarray as xr
            >>> from cmethods import CMethods as cm

            >>> # Note: The data sets must contain the dimension "time"
            >>> #       for the respective variable.
            >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
            >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
            >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
            >>> variable = "tas" # temperatures

            >>> qdm_adjusted = cm.quantile_delta_mapping(
            ...     obs=obs[variable],
            ...     simh=simh[variable],
            ...     simp=simp[variable],
            ...     n_quantiles=250
            ... )
        """
        check_types(obs=obs, simh=simh, simp=simp)

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

            # calculate exact cdf values of $F_{sim,p}[T_{sim,p}(t)]$
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
                max_scaling_factor=cls.MAX_SCALING_FACTOR,
            )
            return QDM1 * delta  # Eq. 2.4
        raise NotImplementedError(
            f"{kind} not available for quantile_delta_mapping. Use '+' or '*' instead."
        )

    @classmethod
    def adjust(cls, method, obs, simh, simp, **kwargs):
        if kwargs.get("group", None) is not None:
            group = kwargs["group"]
            del kwargs["group"]

            obs_g = list(obs.groupby(group))
            simh_g = list(simh.groupby(group))
            simp_g = list(simp.groupby(group))

            result = None
            for index in range(len(list(obs_g))):
                obs_gds = obs_g[index][1]
                simh_gds = simh_g[index][1]
                simp_gds = simp_g[index][1]

                monthly_result = cls.apply_ufunc(
                    method, obs_gds, simh_gds, simp_gds, **kwargs
                )
                if result is None:
                    result = monthly_result
                else:
                    result = xr.concat([result, monthly_result], dim="time")
            return result
        return cls.apply_ufunc(method, obs, simh, simp, **kwargs)

    @classmethod
    def get_function(cls, method):
        """
        Returns the bias correction function corresponding to the ``method`` name.

        :param method: The method name to get the function for
        :type method: str
        :raises UnknownMethodError: If the function is not implemented
        :return: The function of the corresponding method
        :rtype: function
        """
        if method == "linear_scaling":
            return cls.linear_scaling
        if method == "variance_scaling":
            return cls.variance_scaling
        if method == "delta_method":
            return cls.delta_method
        if method == "quantile_mapping":
            return cls.quantile_mapping
        if method == "detrended_quantile_mapping":
            return cls.detrended_quantile_mapping
        if method == "empirical_quantile_mapping":
            return cls.empirical_quantile_mapping
        if method == "quantile_delta_mapping":
            return cls.quantile_delta_mapping
        raise UnknownMethodError(method, METHODS)

    @classmethod
    def apply_ufunc(cls, method, obs, simh, simp, **kwargs):
        result = xr.apply_ufunc(
            cls.get_function(method),
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

    @classmethod
    def get_available_methods(cls) -> List[str]:
        """
        Function to return the available adjustment methods of the CMethods class.

        :return: List of available bias correction methods
        :rtype: List[str]

        .. code-block:: python
            :linenos:
            :caption: Example: Get available methods

            >>> from cmethods import CMethods as cm

            >>> cm.get_available_methods()
            [
                "linear_scaling", "variance_scaling", "delta_method",
                "quantile_mapping", "quantile_delta_mapping"
            ]
        """
        return METHODS
