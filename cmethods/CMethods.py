#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# Github: https://github.com/btschwertfeger
#

"""Module that implements the bias adjustment procedutes"""
import multiprocessing
from typing import List, Union

import numpy as np
import xarray as xr
from tqdm import tqdm

__descrption__ = "Module to help adjusting bias estimated in climate data"
__author__ = "Benjamin Thomas Schwertfeger"
__copyright__ = __author__
__email__ = "development@b-schwertfeger.de"
__link__ = "https://b-schwertfeger.de"
__github__ = "https://github.com/btschwertfeger/Bias-Adjustment-Python"
__description__ = r"                                                                     \
    Class / Script / Methods to adjust bias estimated in climate data                   \
                                                                                        \
    T = Temperatures ($T$)                                                              \
    X = Some climate variable ($X$)                                                     \
    h = historical                                                                      \
    p = scenario; future; predicted                                                     \
    obs = observed data ($T_{obs,h}$)                                                   \
    simh = modeled data with same time period as obs ($T_{sim,h}$)                      \
    simp = data to correct (predicted simulated data) ($T_{sim,p}$)                     \
                                                                                        \
                                                                                        \
    F = Cumulative Distribution Function                                                \
    \mu = mean                                                                          \
    \sigma = standard deviation                                                         \
    i = index                                                                           \
    _{m} = long-term monthly interval                                                   \
"


class UnknownMethodError(Exception):
    """Exception raised for errors if unknown method called in CMethods class.

    ----- P A R A M E T E R S -----
        method (str): Input method name which caused the error
        available_methods (str): List of available methods
    """

    def __init__(self, method: str, available_methods: list):
        super().__init__(
            f'Unknown method "{method}"! Available methods: {available_methods}'
        )


class CMethods:
    """Class used to adjust timeseries climate data."""

    SCALING_METHODS = ["linear_scaling", "variance_scaling", "delta_method"]
    DISTRIBUTION_METHODS = ["quantile_mapping", "quantile_delta_mapping"]

    CUSTOM_METHODS = SCALING_METHODS + DISTRIBUTION_METHODS
    METHODS = CUSTOM_METHODS

    ADDITIVE = ["+", "add"]
    MULTIPLICATIVE = ["*", "mult"]

    MAX_SCALING_FACTOR = 10

    def __init__(self):
        pass

    @classmethod
    def get_available_methods(cls) -> list:
        """Function to return the available adjustment methods"""
        return cls.METHODS

    @classmethod
    def get_function(cls, method: str):
        """Returns the method by name"""
        if method == "linear_scaling":
            return cls.linear_scaling
        if method == "variance_scaling":
            return cls.variance_scaling
        if method == "delta_method":
            return cls.delta_method
        if method == "quantile_mapping":
            return cls.quantile_mapping
        if method == "empirical_quantile_mapping":
            return cls.empirical_quantile_mapping
        if method == "quantile_delta_mapping":
            return cls.quantile_delta_mapping
        raise UnknownMethodError(method, cls.METHODS)

    @classmethod
    def adjust_3d(
        cls,
        method: str,
        obs: xr.core.dataarray.DataArray,
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        n_quantiles: int = 100,
        kind: str = "+",
        group: Union[str, None] = None,
        n_jobs: int = 1,
        **kwargs,
    ) -> xr.core.dataarray.Dataset:
        r"""Function to adjust 3 dimensional climate data

        Note: obs, simh and simp has to be in the format (time, lat, lon)

        ----- P A R A M E T E R S -----

            method (str): adjustment method (see available methods by calling classmethod "get_available_methods")

            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data to adjust

            n_quantiles (int): Number of quantiles to involve
            kind (str): Kind of adjustment ('+' or '*'), default: '+' (always use '+' for temperature)
            group (str): Group data by (e.g.: 'time.month', 'time.dayofyear')
            n_jobs (int): Use n processes, default: 1

        ----- R E T U R N -----

            xarray.core.dataarray.Dataset: Adjusted dataset

        ----- E X A M P L E -----
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simp = xarray.open_dataset('path/to/simulated_future/data.nc')
            > variable = 'tas'

            > adjusted_data = CMethods().adjust_3d(
                method = 'quantile_delta_mapping',
                obs = obs[variable],
                simh = simh[variable],
                simp = simp[variable],
                n_quantiles = 250,
                n_jobs = 4
            )
        """

        obs = obs.transpose("lat", "lon", "time")
        simh = simh.transpose("lat", "lon", "time")
        simp = simp.transpose("lat", "lon", "time").load()

        result = simp.copy(deep=True)
        len_lat, len_lon = len(simp.lat), len(simp.lon)

        if method in cls.CUSTOM_METHODS:
            if n_jobs == 1:
                method = cls.get_function(method)
                for lat in tqdm(range(len_lat)):
                    for lon in range(len_lon):
                        result[lat, lon] = method(
                            obs=obs[lat, lon],
                            simh=simh[lat, lon],
                            simp=simp[lat, lon],
                            group=group,
                            kind=kind,
                            n_quantiles=n_quantiles,
                            **kwargs,
                        )
            else:
                try:
                    pool = multiprocessing.Pool(processes=n_jobs)
                    # with multiprocessing.Pool(processes=n_jobs) as pool:
                    # conntext manager is not use because than the coverage does not work.
                    params: List[dict] = [
                        {
                            "method": method,
                            "obs": obs[lat],
                            "simh": simh[lat],
                            "simp": simp[lat],
                            "group": group,
                            "kind": kind,
                            "n_quaniles": n_quantiles,
                            "kwargs": kwargs,
                        }
                        for lat in range(len_lat)
                    ]
                    for lat, corrected in enumerate(pool.map(cls.pool_adjust, params)):
                        result[lat] = corrected
                finally:
                    pool.close()
                    pool.join()

            return result.transpose("time", "lat", "lon")
        raise UnknownMethodError(method, cls.METHODS)

    @classmethod
    def pool_adjust(cls, params: dict) -> xr.core.dataarray.DataArray:
        """Adjustment along longitude for one specific latitude
        used by cls.adjust_3d as callbackfunction for multiprocessing.Pool
        """
        kwargs = params.get("kwargs", {})

        result = params["simp"].copy(deep=True).load()
        len_lon = len(params["obs"].lon)

        func_adjustment = None
        if params["method"] in cls.CUSTOM_METHODS:
            func_adjustment = cls.get_function(params["method"])
            kwargs["n_quantiles"] = params.get("n_quantiles", 100)
            kwargs["kind"] = params.get("kind", "+")
            for lon in range(len_lon):
                result[lon] = func_adjustment(
                    obs=params["obs"][lon],
                    simh=params["simh"][lon],
                    simp=params["simp"][lon],
                    group=params.get("group", None),
                    **kwargs,
                )
            return np.array(result)

        raise UnknownMethodError(params["method"], cls.METHODS)

    @classmethod
    def grouped_correction(
        cls,
        method: str,
        obs: xr.core.dataarray.DataArray,
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        group: str,
        kind: str = "+",
        **kwargs,
    ) -> xr.core.dataarray.DataArray:
        """Method to adjust 1 dimensional climate data while respecting adjustment groups.

        ----- P A R A M E T E R S -----
            method (str): adjustment method name
            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            method (str): Scaling method (e.g.: 'linear_scaling')
            group (str): [optional]  Group / Period (either: 'time.month', 'time.year' or 'time.dayofyear')

        ----- R E T U R N -----

            xarray.core.dataarray.DataArray: Adjusted data

        """

        func_adjustment = cls.get_function(method)
        result = simp.copy(deep=True).load()
        groups = simh.groupby(group).groups

        for month in groups.keys():
            m_obs, m_simh, m_simp = [], [], []

            for i in groups[month]:
                m_obs.append(obs[i])
                m_simh.append(simh[i])
                m_simp.append(simp[i])

            computed_result = func_adjustment(
                obs=m_obs, simh=m_simh, simp=m_simp, kind=kind, group=None, **kwargs
            )
            for i, index in enumerate(groups[month]):
                result[index] = computed_result[i]

        return result

    # ? -----========= L I N E A R - S C A L I N G =========------
    @classmethod
    def linear_scaling(
        cls,
        obs: xr.core.dataarray.DataArray,
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        group: Union[str, None] = "time.month",
        kind: str = "+",
        **kwargs,
    ) -> xr.core.dataarray.DataArray:
        r"""Method to adjust 1 dimensional climate data by the linear scaling method.

        ----- P A R A M E T E R S -----

            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            group (str): [optional] Group / Period (e.g.: 'time.month')
            kind (str): [optional] '+' or '*', default: '+'

        ----- R E T U R N -----

            xarray.core.dataarray.DataArray: Adjusted data

        ----- E X A M P L E -----
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > simp = xarray.open_dataset('path/to/predicted/data.nc')
            > variable = 'tas'

            > result = CMethods().linear_scaling(
            >    obs=obs[variable],
            >    simh=simh[variable],
            >    simp=simp[variable],
            >    group='time.month' # optional, this is default here
            >)

        ----- E Q U A T I O N S -----
            Add ('+'):
                (1.)    X^{*LS}_{sim,p}(i) = X_{sim,p}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))
            Mult ('*'):
                (2.)    X^{*LS}_{sim,h}(i) = X_{sim,h}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

        ----- R E F E R E N C E S -----

            Based on the equations of Teutschbein, Claudia and Seibert, Jan (2012) Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods
            https://doi.org/10.1016/j.jhydrol.2012.05.052

        """
        if group is not None:
            return cls.grouped_correction(
                method="linear_scaling",
                obs=obs,
                simh=simh,
                simp=simp,
                group=group,
                kind=kind,
                **kwargs,
            )
        if kind in cls.ADDITIVE:
            return np.array(simp) + (np.nanmean(obs) - np.nanmean(simh))  # Eq. 1
        if kind in cls.MULTIPLICATIVE:
            adj_scaling_factor = cls.get_adjusted_scaling_factor(
                np.nanmean(obs) / np.nanmean(simh),
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
        obs: xr.core.dataarray.DataArray,
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        group: Union[str, None] = "time.month",
        kind: str = "+",
        **kwargs,
    ) -> xr.core.dataarray.DataArray:
        r"""Method to adjust 1 dimensional climate data by variance scaling method.

        ----- P A R A M E T E R S -----

            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            group (str): [optional] Group / Period (e.g.: 'time.month')
            kind (str): '+' or '*', default: '+' # '*' is not implemented

        ----- R E T U R N -----

            xarray.core.dataarray.DataArray: Adjusted data

        ----- E X A M P L E -----
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > simp = xarray.open_dataset('path/to/predicted/data.nc')
            > variable = 'tas'
            > result = CMethods().variance_scaling(obs=obs[variable], simh=simh[variable], simp=simp[variable] group='time.month')

        ------ E Q U A T I O N S -----

            (1.) X^{*LS}_{sim,h}(i) = X_{sim,h}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))
            (2.) X^{*LS}_{sim,p}(i) = X_{sim,p}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

            (3.) X^{VS(1)}_{sim,h}(i) = X^{*LS}_{sim,h}(i) - \mu_{m}(X^{*LS}_{sim,h}(i))
            (4.) X^{VS(1)}_{sim,p}(i) = X^{*LS}_{sim,p}(i) - \mu_{m}(X^{*LS}_{sim,p}(i))

            (5.) X^{VS(2)}_{sim,p}(i) = X^{VS(1)}_{sim,p}(i) \cdot \left[\frac{\sigma_{m}(X_{obs,h}(i))}{\sigma_{m}(X^{VS(1)}_{sim,h}(i))}\right]

            (6.) X^{*VS}_{sim,p}(i) = X^{VS(2)}_{sim,p}(i) + \mu_{m}(X^{*LS}_{sim,p}(i))

        ----- R E F E R E N C E S -----

            Based on the equations of Teutschbein, Claudia and Seibert, Jan (2012) Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods
            https://doi.org/10.1016/j.jhydrol.2012.05.052
        """
        if group is not None:
            return cls.grouped_correction(
                method="variance_scaling",
                obs=obs,
                simh=simh,
                simp=simp,
                group=group,
                kind=kind,
                **kwargs,
            )
        if kind in cls.ADDITIVE:
            LS_simh = cls.linear_scaling(obs, simh, simh, group=None)  # Eq. 1
            LS_simp = cls.linear_scaling(obs, simh, simp, group=None)  # Eq. 2

            VS_1_simh = LS_simh - np.nanmean(LS_simh)  # Eq. 3
            VS_1_simp = LS_simp - np.nanmean(LS_simp)  # Eq. 4

            adj_scaling_factor = cls.get_adjusted_scaling_factor(
                np.std(obs) / np.std(VS_1_simh),
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
        group: Union[str, None] = "time.month",
        kind: str = "+",
        **kwargs,
    ) -> xr.core.dataarray.DataArray:
        r"""Method to adjust 1 dimensional climate data by delta method.

        ----- P A R A M E T E R S -----

            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            group (str): [optional] Group / Period (e.g.: 'time.month')
            kind (str): '+' or '*', default: '+'

        ----- R E T U R N -----

            xarray.core.dataarray.DataArray: Adjusted data

        ----- E X A M P L E -----
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > simp = xarray.open_dataset('path/to/predicted/data.nc')
            > variable = 'tas'
            > result = CMethods().delta_method(obs=obs[variable], simh=simh[variable], group='time.month')

        ------ E Q U A T I O N S -----

            Add (+):
                (1.) X^{*DM}_{sim,p}(i) = X_{obs,h}(i) + (\mu_{m}(X_{sim,p}(i)) - \mu_{m}(X_{sim,h}(i)))
            Mult (*):
                (2.) X^{*DM}_{sim,p}(i) = X_{obs,h}(i) \cdot \frac{ \mu_{m}(X_{sim,p}(i)) }{ \mu_{m}(X_{sim,h}(i))}

        ----- R E F E R E N C E S -----
            Beyer, R. and Krapp, M. and Manica, A.: An empirical evaluation of bias correction methods for palaeoclimate simulations (https://doi.org/10.5194/cp-16-1493-2020)

            and
            https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py

        """

        if group is not None:
            return cls.grouped_correction(
                method="delta_method",
                obs=obs,
                simh=simh,
                simp=simp,
                group=group,
                kind=kind,
                **kwargs,
            )
        if kind in cls.ADDITIVE:
            return np.array(obs) + (np.nanmean(simp) - np.nanmean(simh))  # Eq. 1
        if kind in cls.MULTIPLICATIVE:
            adj_scaling_factor = cls.get_adjusted_scaling_factor(
                np.nanmean(simp) / np.nanmean(simh),
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
        obs: xr.core.dataarray.DataArray,
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        n_quantiles: int,
        # group: Union[str, None] = None,
        kind: str = "+",
        **kwargs,
    ) -> xr.core.dataarray.DataArray:
        r"""Quantile Mapping Bias Correction

        ----- P A R A M E T E R S -----

            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            n_quantiles (int): number of quantiles to use
            kind (str): '+' or '*', default: '+'
            detrended (bool): [optional] detrend by shifting mean on long term basis

        ----- R E T U R N -----

        xarray.core.dataarray.DataArray: Adjusted data

        ------ E Q U A T I O N S -----
            Add (+):
                (1.) X^{*QM}_{sim,p}(i) = F^{-1}_{obs,h} \left\{F_{sim,h}\left[X_{sim,p}(i)\right]\right\}
            Mult (*):
                (2.) X^{*QM}_{sim,p}(i) = F^{-1}_{obs,h}
                        \Biggl\{
                            F_{sim,h}\left[
                                \frac{
                                    \mu{X_{sim,h}} \mu{X_{sim,p}(i)}
                                }{
                                    \mu{X_{sim,p}(i)}
                                }
                            \right]
                        \Biggr\}
                        \frac{
                            \mu{X_{sim,p}(i)}
                        }{
                            \mu{X_{sim,h}}
                        }

        ----- R E F E R E N C E S -----
            Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes?
            https://doi.org/10.1175/JCLI-D-14-00754.1)
        """

        res = simp.copy(deep=True)
        obs, simh, simp = np.array(obs), np.array(simh), np.array(simp)

        global_max = max(np.amax(obs), np.amax(simh))
        global_min = min(np.amin(obs), np.amin(simh))
        wide = abs(global_max - global_min) / n_quantiles
        xbins = np.arange(global_min, global_max + wide, wide)

        cdf_obs = cls.get_cdf(obs, xbins)
        cdf_simh = cls.get_cdf(simh, xbins)

        if kwargs.get("detrended", False):
            # detrended => shift mean of $X_{sim,p}$ to range of $X_{sim,h}$ to adjust extremes
            for _, idxs in res.groupby("time.month").groups.items():
                m_simh, m_simp = [], []
                for idx in idxs:
                    m_simh.append(simh[idx])
                    m_simp.append(simp[idx])

                m_simh = np.array(m_simh)
                m_simp = np.array(m_simp)
                m_simh_mean = np.nanmean(m_simh)
                m_simp_mean = np.nanmean(m_simp)

                if kind in cls.ADDITIVE:
                    epsilon = np.interp(
                        m_simp - m_simp_mean + m_simh_mean, xbins, cdf_simh
                    )  # Eq. 1
                    X = (
                        cls.get_inverse_of_cdf(cdf_obs, epsilon, xbins)
                        + m_simp_mean
                        - m_simh_mean
                    )  # Eq. 1

                elif kind in cls.MULTIPLICATIVE:
                    epsilon = np.interp(  # Eq. 2
                        (m_simh_mean * m_simp) / m_simp_mean,
                        xbins,
                        cdf_simh,
                        left=kwargs.get("val_min", 0.0),
                        right=kwargs.get("val_max", None),
                    )
                    X = np.interp(epsilon, cdf_obs, xbins) * (
                        m_simp_mean / m_simh_mean
                    )  # Eq. 2
                for i, idx in enumerate(idxs):
                    res.values[idx] = X[i]
            return res

        if kind in cls.ADDITIVE:
            epsilon = np.interp(simp, xbins, cdf_simh)  # Eq. 1
            res.values = cls.get_inverse_of_cdf(cdf_obs, epsilon, xbins)  # Eq. 1
            return res

        if kind in cls.MULTIPLICATIVE:
            epsilon = np.interp(  # Eq. 2
                simp,
                xbins,
                cdf_simh,
                left=kwargs.get("val_min", 0.0),
                right=kwargs.get("val_max", None),
            )
            res.values = cls.get_inverse_of_cdf(cdf_obs, epsilon, xbins)  # Eq. 2
            return res

        raise NotImplementedError(
            f"{kind} for quantile_mapping is not available. Use '+' or '*' instead."
        )

    # ? -----========= E M P I R I C A L - Q U A N T I L E - M A P P I N G =========------
    @classmethod
    def empirical_quantile_mapping(
        cls,
        obs: xr.core.dataarray.DataArray,
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        n_quantiles: int = 10,
        extrapolate: Union[str, None] = None,
        **kwargs,
    ) -> xr.core.dataarray.DataArray:
        """Method to adjust 1 dimensional climate data by empirical quantile mapping"""
        raise NotImplementedError(
            "Not implemented; please have a look at: https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py "
        )

    # ? -----========= Q U A N T I L E - D E L T A - M A P P I N G =========------
    @classmethod
    def quantile_delta_mapping(
        cls,
        obs: xr.core.dataarray.DataArray,
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        n_quantiles: int,
        kind: str = "+",
        **kwargs,
    ) -> xr.core.dataarray.DataArray:
        r"""Quantile Delta Mapping bias adjustment

        ----- P A R A M E T E R S -----

            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            n_quantiles (int): number of quantiles to use
            kind (str): '+' or '*', default: '+'
            global_min (float): this parameter can be set when kind == '*' to define a custom lower limit. Otherwise 0.0 is used.

        ----- R E T U R N -----

            xarray.core.dataarray.DataArray: Adjusted data

        ------ E Q U A T I O N S -----

            Add (+):
                (1.1) \varepsilon(i) = F_{sim,p}\left[X_{sim,p}(i)\right], \hspace{1em} \varepsilon(i)\in\{0,1\}

                (1.2) X^{QDM(1)}_{sim,p}(i) = F^{-1}_{obs,h}\left[\varepsilon(i)\right]

                (1.3) \Delta(i) & = F^{-1}_{sim,p}\left[\varepsilon(i)\right] - F^{-1}_{sim,h}\left[\varepsilon(i)\right] \\[1pt]
                                & = X_{sim,p}(i) - F^{-1}_{sim,h}\left\{F^{}_{sim,p}\left[X_{sim,p}(i)\right]\right\}

                (1.4) X^{*QDM}_{sim,p}(i) = X^{QDM(1)}_{sim,p}(i) + \Delta(i)

            Mult (*):
                (1.1) --//--

                (1.2) --//--

                (2.3) \Delta(i) & = \frac{ F^{-1}_{sim,p}\left[\varepsilon(i)\right] }{ F^{-1}_{sim,h}\left[\varepsilon(i)\right] } \\[1pt]
                                & = \frac{ X_{sim,p}(i) }{ F^{-1}_{sim,h}\left\{F^{}_{sim,p}\left[X_{sim,p}(i)\right]\right\} }

                (2.4) X^{*QDM}_{sim,p}(i) = X^{QDM(1)}_{sim,p}(i) \cdot \Delta(i)

        ----- R E F E R E N C E S -----
            Tong, Y., Gao, X., Han, Z. et al. Bias correction of temperature and precipitation over China for RCM simulations using the QM and QDM methods. Clim Dyn 57, 1425–1443 (2021).
            https://doi.org/10.1007/s00382-020-05447-4

        """

        if kind in cls.ADDITIVE:
            res = simp.copy(deep=True)
            obs, simh, simp = (
                np.array(obs),
                np.array(simh),
                np.array(simp),
            )  # to achieve higher accuracy
            global_max = kwargs.get("global_max", max(np.amax(obs), np.amax(simh)))
            global_min = kwargs.get("global_min", min(np.amin(obs), np.amin(simh)))
            wide = abs(global_max - global_min) / n_quantiles
            xbins = np.arange(global_min, global_max + wide, wide)

            cdf_obs = cls.get_cdf(obs, xbins)
            cdf_simh = cls.get_cdf(simh, xbins)
            cdf_simp = cls.get_cdf(simp, xbins)

            # calculate exact cdf values of $F_{sim,p}[T_{sim,p}(t)]$
            epsilon = np.interp(simp, xbins, cdf_simp)  # Eq. 1.1
            QDM1 = cls.get_inverse_of_cdf(cdf_obs, epsilon, xbins)  # Eq. 1.2
            delta = simp - cls.get_inverse_of_cdf(cdf_simh, epsilon, xbins)  # Eq. 1.3
            res.values = QDM1 + delta  # Eq. 1.4
            return res

        if kind in cls.MULTIPLICATIVE:
            res = simp.copy(deep=True)
            obs, simh, simp = np.array(obs), np.array(simh), np.array(simp)
            global_max = kwargs.get("global_max", max(np.amax(obs), np.amax(simh)))
            wide = global_max / n_quantiles
            xbins = np.arange(kwargs.get("global_min", 0.0), global_max + wide, wide)

            cdf_obs = cls.get_cdf(obs, xbins)
            cdf_simh = cls.get_cdf(simh, xbins)
            cdf_simp = cls.get_cdf(simp, xbins)

            epsilon = np.interp(simp, xbins, cdf_simp)  # Eq. 1.1
            QDM1 = cls.get_inverse_of_cdf(cdf_obs, epsilon, xbins)  # Eq. 1.2

            with np.errstate(divide="ignore", invalid="ignore"):
                delta = simp / cls.get_inverse_of_cdf(
                    cdf_simh, epsilon, xbins
                )  # Eq. 2.3
            delta[np.isnan(delta)] = 0

            res.values = QDM1 * delta  # Eq. 2.4
            return res
        raise ValueError(f"Unknown kind {kind}!")

    # * -----========= G E N E R A L  =========------
    @staticmethod
    def get_pdf(a: Union[list, np.array], xbins: Union[list, np.array]) -> np.array:
        """returns the probability density function of a based on xbins ($P(x)$)"""
        pdf, _ = np.histogram(a, xbins)
        return pdf

    @staticmethod
    def get_cdf(a: Union[list, np.array], xbins: Union[list, np.array]) -> np.array:
        """returns the cummulative distribution function of a based on xbins ($F_{a}$)"""
        pdf, _ = np.histogram(a, xbins)
        return np.insert(np.cumsum(pdf), 0, 0.0)

    @staticmethod
    def get_inverse_of_cdf(
        base_cdf: Union[list, np.array],
        insert_cdf: Union[list, np.array],
        xbins: Union[list, np.array],
    ) -> np.array:
        r"""returns the inverse cummulative distribution function of base_cdf ($F_{base_cdf}\left[insert_cdf\right])$"""
        return np.interp(insert_cdf, base_cdf, xbins)

    @staticmethod
    def get_adjusted_scaling_factor(
        factor: Union[int, float], max_scaling_factor: Union[int, float]
    ) -> float:
        """Checks if scaling factor is within the desired range"""
        if factor > 0 and factor > abs(max_scaling_factor):
            return abs(max_scaling_factor)
        if factor < 0 and factor < -abs(max_scaling_factor):
            return -abs(max_scaling_factor)
        return factor
