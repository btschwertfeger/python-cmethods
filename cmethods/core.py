# -*- mode: python; coding: utf-8 -*-
#
# Copyright (C) 2024 Benjamin Thomas Schwertfeger
# https://github.com/btschwertfeger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see
# https://www.gnu.org/licenses/gpl-3.0.html.
#

"""
Module providing the main function that is used to apply the implemented bias
correction techniques.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Dict, Optional

import xarray as xr

from cmethods.distribution import quantile_delta_mapping as __quantile_delta_mapping
from cmethods.distribution import quantile_mapping as __quantile_mapping
from cmethods.scaling import delta_method as __delta_method
from cmethods.scaling import linear_scaling as __linear_scaling
from cmethods.scaling import variance_scaling as __variance_scaling
from cmethods.static import SCALING_METHODS
from cmethods.utils import UnknownMethodError, ensure_xr_dataarray

if TYPE_CHECKING:
    from cmethods.types import XRData

__METHODS_FUNC__: Dict[str, Callable] = {
    "linear_scaling": __linear_scaling,
    "variance_scaling": __variance_scaling,
    "delta_method": __delta_method,
    "quantile_mapping": __quantile_mapping,
    "quantile_delta_mapping": __quantile_delta_mapping,
}


def _add_cmethods_metadata(
    result: xr.Dataset | xr.DataArray,
    method: str,
    **kwargs,
) -> xr.Dataset | xr.DataArray:
    """
    Add metadata to the result indicating it was processed by python-cmethods.

    :param result: The bias-corrected dataset or dataarray
    :param method: The method used for bias correction
    :param kwargs: Additional method parameters
    :return: Result with added metadata
    """
    try:
        from importlib.metadata import version  # noqa: PLC0415

        pkg_version = version("python-cmethods")
    except Exception:  # noqa: BLE001
        pkg_version = "unknown"

    attrs_to_add = {
        "cmethods_version": pkg_version,
        "cmethods_method": method,
        "cmethods_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "cmethods_source": "https://github.com/btschwertfeger/python-cmethods",
    }

    if kind := kwargs.get("kind"):
        attrs_to_add["cmethods_kind"] = kind
    if n_quantiles := kwargs.get("n_quantiles"):
        attrs_to_add["cmethods_n_quantiles"] = str(n_quantiles)
    if group := kwargs.get("group"):
        attrs_to_add["cmethods_group"] = str(group)

    if isinstance(result, (xr.Dataset, xr.DataArray)):
        result.attrs.update(attrs_to_add)

    return result


def apply_ufunc(
    method: str,
    obs: xr.xarray.core.dataarray.DataArray,
    simh: xr.xarray.core.dataarray.DataArray,
    simp: xr.xarray.core.dataarray.DataArray,
    **kwargs: dict,
) -> xr.xarray.core.dataarray.DataArray:
    """
    Internal function used to apply the bias correction technique to the
    passed input data.
    """
    ensure_xr_dataarray(obs=obs, simh=simh, simp=simp)
    if method not in __METHODS_FUNC__:
        raise UnknownMethodError(method, __METHODS_FUNC__.keys())

    if kwargs.get("input_core_dims"):
        if not isinstance(kwargs["input_core_dims"], dict):
            raise TypeError("input_core_dims must be an object of type 'dict'")
        if not len(kwargs["input_core_dims"]) == 3 or any(
            not isinstance(value, str) for value in kwargs["input_core_dims"].values()
        ):
            raise ValueError(
                'input_core_dims must have three key-value pairs like: {"obs": "time", "simh": "time", "simp": "time"}',
            )

        input_core_dims = kwargs.pop("input_core_dims")
    else:
        input_core_dims = {"obs": "time", "simh": "time", "simp": "time"}

    result: XRData = xr.apply_ufunc(
        __METHODS_FUNC__[method],
        obs,
        simh,
        # Need to spoof a fake time axis since 'time' coord on full dataset is
        # different than 'time' coord on training dataset.
        simp.rename({input_core_dims["simp"]: "__t_simp__"}),
        dask="parallelized",
        vectorize=True,
        # This will vectorize over the time dimension, so will submit each grid
        # cell independently
        input_core_dims=[
            [input_core_dims["obs"]],
            [input_core_dims["simh"]],
            ["__t_simp__"],
        ],
        # Need to denote that the final output dataset will be labeled with the
        # spoofed time coordinate
        output_core_dims=[["__t_simp__"]],
        kwargs=dict(kwargs),
    )

    # Rename to proper coordinate name.
    result = result.rename({"__t_simp__": input_core_dims["simp"]})

    # ufunc will put the core dimension to the end (time), so want to preserve
    # original order where time is commonly first.
    return result.transpose(*obs.rename({input_core_dims["obs"]: input_core_dims["simp"]}).dims)


def adjust(
    method: str,
    obs: xr.xarray.core.dataarray.DataArray,
    simh: xr.xarray.core.dataarray.DataArray,
    simp: xr.xarray.core.dataarray.DataArray,
    **kwargs,
) -> xr.xarray.core.dataarray.DataArray | xr.xarray.core.dataarray.Dataset:
    """
    Function to apply a bias correction technique on single and multidimensional
    data sets. For more information please refer to the method specific
    requirements and execution examples.

    See https://python-cmethods.readthedocs.io/en/latest/methods.html


    The time dimension of ``obs``, ``simh`` and ``simp`` must be named ``time``.

    If the sizes of time dimensions of the input data sets differ, you have to
    pass the hidden ``input_core_dims`` parameter, see
    https://python-cmethods.readthedocs.io/en/latest/getting_started.html#advanced-usage
    for more information.

    :param method: Technique to apply
    :type method: str
    :param obs: The reference/observational data set
    :type obs: xr.xarray.core.dataarray.DataArray
    :param simh: The modeled data of the control period
    :type simh: xr.xarray.core.dataarray.DataArray
    :param simp: The modeled data of the period to adjust
    :type simp: xr.xarray.core.dataarray.DataArray
    :param kwargs: Any other method-specific parameter (like
        ``n_quantiles`` and ``kind``)
    :type kwargs: dict
    :return: The bias corrected/adjusted data set
    :rtype: xr.xarray.core.dataarray.DataArray | xr.xarray.core.dataarray.Dataset
    """
    metadata_kwargs = {k: v for k, v in kwargs.items() if k in {"kind", "n_quantiles", "group"}}

    kwargs["adjust_called"] = True
    ensure_xr_dataarray(obs=obs, simh=simh, simp=simp)

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
    if kwargs.get("group") is None:
        result = apply_ufunc(method, obs, simh, simp, **kwargs).to_dataset()
        return _add_cmethods_metadata(result, method, **metadata_kwargs)

    if method not in SCALING_METHODS:
        raise ValueError(
            "Can't use group for distribution based methods.",  # except for DQM
        )

    # Grouped correction | scaling-based technique
    group: str | dict[str, str] = kwargs["group"]
    if isinstance(group, str):
        # only for same sized time dimensions
        obs_group = group
        simh_group = group
        simp_group = group
    elif isinstance(group, dict):
        if any(key not in {"obs", "simh", "simp"} for key in group):
            raise ValueError(
                "group must either be a string like 'time' or a dict like "
                '{"obs": "time.month", "simh": "t_simh.month", "simp": "time.month"}',
            )
        # for different sized time dimensions
        obs_group = group["obs"]
        simh_group = group["simh"]
        simp_group = group["simp"]
    else:
        raise ValueError("'group' must be a string or a dict!")

    del kwargs["group"]

    result: Optional[XRData] = None
    for (_, obs_gds), (_, simh_gds), (_, simp_gds) in zip(
        obs.groupby(obs_group),
        simh.groupby(simh_group),
        simp.groupby(simp_group),
    ):
        monthly_result = apply_ufunc(
            method,
            obs_gds,
            simh_gds,
            simp_gds,
            **kwargs,
        )

        result = monthly_result if result is None else xr.merge([result, monthly_result], compat="no_conflicts", join="outer")

    return _add_cmethods_metadata(result, method, **metadata_kwargs)


__all__ = ["adjust"]
