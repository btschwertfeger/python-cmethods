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
Module providing functions for scaling-based bias adjustments. Functions are not
intended to used directly - but as part of the adjustment procedure triggered by
:func:``cmethods.adjust``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from cmethods.static import ADDITIVE, MAX_SCALING_FACTOR, MULTIPLICATIVE
from cmethods.utils import (
    check_adjust_called,
    check_np_types,
    ensure_dividable,
    get_adjusted_scaling_factor,
)

if TYPE_CHECKING:
    from cmethods.types import NPData


# ? -----========= L I N E A R - S C A L I N G =========------
def linear_scaling(
    obs: NPData,
    simh: NPData,
    simp: NPData,
    kind: str = "+",
    **kwargs: Any,
) -> NPData:
    r"""
    **Do not call this function directly, please use :func:`cmethods.adjust`**

    See https://python-cmethods.readthedocs.io/en/latest/methods.html#linear-scaling
    """
    check_adjust_called(
        function_name="linear_scaling",
        adjust_called=kwargs.get("adjust_called"),
    )
    check_np_types(obs=obs, simh=simh, simp=simp)

    if kind in ADDITIVE:
        return np.array(simp) + (np.nanmean(obs) - np.nanmean(simh))  # Eq. 1
    if kind in MULTIPLICATIVE:
        max_scaling_factor: Final[float] = kwargs.get(
            "max_scaling_factor",
            MAX_SCALING_FACTOR,
        )
        adj_scaling_factor: Final[float] = get_adjusted_scaling_factor(
            ensure_dividable(
                np.nanmean(obs),
                np.nanmean(simh),
                max_scaling_factor,
            ),
            max_scaling_factor,
        )
        return np.array(simp) * adj_scaling_factor  # Eq. 2
    raise NotImplementedError(
        f"{kind=} not available for linear_scaling. Use '+' or '*' instead.",
    )


# ? -----========= V A R I A N C E - S C A L I N G =========------


def variance_scaling(
    obs: NPData,
    simh: NPData,
    simp: NPData,
    kind: str = "+",
    **kwargs: Any,
) -> NPData:
    r"""
    **Do not call this function directly, please use :func:`cmethods.CMethods.adjust`**

    See https://python-cmethods.readthedocs.io/en/latest/methods.html#variance-scaling
    """
    check_adjust_called(
        function_name="variance_scaling",
        adjust_called=kwargs.get("adjust_called"),
    )
    check_np_types(obs=obs, simh=simp, simp=simp)

    if kind in ADDITIVE:
        LS_simh = linear_scaling(obs, simh, simh, kind="+", **kwargs)  # Eq. 1
        LS_simp = linear_scaling(obs, simh, simp, kind="+", **kwargs)  # Eq. 2

        VS_1_simh = LS_simh - np.nanmean(LS_simh)  # Eq. 3
        VS_1_simp = LS_simp - np.nanmean(LS_simp)  # Eq. 4
        max_scaling_factor: Final[float] = kwargs.get(
            "max_scaling_factor",
            MAX_SCALING_FACTOR,
        )
        adj_scaling_factor: Final[float] = get_adjusted_scaling_factor(
            ensure_dividable(
                np.std(np.array(obs)),
                np.std(VS_1_simh),
                max_scaling_factor,
            ),
            max_scaling_factor,
        )

        VS_2_simp = VS_1_simp * adj_scaling_factor  # Eq. 5
        return VS_2_simp + np.nanmean(LS_simp)  # Eq. 6

    raise NotImplementedError(
        f"{kind=} not available for variance_scaling. Use '+' instead.",
    )


# ? -----========= D E L T A - M E T H O D =========------
def delta_method(
    obs: NPData,
    simh: NPData,
    simp: NPData,
    kind: str = "+",
    **kwargs: Any,
) -> NPData:
    r"""
    **Do not call this function directly, please use :func:`cmethods.adjust`**
    See https://python-cmethods.readthedocs.io/en/latest/methods.html#delta-method
    """
    check_adjust_called(
        function_name="delta_method",
        adjust_called=kwargs.get("adjust_called"),
    )
    check_np_types(obs=obs, simh=simh, simp=simp)

    if kind in ADDITIVE:
        return np.array(obs) + (np.nanmean(simp) - np.nanmean(simh))  # Eq. 1
    if kind in MULTIPLICATIVE:
        max_scaling_factor: Final[float] = kwargs.get(
            "max_scaling_factor",
            MAX_SCALING_FACTOR,
        )
        adj_scaling_factor = get_adjusted_scaling_factor(
            ensure_dividable(
                np.nanmean(simp),
                np.nanmean(simh),
                max_scaling_factor,
            ),
            max_scaling_factor,
        )
        return np.array(obs) * adj_scaling_factor  # Eq. 2
    raise NotImplementedError(
        f"{kind=} not available for delta_method. Use '+' or '*' instead.",
    )
