#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""Module providing utility functions"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from cmethods.types import NPData, NPData_t, XRData, XRData_t


class UnknownMethodError(Exception):
    """
    Exception raised for errors if unknown method called in CMethods class.
    """

    def __init__(self: UnknownMethodError, method: str, available_methods: list):
        super().__init__(
            f'Unknown method "{method}"! Available methods: {available_methods}'
        )


def check_adjust_called(
    function_name: str,
    adjust_called: Optional[bool] = None,
) -> None:
    """
    Displays a user warning in case a correction function was not called via
    `cmethods.adjust`.

    :param adjust_called: If the function was called via adjust
    :type adjust_called: Optional[bool]
    :param function_name: The function that was called
    :type function_name: str
    """
    if not adjust_called:
        warnings.warn(
            message=f"Do not call {function_name} directly, use `cmethods.adjust` instead!",
            category=UserWarning,
        )


def check_xr_types(obs: XRData, simh: XRData, simp: XRData) -> None:
    """
    Checks if the parameters are in the correct type. **only used internally**
    """
    phrase: str = (
        "must be type xarray.core.dataarray.Dataset or xarray.core.dataarray.DataArray"
    )

    if not isinstance(obs, XRData_t):
        raise TypeError(f"'obs' {phrase}")
    if not isinstance(simh, XRData_t):
        raise TypeError(f"'simh' {phrase}")
    if not isinstance(simp, XRData_t):
        raise TypeError(f"'simp' {phrase}")


def check_np_types(
    obs: NPData,
    simh: NPData,
    simp: NPData,
) -> None:
    """
    Checks if the parameters are in the correct type. **only used internally**
    """
    phrase: str = "must be type list, np.ndarray or np.generic"

    if not isinstance(obs, NPData_t):
        raise TypeError(f"'obs' {phrase}")
    if not isinstance(simh, NPData_t):
        raise TypeError(f"'simh' {phrase}")
    if not isinstance(simp, NPData_t):
        raise TypeError(f"'simp' {phrase}")


def nan_or_equal(value1: float, value2: float) -> bool:
    """
    Returns True if the values are equal or at least one is NaN

    :param value1: First value to check
    :type value1: float
    :param value2: Second value to check
    :type value2: float
    :return: If any value is NaN or values are equal
    :rtype: bool
    """
    return np.isnan(value1) or np.isnan(value2) or value1 == value2


def ensure_devidable(
    numerator: float | np.ndarray,
    denominator: float | np.ndarray,
    max_scaling_factor: float,
) -> np.ndarray:
    """
    Ensures that the arrays can be devided. The numerator will be multiplied by
    the maximum scaling factor of the CMethods class if division by zero.

    :param numerator: Numerator to use
    :type numerator: np.ndarray
    :param denominator: Denominator that can be zero
    :type denominator: np.ndarray
    :return: Zero-ensured devision
    :rtype: np.ndarray | float
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator

    if isinstance(numerator, np.ndarray):
        mask_inf = np.isinf(result)
        result[mask_inf] = numerator[mask_inf] * max_scaling_factor  # type: ignore[index]

        mask_nan = np.isnan(result)
        result[mask_nan] = 0  # type: ignore[index]
    elif np.isinf(result):
        result = numerator * max_scaling_factor
    elif np.isnan(result):
        result = 0.0

    return result


def get_pdf(x: list | np.ndarray, xbins: list | np.ndarray) -> np.ndarray:
    r"""
    Compuites and returns the the probability density function :math:`P(x)`
    of ``x`` based on ``xbins``.

    :param x: The vector to get :math:`P(x)` from
    :type x: list | np.ndarray
    :param xbins: The boundaries/bins of :math:`P(x)`
    :type xbins: list | np.ndarray
    :return: The probability densitiy function of ``x``
    :rtype: np.ndarray

    .. code-block:: python
        :linenos:
        :caption: Compute the probability density function :math:`P(x)`

        >>> from cmethods import CMethods as cm

        >>> x = [1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10]
        >>> xbins = [0, 3, 6, 10]
        >>> print(cm.get_pdf(x=x, xbins=xbins))
        [2, 5, 5]
    """
    pdf, _ = np.histogram(x, xbins)
    return pdf


def get_cdf(x: list | np.ndarray, xbins: list | np.ndarray) -> np.ndarray:
    r"""
    Computes and returns returns the cumulative distribution function :math:`F(x)`
    of ``x`` based on ``xbins``.

    :param x: Vector to get :math:`F(x)` from
    :type x: list | np.ndarray
    :param xbins: The boundaries/bins of :math:`F(x)`
    :type xbins: list | np.ndarray
    :return: The cumulative distribution function of ``x``
    :rtype: np.ndarray


    .. code-block:: python
        :linenos:
        :caption: Compute the cmmulative distribution function :math:`F(x)`

        >>> from cmethods import CMethods as cm

        >>> x = [1, 2, 3, 4, 5, 5, 5, 6, 7, 8, 9, 10]
        >>> xbins = [0, 3, 6, 10]
        >>> print(cm.get_cdf(x=x, xbins=xbins))
        [0, 2, 7, 12]
    """
    pdf, _ = np.histogram(x, xbins)
    return np.insert(np.cumsum(pdf), 0, 0.0)


def get_inverse_of_cdf(
    base_cdf: list | np.ndarray,
    insert_cdf: list | np.ndarray,
    xbins: list | np.ndarray,
) -> np.ndarray:
    r"""
    Returns the inverse cumulative distribution function as:
    :math:`F^{-1}_{x}\left[y\right]` where :math:`x` represents ``base_cdf`` and
    ``insert_cdf`` is represented by :math:`y`.

    :param base_cdf: The basis
    :type base_cdf: list | np.ndarray
    :param insert_cdf: The CDF that gets inserted
    :type insert_cdf: list | np.ndarray
    :param xbins: Probability boundaries
    :type xbins: list | np.ndarray
    :return: The inverse CDF
    :rtype: np.ndarray
    """
    return np.interp(insert_cdf, base_cdf, xbins)


def get_adjusted_scaling_factor(
    factor: int | float, max_scaling_factor: int | float
) -> float:
    r"""
    Returns:
        - :math:`x` if :math:`-|y| \le x \le |y|`,
        - :math:`|y|` if :math:`x > |y|`, or
        - :math:`-|y|` if :math:`x < -|y|`

        where:
            - :math:`x` is ``factor``
            - :math:`y` is ``max_scaling_factor``

    :param factor: The value to check for
    :type factor: int | float
    :param max_scaling_factor: The maximum/minimum allowed value
    :type max_scaling_factor: int | float
    :return: The correct value
    :rtype: float
    """
    if factor > 0 and factor > abs(max_scaling_factor):
        return abs(max_scaling_factor)
    if factor < 0 and factor < -abs(max_scaling_factor):
        return -abs(max_scaling_factor)
    return factor
