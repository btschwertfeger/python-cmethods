# -*- mode: python; coding: utf-8 -*-
#
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
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

"""Module providing static information for the python-cmethods package"""

from __future__ import annotations

from typing import List

SCALING_METHODS: List[str] = [
    "linear_scaling",
    "variance_scaling",
    "delta_method",
]
DISTRIBUTION_METHODS: List[str] = [
    "quantile_mapping",
    "detrended_quantile_mapping",
    "quantile_delta_mapping",
]

CUSTOM_METHODS: List[str] = SCALING_METHODS + DISTRIBUTION_METHODS
METHODS: List[str] = CUSTOM_METHODS

ADDITIVE: List[str] = ["+", "add"]
MULTIPLICATIVE: List[str] = ["*", "mult"]
MAX_SCALING_FACTOR: int = 10

__all__ = [
    "ADDITIVE",
    "CUSTOM_METHODS",
    "DISTRIBUTION_METHODS",
    "MAX_SCALING_FACTOR",
    "METHODS",
    "MULTIPLICATIVE",
    "SCALING_METHODS",
]
