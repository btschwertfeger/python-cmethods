#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
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
