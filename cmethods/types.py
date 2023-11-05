#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""Module providing custom types"""

from __future__ import annotations

from typing import TypeVar

from numpy import generic, ndarray
from xarray.core.dataarray import DataArray, Dataset

XRData = TypeVar("XRData", Dataset, DataArray)
NPData = TypeVar("NPData", list, ndarray, generic)
