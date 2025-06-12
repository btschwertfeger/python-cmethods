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

"""Module providing custom types"""

from __future__ import annotations

from typing import TypeVar

from numpy import generic, ndarray
from xarray.core.dataarray import DataArray, Dataset

XRData_t = (Dataset, DataArray)
NPData_t = (list, ndarray, generic)
XRData = TypeVar("XRData", Dataset, DataArray)
NPData = TypeVar("NPData", list, ndarray, generic)
