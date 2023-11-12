#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

from __future__ import annotations

from functools import cache

import pytest

from cmethods import CMethods

from .helper import get_datasets


@pytest.fixture()
def datasets() -> dict:
    obsh_add, obsp_add, simh_add, simp_add = get_datasets(kind="+")
    obsh_mult, obsp_mult, simh_mult, simp_mult = get_datasets(kind="*")

    return {
        "+": {
            "obsh": obsh_add["+"],
            "obsp": obsp_add["+"],
            "simh": simh_add["+"],
            "simp": simp_add["+"],
        },
        "*": {
            "obsh": obsh_mult["*"],
            "obsp": obsp_mult["*"],
            "simh": simh_mult["*"],
            "simp": simp_mult["*"],
        },
    }


@pytest.fixture()
def cm() -> CMethods:
    return CMethods()
