#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

r"""
    Module providing the a method named "adjust" to apply different bias
    correction techniques to time-series climate data.

    Some variables used in this package:

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

from cmethods.core import adjust

__author__ = "Benjamin Thomas Schwertfeger"
__copyright__ = __author__
__email__ = "contact@b-schwertfeger.de"
__link__ = "https://github.com/btschwertfeger"
__github__ = "https://github.com/btschwertfeger/python-cmethods"

__all__ = ["adjust"]
