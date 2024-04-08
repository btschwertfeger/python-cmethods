#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#
# pylint: disable=consider-using-f-string,logging-not-lazy

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

from __future__ import annotations

import logging

import cloup
import xarray as xr
from cloup import (
    HelpFormatter,
    HelpTheme,
    Path,
    Style,
    command,
    option,
    option_group,
    version_option,
)
from cloup.constraints import Equal, If, require_all

from cmethods.core import adjust

__all__ = ["adjust"]

formatter_settings = HelpFormatter.settings(
    theme=HelpTheme(
        invoked_command=Style(fg="bright_yellow"),
        heading=Style(fg="bright_white", bold=True),
        constraint=Style(fg="magenta"),
        col1=Style(fg="bright_yellow"),
    ),
)


@command(
    context_settings={
        "auto_envvar_prefix": "CMETHODS",
        "help_option_names": ["-h", "--help"],
    },
    formatter_settings=formatter_settings,
)
@version_option(message="%version%")
@option(
    "--obs",
    "--observations",
    required=True,
    type=Path(exists=True),
    help="Reference data set (control period)",
)
@option(
    "--simh",
    "--simulated-historical",
    required=True,
    type=Path(exists=True),
    help="Modeled data set (control period)",
)
@option(
    "--simp",
    "--simulated-scenario",
    required=True,
    type=Path(exists=True),
    help="Modeled data set (scenario period)",
)
@option(
    "--method",
    required=True,
    type=cloup.Choice(
        [
            "linear_scaling",
            "variance_scaling",
            "delta_method",
            "quantile_mapping",
            "quantile_delta_mapping",
        ],
        case_sensitive=False,
    ),
    help="Bias adjustment method to apply",
)
@option(
    "--kind",
    required=True,
    type=cloup.Choice(["add", "mult"]),
    help="Kind of adjustment",
)
@option(
    "--variable",
    required=True,
    type=str,
    help="Variable of interest",
)
@option(
    "-o",
    "--output",
    required=True,
    type=str,
    callback=lambda _, _, value: (  # noqa: E999
        value if value.endswith(".nc") else f"{value}.nc"
    ),
    help="Output file name",
)
@option_group(
    "Scaling-Based Adjustment Options",
    option(
        "--group",
        type=str,
        default="time.month",
        help="Temporal grouping to apply",
    ),
    constraint=If(
        Equal("method", "linear_scaling")
        & Equal("method", "variance_scaling")
        & Equal("method", "delta_method"),
        then=require_all,
    ),
)
@option_group(
    "Distribution-Based Adjustment Options",
    option(
        "--quantiles",
        type=int,
        help="Quantiles to respect",
    ),
    constraint=If(
        Equal("method", "quantile_mapping") & Equal("method", "quantile_delta_mapping"),
        then=require_all,
    ),
)
def cli(**kwargs) -> None:
    """Command line tool to apply bias adjustment procedures to climate data."""

    logging.basicConfig(
        format="%(asctime)s %(levelname)8s | %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )

    logging.info("Loading data sets ...")
    kwargs["obs"] = xr.open_dataset(kwargs["obs"])[kwargs["variable"]]
    kwargs["simh"] = xr.open_dataset(kwargs["simh"])[kwargs["variable"]]
    kwargs["simp"] = xr.open_dataset(kwargs["simp"])[kwargs["variable"]]
    logging.info("Data sets loaded ...")

    logging.info("Applying %s ..." % kwargs["method"])
    result = adjust(**kwargs)

    logging.info("Saving result to %s ..." % kwargs["output"])

    result.to_netcdf(kwargs["output"])
