#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# Github: https://github.com/btschwertfeger
#
# Note: This is just an example on how to use the python-cmethods module.
#       This is in no way an optimal solution and exists only for demonstration
#       purposes.

import sys

import click
from xarray import open_dataset

from cmethods import CMethods as cm


def save_to_netcdf(ds, **kwargs) -> None:
    """
    Saves the data set to file

    :param ds: The data set to save
    :type ds: xarray.core.dataarray.Dataset
    """
    ds.to_netcdf(
        f"{kwargs['method']}_result_var-{kwargs['variable']}{kwargs['descr1']}_kind-{kwargs['kind']}_group-{kwargs['group']}_{kwargs['start_date']}_{kwargs['end_date']}.nc"
    )


@click.command()
@click.option(
    "--ref",
    "--reference",
    required=True,
    type=str,
    help="The reference data set (control period)",
)
@click.option(
    "--contr",
    "--control",
    required=True,
    type=str,
    help="The modeled data set (control period)",
)
@click.option(
    "--scen",
    "--scenario",
    required=True,
    type=str,
    help="The modeled data set to adjust (scenario period)",
)
@click.option(
    "-m", "--method", required=True, type=str, help="The bias correction method"
)
@click.option(
    "-v", "--variable", required=True, type=str, help="The variable to adjust"
)
@click.option(
    "-k",
    "--kind",
    type=str,
    default="+",
    help="The adjustment variant/kind ('+' or '*', default: '+')",
)
@click.option(
    "-g",
    "--group",
    type=str,
    default="time.month",
    help="The grouping basis (only for scaling-based methods; default: 'time.month')",
)
@click.option(
    "-q",
    "--quantiles",
    type=int,
    default=100,
    help="The number of quantiles to use (only for distribution-based methods; default: 100)",
)
@click.option(
    "-p",
    "--processes",
    type=int,
    default=1,
    help="The number of processes to use (only for 3-dimensional corrections: default: 1)",
)
def main(**kwargs) -> None:
    """
    The Main program that uses the passed arguments to perform the bias correction procedure.
    """
    if kwargs["method"] not in cm.get_available_methods():
        raise ValueError(
            f"Unknown method {kwargs['method']}. Available methods: {cm.get_available_methods()}"
        )

    ds_obs = open_dataset(kwargs["ref"])[kwargs["variable"]]
    ds_simh = open_dataset(kwargs["contr"])[kwargs["variable"]]
    ds_simp = open_dataset(kwargs["scen"])[kwargs["variable"]]

    print("**Data sets loaded**")

    start_date: str = ds_simp["time"][0].dt.strftime("%Y%m%d").values.ravel()[0]
    end_date: str = ds_simp["time"][-1].dt.strftime("%Y%m%d").values.ravel()[0]

    descr1 = ""
    if kwargs["method"] in cm.DISTRIBUTION_METHODS:
        descr1 = f"_quantiles-{kwargs['quantiles']}"
        kwargs["group"] = None
    else:
        kwargs["quantiles"] = None

    kwargs.update({"start_date": start_date, "end_date": end_date, "descr1": descr1})

    print("**Starting correction**")

    n_dims = len(ds_obs.dims)
    if 1 == n_dims:
        result = ds_simp.copy(deep=True)
        result.values = cm.get_function(kwargs["m"])(
            obs=ds_obs,
            simh=ds_simh,
            simp=ds_simp,
            kind=kwargs["kind"],
            n_quantiles=kwargs["quantiles"],
            group=kwargs["group"],
            **kwargs,
        )
        result = result.to_dataset(name=kwargs["variable"])

    elif 3 == n_dims:
        result = cm.adjust_3d(
            method=kwargs["method"],
            obs=ds_obs,
            simh=ds_simh,
            simp=ds_simp,
            n_quantiles=kwargs["quantiles"],
            kind=kwargs["kind"],
            group=kwargs["group"],
            n_jobs=kwargs["processes"],
        )
    else:
        raise ValueError(
            "Cannot correct this data! Idk how to handle more than 3 dimensions!"
        )

    print("**Computation done - saving the result now**")

    save_to_netcdf(result, **kwargs)
    print("**Done**")


if __name__ == "__main__":
    main()
    sys.exit(0)
