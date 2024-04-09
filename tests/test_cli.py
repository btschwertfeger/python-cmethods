#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2024 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

"""Module implementing the tests regarding the CLI"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from click.testing import CliRunner
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from cmethods import cli


@pytest.mark.parametrize(
    ("method", "kind", "exclusive"),
    [
        ("linear_scaling", "+", "--group=time.month"),
        ("linear_scaling", "*", "--group=time.month"),
        ("variance_scaling", "+", "--group=time.month"),
        ("delta_method", "+", "--group=time.month"),
        ("delta_method", "*", "--group=time.month"),
        ("quantile_mapping", "+", "--quantiles=100"),
        ("quantile_mapping", "*", "--quantiles=100"),
        ("quantile_delta_mapping", "+", "--quantiles=100"),
        ("quantile_delta_mapping", "*", "--quantiles=100"),
    ],
)
def test_cli_runner(
    method: str,
    kind: str,
    exclusive: str,
    cli_runner: CliRunner,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test checking the command-line interface."""
    logging.root.setLevel(logging.DEBUG)
    with TemporaryDirectory() as tmp_dir:
        output = f"{os.path.join(tmp_dir, method)}.nc"
        cmd: list[str] = [
            f"--obs={os.path.join('examples', 'input_data', 'observations.nc')}",
            f"--simh={os.path.join('examples', 'input_data', 'control.nc')}",
            f"--simp={os.path.join('examples', 'input_data', 'scenario.nc')}",
            f"--method={method}",
            f"--kind={kind}",
            "--variable=tas",
            exclusive,
            f"--output={output}",
        ]
        result = cli_runner.invoke(cli, cmd)
        assert result.exit_code == 0, result.exception
        assert Path(output).is_file()

        for phrase in (
            "Loading data sets ...",
            "Data sets loaded ...",
            f"Applying {method} ...",
            f"Saving result to {output}",
        ):
            assert phrase in caplog.text


def test_cli_runner_missing_variable(
    cli_runner: CliRunner,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Test checking the command-line interface for failure due to missing variable
    in data set.
    """
    logging.root.setLevel(logging.DEBUG)
    with TemporaryDirectory() as tmp_dir:
        output = f"{os.path.join(tmp_dir, 'linear_scaling.nc')}"
        cmd: list[str] = [
            f"--obs={os.path.join('examples', 'input_data', 'observations.nc')}",
            f"--simh={os.path.join('examples', 'input_data', 'control.nc')}",
            f"--simp={os.path.join('examples', 'input_data', 'scenario.nc')}",
            "--method=linear_scaling",
            "--kind=add",
            "--variable=proc",
            "--group=time.month",
            f"--output={output}",
        ]
        result = cli_runner.invoke(cli, cmd)
        assert result.exit_code == 1, result.exception
        assert not Path(output).is_file()

    for phrase in (
        "Loading data sets ...",
        "Variable 'proc' is missing in the observation data set",
    ):
        assert phrase in caplog.text
