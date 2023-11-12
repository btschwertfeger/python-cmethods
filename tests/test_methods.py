#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023 Benjamin Thomas Schwertfeger
# GitHub: https://github.com/btschwertfeger
#

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cmethods.types import NPData_t, XRData_t

if TYPE_CHECKING:
    from cmethods import CMethods

from .helper import is_1d_rmse_better, is_3d_rmse_better

GROUP: str = "time.month"
N_QUANTILES: int = 100


def test_linear_scaling_add_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "linear_scaling"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_linear_scaling_add_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "linear_scaling"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    # not grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_linear_scaling_mult_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    method: str = "linear_scaling"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_linear_scaling_mult_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    method: str = "linear_scaling"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    # not grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_variance_scaling_add_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "variance_scaling"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )

    # not group
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_variance_scaling_add_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "variance_scaling"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    # not grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_delta_method_add_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "delta_method"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_delta_method_add_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "delta_method"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    # not grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_delta_method_mult_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    method: str = "delta_method"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )
    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_delta_method_mult_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    method: str = "delta_method"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    # not grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)

    # grouped
    result: XRData_t = cm.adjust(
        method=method, obs=obsh, simh=simh, simp=simp, kind=kind, group=GROUP
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result[kind], obsp=obsp, simp=simp)


def test_quantile_mapping_add_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "quantile_mapping"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_quantile_mapping_add_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "quantile_mapping"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_quantile_mapping_mult_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    method: str = "quantile_mapping"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_quantile_mapping_mult_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    method: str = "quantile_mapping"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)


@pytest.mark.wip()
def test_detrended_quantile_mapping_add_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.detrended_quantile_mapping(
        obs=obsh, simh=simh, simp=simp, kind=kind, n_quantiles=N_QUANTILES
    )
    assert isinstance(result, NPData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)


@pytest.mark.wip()
def test_detrended_quantile_mapping_mult_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    # not group
    result: XRData_t = cm.detrended_quantile_mapping(
        obs=obsh, simh=simh, simp=simp, kind=kind, n_quantiles=N_QUANTILES
    )
    assert isinstance(result, NPData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_quantile_delta_mapping_add_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "quantile_delta_mapping"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_quantile_delta_mapping_add_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "+"
    method: str = "quantile_delta_mapping"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_quantile_delta_mapping_mult_1d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    method: str = "quantile_delta_mapping"
    obsh: XRData_t = datasets[kind]["obsh"][:, 0, 0]
    obsp: XRData_t = datasets[kind]["obsp"][:, 0, 0]
    simh: XRData_t = datasets[kind]["simh"][:, 0, 0]
    simp: XRData_t = datasets[kind]["simp"][:, 0, 0]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_1d_rmse_better(result=result, obsp=obsp, simp=simp)


def test_quantile_delta_mapping_mult_3d(cm: CMethods, datasets: dict) -> None:
    kind: str = "*"
    method: str = "quantile_delta_mapping"
    obsh: XRData_t = datasets[kind]["obsh"]
    obsp: XRData_t = datasets[kind]["obsp"]
    simh: XRData_t = datasets[kind]["simh"]
    simp: XRData_t = datasets[kind]["simp"]

    result: XRData_t = cm.adjust(
        method=method,
        obs=obsh,
        simh=simh,
        simp=simp,
        kind=kind,
        n_quantiles=N_QUANTILES,
    )

    assert isinstance(result, XRData_t)
    assert is_3d_rmse_better(result=result, obsp=obsp, simp=simp)
