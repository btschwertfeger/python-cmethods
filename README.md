# python-cmethods

<div align="center">

[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/btschwertfeger/Bias-Adjustment-Python)
[![Generic badge](https://img.shields.io/badge/python-3.8_|_3.9_|_3.10_|_3.11-yellow.svg)](https://shields.io/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Downloads](https://pepy.tech/badge/python-cmethods)](https://pepy.tech/project/python-cmethods)

![CodeQL](https://github.com/btschwertfeger/Bias-Adjustment-Python/actions/workflows/codeql.yml/badge.svg)
[![CI/CD](https://github.com/btschwertfeger/python-cmethods/actions/workflows/cicd.yml/badge.svg?branch=master)](https://github.com/btschwertfeger/python-cmethods/actions/workflows/cicd.yml)
[![codecov](https://codecov.io/github/btschwertfeger/python-cmethods/branch/master/graph/badge.svg?token=OSO4PAABPD)](https://codecov.io/github/btschwertfeger/python-cmethods)

![release](https://shields.io/github/release-date/btschwertfeger/python-cmethods)
![release](https://shields.io/github/v/release/btschwertfeger/python-cmethods?display_name=tag)
[![DOI](https://zenodo.org/badge/496160109.svg)](https://zenodo.org/badge/latestdoi/496160109)
[![Documentation Status](https://readthedocs.org/projects/python-cmethods/badge/?version=stable)](https://python-cmethods.readthedocs.io/en/latest/?badge=stable)

</div>

This Python module serves as a collection of different scale- and distribution-based bias correction techniques for climatic research

The documentation is available at: [https://python-cmethods.readthedocs.io/en/stable/](https://python-cmethods.readthedocs.io/en/stable/)

> ⚠️ For the application of bias corrections on _lage data sets_ it is recomanded to use the command-line tool [BiasAdjustCXX](https://github.com/btschwertfeger/BiasAdjustCXX) since bias corrections are complex statistical transformation which are very slow in Python compared to the C++ implementation.

---

## Table of Contents

1. [ About ](#about)
2. [ Available Methods ](#methods)
3. [ Installation ](#installation)
4. [ Usage and Examples ](#examples)
5. [ Notes ](#notes)
6. [ References ](#references)

---

<a name="about"></a>

## 1. About

These programs and data structures are designed to help minimize discrepancies between modeled and observed climate data. Data from past periods are used to adjust variables from current and future time series so that their distributional properties approximate possible actual values.

<figure>
  <img
  src="docs/_static/images/biasCdiagram.png?raw=true"
  alt="Schematic representation of a bias adjustment procedure"
  style="background-color: white; border-radius: 7px">
  <figcaption>Figure 1: Schematic representation of a bias adjustment procedure</figcaption>
</figure>

In this way, for example, modeled data, which on average represent values that are too cold, can be adjusted by applying an adjustment procedure. The following figure shows the observed, the modeled, and the adjusted values. It is directly visible that the delta adjusted time series ($T^{\*DM}_{sim,p}$) are much more similar to the observed data ($T_{obs,p}$) than the raw modeled data ($T_{sim,p}$).

<figure>
  <img
  src="docs/_static/images/dm-doy-plot.png?raw=true"
  alt="Temperature per day of year in modeled, observed and bias-adjusted climate data"
  style="background-color: white; border-radius: 7px">
  <figcaption>Figure 2: Temperature per day of year in observed, modeled, and bias-adjusted climate data</figcaption>
</figure>

---

<a name="methods"></a>

## 2. Available methods

All methods except the `adjust_3d` function requires that the input data sets only contain one dimension.

| Function name            | Description                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `linear_scaling`         | Linear Scaling (additive and multiplicative)                                                                                                                 |
| `variance_scaling`       | Variance Scaling (additive)                                                                                                                                  |
| `delta_method`           | Delta (Change) Method (additive and multiplicative)                                                                                                          |
| `quantile_mapping`       | Quantile Mapping (additive and multiplicative) and Detrended Quantile Mapping (additive and multiplicative; to use DQM, set parameter `detrended` to `True`) |
| `quantile_delta_mapping` | Quantile Delta Mapping (additive and multiplicative)                                                                                                         |
| `adjust_3d`              | requires a method name and the respective parameters to adjust all time series of a 3-dimensional data set                                                   |

Except for the variance scaling, all methods can be applied on stochastic and non-stochastic
climate variables. Variance scaling can only be applied on non-stochastic climate variables.

- Stochastic climate variables are those that are subject to random fluctuations
  and are not predictable. They have no predictable trend or pattern. Examples of
  stochastic climate variables include precipitation, air temperature, and humidity.

- Non-stochastic climate variables, on the other hand, have clear trend and pattern histories
  and can be readily predicted. They are often referred to as climate elements and include
  variables such as water temperature and air pressure.

---

<a name="installation"></a>

## 3. Installation

```bash
python3 -m pip install python-cmethods
```

---

<a name="examples"></a>

## 4. Usage and Examples

```python
import xarray as xr
from cmethods import CMethods as cm

obsh = xr.open_dataset('input_data/observations.nc')
simh = xr.open_dataset('input_data/control.nc')
simp = xr.open_dataset('input_data/scenario.nc')

ls_result = cm.linear_scaling(
    obs = obsh['tas'][:,0,0],
    simh = simh['tas'][:,0,0],
    simp = simp['tas'][:,0,0],
    kind = '+'
)

qdm_result = cm.adjust_3d( # 3d = 2 spatial and 1 time dimension
    method = 'quantile_delta_mapping',
    obs = obsh['tas'],
    simh = simh['tas'],
    simp = simp['tas'],
    n_quaniles = 1000,
    kind = '+'
)
# to calculate the relative rather than the absolute change,
# '*' can be used instead of '+' (this is prefered when adjusting
# stochastic variables like precipitation)
```

Notes:

- When using the `adjust_3d` method you have to specify the method by name.
- For the multiplicative linear scaling and the delta method as well as the variance scaling method a maximum scaling factor of 10 is defined. This can be changed by the parameter `max_scaling_factor`.

## Examples (see repository on [GitHub](https://github.com/btschwertfeger/python-cmethods))

Notebook with different methods and plots: `/examples/examples.ipynb`

There is also an exmple script (`/examples/biasadjust.py`) that can be used to apply the available bias correction methods
on 1- and 3-dimensional data sets (see `/examples/input_data/*.nc`).

Help:

```bash
╰─ python3 biasadjust.py --help
```

(1.) Example - Quantile Mapping bias correction on the provided example data:

```bash
╰─ python3 biasadjust.py              \
    --ref input_data/observations.nc  \
    --contr input_data/control.nc     \
    --scen input_data/scenario.nc     \
    --kind "+"                        \
    --variable "tas"                  \
    --method quantile_mapping
```

(2.) Example - Linear Scaling bias correction on the provided example data:

```bash
╰─ python3 biasadjust.py              \
    --ref input_data/observations.nc  \
    --contr input_data/control.nc     \
    --scen input_data/scenario.nc     \
    --kind "+"                        \
    --variable "tas"                  \
    --group "time.month"              \
    --method linear_scaling
```

Notes:

- Data sets must have the same spatial resolutions.
- This script is far away from perfect - so please look at it, as a starting point. (:

---

<a name="notes"></a>

## 5. Notes

- Computation in Python takes some time, so this is only for demonstration. When adjusting large datasets, its best to use the command-line tool [BiasAdjustCXX](https://github.com/btschwertfeger/BiasAdjustCXX).
- Formulas and references can be found in the implementations of the corresponding functions, on the bottom of the README.md and in the [documentation](https://python-kraken-sdk.readthedocs.io/en/stable/).

### Space for improvements:

- Since the scaling methods implemented so far scale by default over the mean values of the respective months, unrealistic long-term mean values may occur at the month transitions. This can be prevented either by selecting `group='time.dayofyear'`. Alternatively, it is possible not to scale using long-term mean values, but using a 31-day interval, which takes the 31 surrounding values over all years as the basis for calculating the mean values. This is not yet implemented, because even the computation for this takes so much time, that it is not worth implementing it in python - but this is available in [BiasAdjustCXX](https://github.com/btschwertfeger/BiasAdjustCXX).

---

<a name="references"></a>

## 6. References

- Schwertfeger, Benjamin Thomas and Lohmann, Gerrit and Lipskoch, Henrik (2023) _"Introduction of the BiasAdjustCXX command-line tool for the application of fast and efficient bias corrections in climatic research"_, SoftwareX, Volume 22, 101379, ISSN 2352-7110, (https://doi.org/10.1016/j.softx.2023.101379)
- Schwertfeger, Benjamin Thomas (2022) _"The influence of bias corrections on variability, distribution, and correlation of temperatures in comparison to observed and modeled climate data in Europe"_ (https://epic.awi.de/id/eprint/56689/)
- Linear Scaling and Variance Scaling based on: Teutschbein, Claudia and Seibert, Jan (2012) _"Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods"_ (https://doi.org/10.1016/j.jhydrol.2012.05.052)
- Delta Method based on: Beyer, R. and Krapp, M. and Manica, A.: _"An empirical evaluation of bias correction methods for palaeoclimate simulations"_ (https://doi.org/10.5194/cp-16-1493-2020)
- Quantile and Detrended Quantile Mapping based on: Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock _"Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes?"_ (https://doi.org/10.1175/JCLI-D-14-00754.1)
- Quantile Delta Mapping based on: Tong, Y., Gao, X., Han, Z. et al. _"Bias correction of temperature and precipitation over China for RCM simulations using the QM and QDM methods"_. Clim Dyn 57, 1425–1443 (2021). (https://doi.org/10.1007/s00382-020-05447-4)

---
