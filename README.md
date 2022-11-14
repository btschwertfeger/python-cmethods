# Bias adjustment/correction procedures for climatic reasearch

<div align="center">

[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/btschwertfeger/Bias-Adjustment-Python)
[![Generic badge](https://img.shields.io/badge/python-3.7+-green.svg)](https://shields.io/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Downloads](https://pepy.tech/badge/python-cmethods)](https://pepy.tech/project/python-cmethods)

</div>

This Python module contains a collection of different scale- and distribution-based bias adjustment techniques for climatic research (see `examples.ipynb` for help).

Since the Python programming language is very slow and bias adjustments are complex statistical transformations, it is recommended to use the C++ implementation on large data sets. This can be found [here](https://github.com/btschwertfeger/Bias-Adjustment-Cpp).

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
  src="images/biasCdiagram.png?raw=true"
  alt="Schematic representation of a bias adjustment procedure"
  style="background-color: white; border-radius: 7px">
  <figcaption>Figure 1: Schematic representation of a bias adjustment procedure</figcaption>
</figure>

In this way, for example, modeled data, which on average represent values that are too cold, can be adjusted by applying an adjustment procedure. The following figure shows the observed, the modeled and the adjusted values. It is directly visible that the delta adjusted time series ($T^{*DM}_{sim,p}$) are much more similar to the observed data ($T_{obs,p}$) than the raw modeled data ($T_{sim,p}$).

<figure>
  <img
  src="images/dm-doy-plot.png?raw=true"
  alt="Temperature per day of year in modeled, observed and bias-adjusted climate data"
  style="background-color: white; border-radius: 7px">
  <figcaption>Figure 2: Temperature per day of year in observed, modeled and bias-adjusted climate data</figcaption>
</figure>

---

<a name="methods"></a>

## 2. Available methods:

All methods except the `adjust_3d` function requires the application on one time series.

| Function name            | Description                                                                                                |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `linear_scaling`         | Linear Scaling (additive and multiplicative)                                                               |
| `variance_scaling`       | Variance Scaling (additive)                                                                                |
| `delta_method`           | Delta (Change) Method (additive and multiplicative)                                                        |
| `quantile_mapping`       | Quantile Mapping (additive) and Detrended Quantile Mapping (additive and multiplicative)                   |
| `quantile_delta_mapping` | Quantile Delta Mapping (additive and multiplicative)                                                       |
| `adjust_3d`              | requires a method name and the respective parameters to adjust all time series of a 3-dimensional data set |

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
from cmethods.CMethods import CMethods
cm = CMethods()

obsh = xr.open_dataset('input_data/observations.nc')
simh = xr.open_dataset('input_data/control.nc')
simp = xr.open_dataset('input_data/scenario.nc')

ls_result = cm.linear_scaling(
    obs = obsh['tas'][:,0,0],
    simh = simh['tas'][:,0,0],
    simp = simp['tas'][:,0,0],
    kind = '+' # *
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
# ratio based variables like precipitation)
```

Notes:

- When using the `adjust_3d` method you have to specify the method by name.
- For the multiplicative linear scaling and the delta method as well as the variance scaling method a maximum scaling factor of 10 is defined. This can be changed by the parameter `max_scaling_factor`.

## Examples (see repository on [GitHub](https://github.com/btschwertfeger/Bias-Adjustment-Python))

Notebook with different methods and plots: `/examples/examples.ipynb`

Example script for adjusting climate data: `/examples/do_bias_correction.py`

```bash
python3 do_bias_correction.py         \
    --obs input_data/observations.nc  \
    --contr input_data/control.nc     \
    --scen input_data/scenario.nc     \
    --method linear_scaling           \
    --variable tas                    \
    --unit '°C'                       \
    --group 'time.month'              \
    --kind +
```

- Linear and variance, as well as delta change method require `--group time.month` as argument.
- Adjustment methods that apply changes in distributional biasses (QM, QDM, DQM, ...) require the `--nquantiles` argument set to some integer.
- Data sets must have the same spatial resolutions.

---

<a name="notes"></a>

## 5. Notes

- Computation in Python takes some time, so this is only for demonstration. When adjusting large datasets, its best to use the C++ implementation mentioned above.
- Formulas and references can be found in the implementations of the corresponding functions.

### Space for improvements:

Since the scaling methods implemented so far scale by default over the mean values of the respective months, unrealistic long-term mean values may occur at the month transitions. This can be prevented either by selecting `group='time.dayofyear'`. Alternatively, it is possible not to scale using long-term mean values, but using a 31-day interval, which takes the 31 surrounding values over all years as the basis for calculating the mean values. This is not yet implemented in this module, but is available in the C++ implementation [here](https://github.com/btschwertfeger/Bias-Adjustment-Cpp).

---

<a name="references"></a>

## 6. References

- Schwertfeger, Benjamin Thomas (2022) The influence of bias corrections on variability, distribution, and correlation of temperatures in comparison to observed and modeled climate data in Europe (https://epic.awi.de/id/eprint/56689/)
- Linear Scaling and Variance Scaling based on: Teutschbein, Claudia and Seibert, Jan (2012) Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods (https://doi.org/10.1016/j.jhydrol.2012.05.052)
- Delta Method based on: Beyer, R. and Krapp, M. and Manica, A.: An empirical evaluation of bias correction methods for palaeoclimate simulations (https://doi.org/10.5194/cp-16-1493-2020)
- Quantile and Detrended Quantile Mapping based on: Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes? (https://doi.org/10.1175/JCLI-D-14-00754.1)
- Quantile Delta Mapping based on: Tong, Y., Gao, X., Han, Z. et al. Bias correction of temperature and precipitation over China for RCM simulations using the QM and QDM methods. Clim Dyn 57, 1425–1443 (2021). (https://doi.org/10.1007/s00382-020-05447-4)

---
