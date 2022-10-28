# Bias-Adjustment-Python

<div style="text-align: center">

[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/btschwertfeger/Bias-Adjustment-Python)
[![Generic badge](https://img.shields.io/badge/python-3.7+-green.svg)](https://shields.io/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Downloads](https://pepy.tech/badge/python-cmethods)](https://pepy.tech/project/python-cmethods)

</div>

Collection of different scale- and distribution-based bias adjustment techniques for climatic research. (see `examples.ipynb` for help)

Bias adjustment procedures in Python are very slow, so they should not be used on large data sets.
A C++ implementation that works way faster can be found [here](https://github.com/btschwertfeger/Bias-Adjustment-Cpp).

## About

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
  <figcaption>Figure 2: Temperature per day of year in modeled, observed and bias-adjusted climate data</figcaption>
</figure>

---

## Available methods:

- Linear Scaling (additive and multiplicative)
- Variance Scaling (additive)
- Delta (Change) Method (additive and multiplicative)
- Quantile Mapping (additive)
- Detrended Quantile Mapping (additive and multiplicative)
- Quantile Delta Mapping (additive and multuplicative)

---

## Usage

### Installation

```bash
python3 -m pip install python-cmethods
```

### Import and application

```python
import xarray as xr
from cmethods.CMethods import CMethods
cm = CMethods()

obsh = xr.open_dataset('input_data/obs.nc')
simh = xr.open_dataset('input_data/contr.nc')
simp = xr.open_dataset('input_data/scen.nc')

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
    kind = '+' # *
)
# * to calculate the relative rather than the absolute change,
# '*' can be used instead of '+' (this is prefered when adjusting
# ratio based variables like precipitation)
```

---

## Examples (see repository on [GitHub](https://github.com/btschwertfeger/Bias-Adjustment-Python))

`/examples/examples.ipynb`: Notebook containing different methods and plots

`/examples/do_bias_correction.py`: Example script for adjusting climate data

```bash
python3 do_bias_correction.py   \
    --obs input_data/obs.nc     \
    --contr input_data/contr.nc \
    --scen input_data/scen.nc   \
    --method linear_scaling     \
    --variable tas              \
    --unit '°C'                 \
    --group time.month          \
    --kind +
```

- Linear and variance, as well as delta change method require `--group time.month` as argument.
- Adjustment methods that apply changes in distributional biasses (QM, QDM, DQM; EQM, ...) need the `--nquantiles` argument set to some integer.
- Data sets should have the same spatial resolutions.

---

## Notes:

- Computation in Python takes some time, so this is only for demonstration. When adjusting large datasets, its best to the C++ implementation mentioned above.
- Formulas and references can be found in the implementations of the corresponding functions.

## References

- Schwertfeger, Benjamin Thomas (2022) The influence of bias corrections on variability, distribution, and correlation of temperatures in comparison to observed and modeled climate data in Europe (https://epic.awi.de/id/eprint/56689/)
- Linear Scaling and Variance Scaling based on: Teutschbein, Claudia and Seibert, Jan (2012) Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods (https://doi.org/10.1016/j.jhydrol.2012.05.052)
- Delta Method based on: Beyer, R. and Krapp, M. and Manica, A.: An empirical evaluation of bias correction methods for palaeoclimate simulations (https://doi.org/10.5194/cp-16-1493-2020)
- Quantile and Detrended Quantile Mapping based on: Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes? (https://doi.org/10.1175/JCLI-D-14-00754.1)
- Quantile Delta Mapping based on: Tong, Y., Gao, X., Han, Z. et al. Bias correction of temperature and precipitation over China for RCM simulations using the QM and QDM methods. Clim Dyn 57, 1425–1443 (2021). (https://doi.org/10.1007/s00382-020-05447-4)
