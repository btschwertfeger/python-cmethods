# python-cmethods

<div align="center">

[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/btschwertfeger/python-cmethods)
[![Generic badge](https://img.shields.io/badge/python-3.8_|_3.9_|_3.10_|_3.11|_3.12-blue.svg)](https://shields.io/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-orange.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Downloads](https://pepy.tech/badge/python-cmethods)](https://pepy.tech/project/python-cmethods)

[![CI/CD](https://github.com/btschwertfeger/python-cmethods/actions/workflows/cicd.yaml/badge.svg?branch=master)](https://github.com/btschwertfeger/python-cmethods/actions/workflows/cicd.yaml)
[![codecov](https://codecov.io/github/btschwertfeger/python-cmethods/branch/master/graph/badge.svg?token=OSO4PAABPD)](https://codecov.io/github/btschwertfeger/python-cmethods)

[![OpenSSF ScoreCard](https://img.shields.io/ossf-scorecard/github.com/btschwertfeger/python-cmethods?label=openssf%20scorecard&style=flat)](https://securityscorecards.dev/viewer/?uri=github.com/btschwertfeger/python-cmethods)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8666/badge)](https://www.bestpractices.dev/projects/8666)

![release](https://shields.io/github/release-date/btschwertfeger/python-cmethods)
![release](https://shields.io/github/v/release/btschwertfeger/python-cmethods?display_name=tag)
[![DOI](https://zenodo.org/badge/496160109.svg)](https://zenodo.org/badge/latestdoi/496160109)
[![Documentation Status](https://readthedocs.org/projects/python-cmethods/badge/?version=stable)](https://python-cmethods.readthedocs.io/en/latest/?badge=stable)

</div>

Welcome to python-cmethods, a powerful Python package designed for bias
correction and adjustment of climate data. Built with a focus on ease of use and
efficiency, python-cmethods offers a comprehensive suite of functions tailored
for applying bias correction methods to climate model simulations and
observational datasets via command-line interface and API.

Please cite this project as described in
https://zenodo.org/doi/10.5281/zenodo.7652755.

## Table of Contents

1. [ About ](#about)
2. [ Available Methods ](#methods)
3. [ Installation ](#installation)
4. [ Usage and Examples ](#examples)
5. [ Notes ](#notes)
6. [ Contribution ](#contribution)
7. [ References ](#references)

<a name="about"></a>

## 1. About

Bias correction in climate research involves the adjustment of systematic errors
or biases present in climate model simulations or observational datasets to
improve their accuracy and reliability, ensuring that the data better represents
actual climate conditions. This process typically involves statistical methods
or empirical relationships to correct for biases caused by factors such as
instrument calibration, spatial resolution, or model deficiencies.

<figure>
  <img
  src="doc/_static/images/biasCdiagram.png?raw=true"
  alt="Schematic representation of a bias adjustment procedure"
  style="background-color: white; border-radius: 7px">
  <figcaption>Figure 1: Schematic representation of a bias adjustment procedure</figcaption>
</figure>

python-cmethods empowers scientists to effectively address those biases in
climate data, ensuring greater accuracy and reliability in research and
decision-making processes. By leveraging cutting-edge techniques and seamless
integration with popular libraries like [xarray](https://xarray.dev/) and
[Dask](https://docs.dask.org/en/stable/), this package simplifies the process
of bias adjustment, even when dealing with large-scale climate simulations and
extensive spatial domains.

In this way, for example, modeled data, which on average represent values that
are too cold, can be easily bias-corrected by applying any adjustment procedure
included in this package.

For instance, modeled data can report values that are way colder than the those
data reported by reanalysis time-series. To address this issue, an adjustment
procedure can be employed. The figure below illustrates the observed, modeled,
and adjusted values, revealing that the delta-adjusted time series
($T^{*DM}_{sim,p}$) is significantly more similar to the observational data
($T{obs,p}$) than the raw model output ($T_{sim,p}$).

<figure>
  <img
  src="doc/_static/images/dm-doy-plot.png?raw=true"
  alt="Temperature per day of year in modeled, observed and bias-adjusted climate data"
  style="background-color: white; border-radius: 7px">
  <figcaption>Figure 2: Temperature per day of year in observed, modeled, and bias-adjusted climate data</figcaption>
</figure>

The mathematical foundations supporting each bias correction technique
implemented in python-cmethods are integral to the package, ensuring
transparency and reproducibility in the correction process. Each method is
accompanied by references to trusted publications, reinforcing the reliability
and rigor of the corrections applied.

<a name="methods"></a>

## 2. Available Methods

python-cmethods provides the following bias correction techniques:

- Linear Scaling
- Variance Scaling
- Delta Method
- Quantile Mapping
- Detrended Quantile Mapping
- Quantile Delta Mapping

Please refer to the official documentation for more information about these
methods as well as sample scripts:
https://python-cmethods.readthedocs.io/en/stable/

- Except for the variance scaling, all methods can be applied on stochastic and
  non-stochastic climate variables. Variance scaling can only be applied on
  non-stochastic climate variables.

  - Non-stochastic climate variables are those that can be predicted with
    relative certainty based on factors such as location, elevation, and season.
    Examples of non-stochastic climate variables include air temperature, air
    pressure, and solar radiation.

  - Stochastic climate variables, on the other hand, are those that exhibit a
    high degree of variability and unpredictability, making them difficult to
    forecast accurately. Precipitation is an example of a stochastic climate
    variable because it can vary greatly in timing, intensity, and location due
    to complex atmospheric and meteorological processes.

- Except for the detrended quantile mapping (DQM) technique, all methods can be
  applied to 1- and 3-dimensional data sets. The implementation of DQM to
  3-dimensional data is still in progress.

- Except for DQM, all methods can be applied using `cmethods.adjust`. Chunked
  data for computing e.g. in a dask cluster is possible as well.

- For any questions -- please open an issue at
  https://github.com/btschwertfeger/python-cmethods/issues

<a name="installation"></a>

## 3. Installation

> For optimal compatibility on macOS, ensure that 'hdf5' and 'netcdf' are
> pre-installed using Homebrew (`brew install hdf5 netcdf`).

```bash
python3 -m pip install python-cmethods
```

<a name="examples"></a>

## 4. CLI Usage

The python-cmethods package provides a command-line interface for applying
various bias correction methods out of the box.

Keep in mind that due to the various kinds of data and possibilities to
pre-process those, the CLI only provides a basic application of the implemented
techniques. For special parameters, adjustments, and data preparation, please
use programming interface.

Listing the parameters and their requirements is available by passing the
`--help` option:

```bash
cmethods --help
```

Applying the cmethods tool on the provided example data using the linear scaling
approach is shown below:

```bash
cmethods \
  --obs examples/input_data/observations.nc \
  --simh examples/input_data/control.nc \
  --simp examples/input_data/scenario.nc \
  --method linear_scaling \
  --kind add \
  --variable tas \
  --group time.month \
  --output linear_scaling.nc

2024/04/08 18:11:12     INFO | Loading data sets ...
2024/04/08 18:11:12     INFO | Data sets loaded ...
2024/04/08 18:11:12     INFO | Applying linear_scaling ...
2024/04/08 18:11:15     INFO | Saving result to linear_scaling.nc ...
```

For applying a distribution-based bias correction technique, the following
example may help:

```bash
cmethods \
  --obs examples/input_data/observations.nc \
  --simh examples/input_data/control.nc \
  --simp examples/input_data/scenario.nc \
  --method quantile_delta_mapping \
  --kind add \
  --variable tas \
  --quantiles 1000 \
  --output quantile_delta_mapping.nc

2024/04/08 18:16:34     INFO | Loading data sets ...
2024/04/08 18:16:35     INFO | Data sets loaded ...
2024/04/08 18:16:35     INFO | Applying quantile_delta_mapping ...
2024/04/08 18:16:35     INFO | Saving result to quantile_delta_mapping.nc ...
```

## 5. Programming Interface Usage and Examples

```python
import xarray as xr
from cmethods import adjust

obsh = xr.open_dataset("input_data/observations.nc")
simh = xr.open_dataset("input_data/control.nc")
simp = xr.open_dataset("input_data/scenario.nc")

# adjust only one grid cell
ls_result = adjust(
    method="linear_scaling",
    obs=obsh["tas"][:, 0, 0],
    simh=simh["tas"][:, 0, 0],
    simp=simp["tas"][:, 0, 0],
    kind="+",
    group="time.month",
)

# adjust all grid cells
qdm_result = adjust(
    method="quantile_delta_mapping",
    obs=obsh["tas"],
    simh=simh["tas"],
    simp=simp["tas"],
    n_quantiles=1000,
    kind="+",
)

# to calculate the relative rather than the absolute change,
# '*' can be used instead of '+' (this is preferred when adjusting
# stochastic variables like precipitation)
```

It is also possible to adjust chunked data sets. Feel free to have a look into
`tests/test_zarr_dask_compatibility.py` to get a starting point.

Notes:

- For the multiplicative techniques a maximum scaling factor of 10 is defined.
  This can be changed by passing the optional parameter `max_scaling_factor`.
- Except for detrended quantile mapping, all implemented techniques can be
  applied to single and multi-dimensional data sets by executing the
  `cmethods.adjust` function.
- A Jupyter notebook applying all those methods is provided here:
  `/examples/examples.ipynb`
- The example data is located at: `/examples/input_data/*.nc`

<a name="notes"></a>

## 5. Notes

- Computation in Python takes some time, so this is only for demonstration. When
  adjusting large datasets, you should either use chunked data using for example
  a dask cluster or to apply the command-line tool
  [BiasAdjustCXX](https://github.com/btschwertfeger/BiasAdjustCXX).
- Formulas and references can be found in the implementations of the
  corresponding functions, on the bottom of the README.md and in the
  [documentation](https://python-kraken-sdk.readthedocs.io/en/stable/).

### Space for improvements

- Since the scaling methods implemented so far scale by default over the mean
  values of the respective months, unrealistic long-term mean values may occur
  at the month transitions. This can be prevented either by selecting
  `group='time.dayofyear'`. Alternatively, it is possible not to scale using
  long-term mean values, but using a 31-day interval, which takes the 31
  surrounding values over all years as the basis for calculating the mean
  values. This is not yet implemented, because even the computation for this
  takes so much time, that it is not worth implementing it in python - but this
  is available in
  [BiasAdjustCXX](https://github.com/btschwertfeger/BiasAdjustCXX).

<a name="contribution"></a>

## 6. 🆕 Contributions

… are welcome but:

- First check if there is an existing issue or PR that addresses your
  problem/solution. If not - create one first - before creating a PR.
- Typo fixes, project configuration, CI, documentation or style/formatting PRs will be
  rejected. Please create an issue for that.
- PRs must provide a reasonable, easy to understand and maintain solution for an
  existing problem. You may want to propose a solution when creating the issue
  to discuss the approach before creating a PR.

<a name="references"></a>

## 7. References

- Schwertfeger, Benjamin Thomas and Lohmann, Gerrit and Lipskoch, Henrik (2023) _"Introduction of the BiasAdjustCXX command-line tool for the application of fast and efficient bias corrections in climatic research"_, SoftwareX, Volume 22, 101379, ISSN 2352-7110, (https://doi.org/10.1016/j.softx.2023.101379)
- Schwertfeger, Benjamin Thomas (2022) _"The influence of bias corrections on variability, distribution, and correlation of temperatures in comparison to observed and modeled climate data in Europe"_ (https://epic.awi.de/id/eprint/56689/)
- Linear Scaling and Variance Scaling based on: Teutschbein, Claudia and Seibert, Jan (2012) _"Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods"_ (https://doi.org/10.1016/j.jhydrol.2012.05.052)
- Delta Method based on: Beyer, R. and Krapp, M. and Manica, A.: _"An empirical evaluation of bias correction methods for palaeoclimate simulations"_ (https://doi.org/10.5194/cp-16-1493-2020)
- Quantile and Detrended Quantile Mapping based on: Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock _"Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes?"_ (https://doi.org/10.1175/JCLI-D-14-00754.1)
- Quantile Delta Mapping based on: Tong, Y., Gao, X., Han, Z. et al. _"Bias correction of temperature and precipitation over China for RCM simulations using the QM and QDM methods"_. Clim Dyn 57, 1425–1443 (2021). (https://doi.org/10.1007/s00382-020-05447-4)
- I'd like to express my gratitude to @riley-brady for initiating and
  contributing to the discussion on
  https://github.com/btschwertfeger/python-cmethods/issues/47. I appreciate all
  the valuable suggestions provided throughout the implementation of the
  subsequent changes.
