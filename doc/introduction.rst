.. -*- coding: utf-8 -*-
.. Copyright (C) 2023 Benjamin Thomas Schwertfeger
.. GitHub: https://github.com/btschwertfeger
..

python-cmethods
===============

|GitHub badge| |License badge| |PyVersions badge| |Downloads badge|
|CI/CD badge| |codecov badge| |OSSF Scorecard| |OSSF Best Practices|
|Release date badge| |Release version badge| |DOI badge| |Docs stable|

About
-----

Welcome to `python-cmethods`_, a powerful Python package designed for bias
correction and adjustment of climate data. Built with a focus on ease of use and
efficiency, python-cmethods offers a comprehensive suite of functions tailored
for applying bias correction methods to climate model simulations and
observational datasets.

Bias correction in climate research involves the adjustment of systematic errors
or biases present in climate model simulations or observational datasets to
improve their accuracy and reliability, ensuring that the data better represents
actual climate conditions. This process typically involves statistical methods
or empirical relationships to correct for biases caused by factors such as
instrument calibration, spatial resolution, or model deficiencies.

.. figure:: _static/images/biasCdiagram.png
    :width: 600
    :align: center
    :alt: Schematic representation of a bias adjustment procedure

    Fig 1: Schematic representation of a bias adjustment procedure

In this way, for example, modeled data, which on average represent values that
are too cold, can be easily bias-corrected by applying any adjustment procedure
included in this package.

For instance, modeled data can report values that are way colder than the those
data reported by reanalysis time-series. To address this issue, an adjustment
procedure can be employed. The figure below illustrates the observed, modeled,
and adjusted values, revealing that the delta-adjusted time series
(:math:`T^{*DM}_{sim,p}`) is significantly more similar to the observational
data (:math:`T_{obs,p}`) than the raw model output (:math:`T_{sim,p}`).

.. figure:: _static/images/dm-doy-plot.png
    :width: 600
    :align: center
    :alt: Temperature per day of year in modeled, observed and bias-adjusted climate data

    Fig 2: Temperature per day of year in modeled, observed and bias-adjusted climate data

The mathematical foundations supporting each bias correction technique
implemented in python-cmethods are integral to the package, ensuring
transparency and reproducibility in the correction process. Each method is
accompanied by references to trusted publications, reinforcing the reliability
and rigor of the corrections applied.


Available Methods
-----------------

python-cmethods provides the following bias correction techniques:

- :ref:`Linear Scaling<linear-scaling>`
- :ref:`Variance Scaling<variance-scaling>`
- :ref:`Delta Method<delta-method>`
- :ref:`Quantile Mapping<quantile-mapping>`
- :ref:`Detrended Quantile Mapping<detrended-quantile-mapping>`
- :ref:`Quantile Delta Mapping<quantile-delta-mapping>`

Please refer to the official documentation for more information about these
methods as well as sample scripts:
https://python-cmethods.readthedocs.io/en/stable/

- Except for the variance scaling, all methods can be applied on stochastic and
  non-stochastic climate variables. Variance scaling can only be applied on
  non-stochastic climate variables.

  - Non-stochastic climate variables are those that can be predicted with relative
    certainty based on factors such as location, elevation, and season. Examples
    of non-stochastic climate variables include air temperature, air pressure, and
    solar radiation.

  - Stochastic climate variables, on the other hand, are those that exhibit a high
    degree of variability and unpredictability, making them difficult to forecast
    accurately. Precipitation is an example of a stochastic climate variable
    because it can vary greatly in timing, intensity, and location due to complex
    atmospheric and meteorological processes.

- Except for the detrended quantile mapping (DQM) technique, all methods can be
  applied to single and multidimensional data sets. The implementation of DQM to
  3-dimensional data is still in progress.

- For any questions -- please open an issue at
  https://github.com/btschwertfeger/python-cmethods/issues. Examples can be found
  in the `python-cmethods`_ repository and of course within this documentation.

References
----------

- Schwertfeger, Benjamin Thomas and Lohmann, Gerrit and Lipskoch, Henrik (2023) *"Introduction of the BiasAdjustCXX command-line tool for the application of fast and efficient bias corrections in climatic research"*, SoftwareX, Volume 22, 101379, ISSN 2352-7110, (https://doi.org/10.1016/j.softx.2023.101379)
- Schwertfeger, Benjamin Thomas (2022) *"The influence of bias corrections on variability, distribution, and correlation of temperatures in comparison to observed and modeled climate data in Europe"* (https://epic.awi.de/id/eprint/56689/)
- Linear Scaling and Variance Scaling based on: Teutschbein, Claudia and Seibert, Jan (2012) *"Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods"* (https://doi.org/10.1016/j.jhydrol.2012.05.052)
- Delta Method based on: Beyer, R. and Krapp, M. and Manica, A.: *"An empirical evaluation of bias correction methods for palaeoclimate simulations"* (https://doi.org/10.5194/cp-16-1493-2020)
- Quantile and Detrended Quantile Mapping based on: Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock *"Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods Preserve Changes in Quantiles and Extremes?"* (https://doi.org/10.1175/JCLI-D-14-00754.1)
- Quantile Delta Mapping based on: Tong, Y., Gao, X., Han, Z. et al. *"Bias correction of temperature and precipitation over China for RCM simulations using the QM and QDM methods"*. Clim Dyn 57, 1425–1443 (2021). (https://doi.org/10.1007/s00382-020-05447-4)
