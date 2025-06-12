.. -*- mode: rst; coding: utf-8 -*-
..
.. Copyright (C) 2023 Benjamin Thomas Schwertfeger
.. https://github.com/btschwertfeger
..
.. This program is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, either version 3 of the License, or
.. (at your option) any later version.
..
.. This program is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
..
.. You should have received a copy of the GNU General Public License
.. along with this program. If not, see
.. https://www.gnu.org/licenses/gpl-3.0.html.
..

Bias Correction Methods
=======================

General
-------

- Please note that the formulas are meant to be applied to a single time series.
  The python-cmethods package can apply these formulas to single and
  multidimensional data.
- Selecting the appropriate number of quantiles for distribution-based
  techniques is a challenging task. The number of quantiles determines the
  number of bins of the probability density function and the cumulative
  distribution function, which represent the distribution of the time series.
  The number of quantiles should be large enough to capture the nuances of the
  distribution functions but not excessively high, as this can result in a
  straight-line distribution, which lead to artificial bias while interpolating.


.. _linear-scaling:

Linear Scaling
--------------

The Linear Scaling bias correction technique can be applied on stochastic and
non-stochastic climate variables to minimize deviations in the mean values
between predicted and observed time-series of past and future time periods.

This method requires that the time series can be grouped by ``time.month``.

Since the multiplicative scaling can result in very high scaling factors, a
maximum scaling factor of 10 is set. This can be changed by passing the desired
value to the hidden ``max_scaling_factor`` argument.

The Linear Scaling bias correction technique implemented here is based on the
method described in the equations of Teutschbein, Claudia and Seibert, Jan
(2012) *"Bias correction of regional climate model simulations for hydrological
climate-change impact studies: Review and evaluation of different methods"*
(https://doi.org/10.1016/j.jhydrol.2012.05.052). In the following the equations
for both additive and multiplicative Linear Scaling are shown:

**Additive**:

    In Linear Scaling, the long-term monthly mean (:math:`\mu_m`) of the modeled
    data :math:`X_{sim,h}` is subtracted from the long-term monthly mean of the
    reference data :math:`X_{obs,h}` at time step :math:`i`. This difference in
    month-dependent long-term mean is than added to the value of time step
    :math:`i`, in the time-series that is to be adjusted (:math:`X_{sim,p}`).

    .. math::

        X^{*LS}_{sim,p}(i) = X_{sim,p}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

where:

    .. math::

        \mu_m(X\ldots(i)) = \text{long-term monthly mean of month related to day at index } i

**Multiplicative**:

    The multiplicative Linear Scaling differs from the additive variant in such
    way, that the changes are not computed in absolute but in relative values.

    .. math::

        X^{*LS}_{sim,h}(i) = X_{sim,h}(i) \cdot \left[\frac{\mu_{m}(X_{obs,h}(i))}{\mu_{m}(X_{sim,h}(i))}\right]

where:

    .. math::

        \mu_m(X\ldots(i)) = \text{long-term monthly mean of month related to day at index } i


.. code-block:: python
    :linenos:
    :caption: Example: Linear Scaling

    >>> import xarray as xr
    >>> from cmethods import adjust

    >>> # Note: The data sets must contain the dimension "time"
    >>> #       for the respective variable.
    >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
    >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
    >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
    >>> variable = "tas" # temperatures
    >>> result = adjust(
    ...     method="linear_scaling",
    ...     obs=obsh[variable],
    ...     simh=simh[variable],
    ...     simp=simp[variable],
    ...     kind="+",
    ...     group="time.month" # this is important!
    ... )

.. _variance-scaling:

Variance Scaling
----------------

The Variance Scaling bias correction technique can be applied only on
non-stochastic climate variables to minimize deviations in the mean and variance
between predicted and observed time-series of past and future time periods.

This method requires that the time series can be grouped by ``time.month``.

Since the the scaling by ratio can result in very high scaling factors, a
maximum scaling factor of 10 is set. This can be changed by passing the desired
value to the hidden ``max_scaling_factor`` argument.

The Variance Scaling bias correction technique implemented here is based on the
method described in the equations of Teutschbein, Claudia and Seibert, Jan
(2012) *"Bias correction of regional climate model simulations for hydrological
climate-change impact studies: Review and evaluation of different methods"*
(https://doi.org/10.1016/j.jhydrol.2012.05.052). In the following the equations
of the Variance Scaling approach are shown:

**(1)** First, the modeled data of the control and scenario period must be
bias-corrected using the additive linear scaling technique. This adjusts the
deviation in the mean.

.. math::

    X^{*LS}_{sim,h}(i) = X_{sim,h}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

    X^{*LS}_{sim,p}(i) = X_{sim,p}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

where:

    .. math::

        \mu_m(X\ldots(i)) = \text{long-term monthly mean of month related to day at index } i

**(2)** In the second step, the time-series are shifted to a zero mean. This
enables the adjustment of the standard deviation in the following step.

.. math::

    X^{VS(1)}_{sim,h}(i) = X^{*LS}_{sim,h}(i) - \mu_{m}(X^{*LS}_{sim,h}(i))

    X^{VS(1)}_{sim,p}(i) = X^{*LS}_{sim,p}(i) - \mu_{m}(X^{*LS}_{sim,p}(i))

**(3)** Now the standard deviation (so variance too) can be scaled.

.. math::

    X^{VS(2)}_{sim,p}(i) = X^{VS(1)}_{sim,p}(i) \cdot \left[\frac{\sigma_{m}(X_{obs,h}(i))}{\sigma_{m}(X^{VS(1)}_{sim,h}(i))}\right]

**(4)** Finally the previously removed mean is shifted back

.. math::

    X^{*VS}_{sim,p}(i) = X^{VS(2)}_{sim,p}(i) + \mu_{m}(X^{*LS}_{sim,p}(i))

.. code-block:: python
    :linenos:
    :caption: Example: Variance Scaling

    >>> import xarray as xr
    >>> from cmethods import adjust

    >>> # Note: The data sets must contain the dimension "time"
    >>> #       for the respective variable.
    >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
    >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
    >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
    >>> variable = "tas" # temperatures
    >>> result = adjust(
    ...     method="variance_scaling",
    ...     obs=obsh[variable],
    ...     simh=simh[variable],
    ...     simp=simp[variable],
    ...     kind="+",
    ...     group="time.month" # this is important!
    ... )

.. _delta-method:

Delta Method
------------

The Delta Method bias correction technique can be applied on stochastic and
non-stochastic climate variables to minimize deviations in the mean values
between predicted and observed time-series of past and future time periods.

This method requires that the time series can be grouped by ``time.month`` while
the reference data of the control period must have the same temporal resolution
as the data that is going to be adjusted.

Since the multiplicative scaling can result in very high scaling factors, a
maximum scaling factor of 10 is set. This can be changed by passing the desired
value to the hidden ``max_scaling_factor`` argument.

The Delta Method bias correction technique implemented here is based on the
method described in the equations of Beyer, R. and Krapp, M. and Manica, A. (2020)
*"An empirical evaluation of bias correction methods for paleoclimate simulations"*
(https://doi.org/10.5194/cp-16-1493-2020). In the following the equations
for both additive and multiplicative Delta Method are shown:

**Additive**:

    The Delta Method looks like the Linear Scaling method but the important
    difference is, that the Delta method uses the change between the modeled
    data instead of the difference between the modeled and reference data of the
    control period. This means that the long-term monthly mean (:math:`\mu_m`)
    of the modeled data of the control period :math:`T_{sim,h}` is subtracted
    from the long-term monthly mean of the modeled data from the scenario period
    :math:`T_{sim,p}` at time step :math:`i`. This change in month-dependent
    long-term mean is than added to the long-term monthly mean for time step
    :math:`i`, in the time-series that represents the reference data of the
    control period (:math:`T_{obs,h}`).

    .. math::

        X^{*DM}_{sim,p}(i) = X_{obs,h}(i) + \mu_{m}(X_{sim,p}(i)) - \mu_{m}(X_{sim,h}(i))

where:

    .. math::

        \mu_m(X\ldots(i)) = \text{long-term monthly mean of month related to day at index } i

**Multiplicative**:

    The multiplicative variant behaves like the additive, but with the
    difference that the change is computed using the relative change instead of
    the absolute change.

    .. math::

        X^{*DM}_{sim,p}(i) = X_{obs,h}(i) \cdot \left[\frac{ \mu_{m}(X_{sim,p}(i)) }{ \mu_{m}(X_{sim,h}(i))}\right]

where:

    .. math::

        \mu_m(X\ldots(i)) = \text{long-term monthly mean of month related to day at index } i

.. code-block:: python
    :linenos:
    :caption: Example: Delta Method

    >>> import xarray as xr
    >>> from cmethods import adjust

    >>> # Note: The data sets must contain the dimension "time"
    >>> #       for the respective variable.
    >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
    >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
    >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
    >>> variable = "tas" # temperatures
    >>> result = adjust(
    ...     method="delta_method",
    ...     obs=obsh[variable],
    ...     simh=simh[variable],
    ...     simp=simp[variable],
    ...     kind="+",
    ...     group="time.month" # this is important!
    ... )

.. _quantile-mapping:

Quantile Mapping
----------------
The Quantile Mapping bias correction technique can be used to minimize
distributional biases between modeled and observed time-series climate data. Its
interval-independent behavior ensures that the whole time series is taken into
account to redistribute its values, based on the distributions of the modeled
and observed/reference data of the control period.

The Quantile Mapping technique implemented here is based on the equations of
Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock (2015) *"Bias
Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods
Preserve Changes in Quantiles and Extremes?"*
(https://doi.org/10.1175/JCLI-D-14-00754.1).

The regular Quantile Mapping is bounded to the value range of the modeled data
of the control period. To avoid this, the Detrended Quantile Mapping can be
used.

In the following the equations of Alex J. Cannon (2015) are shown and explained:

**Additive**:

    .. math::

        X^{*QM}_{sim,p}(i) = F^{-1}_{obs,h} \left\{F_{sim,h}\left[X_{sim,p}(i)\right]\right\}


    The additive quantile mapping procedure consists of inserting the value to
    be adjusted (:math:`X_{sim,p}(i)`) into the cumulative distribution function
    of the modeled data of the control period (:math:`F_{sim,h}`). This
    determines, in which quantile the value to be adjusted can be found in the
    modeled data of the control period The following images show this by using
    :math:`T` for temperatures.

    .. figure:: _static/images/qm-cdf-plot-1.png
        :width: 600
        :align: center
        :alt: Determination of the quantile value

        Fig 1: Inserting :math:`X_{sim,p}(i)` into :math:`F_{sim,h}` to determine the quantile value

    This value, which of course lies between 0 and 1, is subsequently inserted
    into the inverse cumulative distribution function of the reference data of
    the control period to determine the bias-corrected value at time step
    :math:`i`.

    .. figure:: _static/images/qm-cdf-plot-2.png
        :width: 600
        :align: center
        :alt: Determination of the QM bias-corrected value

        Fig 1: Inserting the quantile value into :math:`F^{-1}_{obs,h}` to determine the bias-corrected value for time step :math:`i`

**Multiplicative**:

    The formula is the same as for the additive variant, but the values are
    bound to the lower level of zero. The upper and lower boundary can be
    adjusted by passing the hidden arguments ``val_min`` and ``val_max``.

.. code-block:: python
    :linenos:
    :caption: Example: Quantile Mapping

    >>> import xarray as xr
    >>> from cmethods import adjust

    >>> # Note: The data sets must contain the dimension "time"
    >>> #       for the respective variable.
    >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
    >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
    >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
    >>> variable = "tas" # temperatures
    >>> qm_adjusted = adjust(
    ...     method="quantile_mapping",
    ...     obs=obsh[variable],
    ...     simh=simh[variable],
    ...     simp=simp[variable],
    ...     n_quantiles=250,
    ...     kind="+",
    ... )

.. _detrended-quantile-mapping:

Detrended Quantile Mapping
--------------------------

The Detrended Quantile Mapping bias correction technique can be used to minimize
distributional biases between modeled and observed time-series climate data like
the regular Quantile Mapping. Detrending means, that the values of
:math:`X_{sim,p}` are shifted by the mean of :math:`X_{sim,p}` before the
regular Quantile Mapping is applied. The shift is performed on a monthly basis.
After the Quantile Mapping was applied, the mean is shifted back. Since it does
not make sense to take the whole mean to rescale the data, the month-dependent
long-term mean is used.

This method must be applied on a 1-dimensional data set i.e., there is only one
time-series passed for each of ``obs``, ``simh``, and ``simp``. This method
requires that the time series can be grouped by ``time.month``.

Since the ratio when applying the multiplicative variant can result in extreme
factors, a maximum scaling factor of 10 is set. This can be changed by passing
the desired value to the hidden ``max_scaling_factor`` argument.

The Detrended Quantile Mapping technique implemented here is based on the
equations of Alex J. Cannon and Stephen R. Sobie and Trevor Q. Murdock (2015)
*"Bias Correction of GCM Precipitation by Quantile Mapping: How Well Do Methods
Preserve Changes in Quantiles and Extremes?"*
(https://doi.org/10.1175/JCLI-D-14-00754.1).

The following equations qre based on Alex J. Cannon (2015) but extended the
shift of :math:`X_{sim,p}(i)`:

**Additive**:

    .. math::

        X_{sim,p}^{*DT}(i) & = X_{sim,p}(i) - \mu_m(X_{sim,p}(i)) \\[1pt]
        X_{sim,p}^{*DQM}(i) & = F_{obs,h}^{-1}\left\{F_{sim,h}\left[X_{sim,p}^{*DT}(i)\right]\right\}

where:

    .. math::

        \mu_m(X\ldots(i)) = \text{long-term monthly mean of month related to day at index } i

**Multiplicative**:

    .. math::

        X^{*DQM}_{sim,p}(i) = F^{-1}_{obs,h}\Biggl\{F_{sim,h}\left[\frac{\mu_m(X_{sim,h}(i)) \cdot X_{sim,p}(i)}{\mu_m(X_{sim,p}(i))}\right]\Biggr\}\frac{\mu_m(X_{sim,p}(i))}{\mu_m(X_{sim,h}(i))}

where:

    .. math::

        \mu_m(X\ldots(i)) = \text{long-term monthly mean of month related to day at index } i

.. code-block:: python
    :linenos:
    :caption: Example: Quantile Mapping

    >>> import xarray as xr
    >>> from cmethods.distribution import detrended_quantile_mapping

    >>> # Note: The data sets must contain the dimension "time"
    >>> #       for the respective variable.
    >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
    >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
    >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
    >>> variable = "tas" # temperatures
    >>> qm_adjusted = detrended_quantile_mapping(
    ...     obs=obsh[variable],
    ...     simh=simh[variable],
    ...     simp=simp[variable],
    ...     n_quantiles=250
    ...     kind="+"
    ... )


.. _quantile-delta-mapping:

Quantile Delta Mapping
-----------------------

The Quantile Delta Mapping bias correction technique can be used to minimize
distributional biases between modeled and observed time-series climate data. Its
interval-independent behavior ensures that the whole time series is taken into
account to redistribute its values, based on the distributions of the modeled
and observed/reference data of the control period. In contrast to the regular
Quantile Mapping (:func:`cmethods.CMethods.quantile_mapping`) the Quantile Delta
Mapping also takes the change between the modeled data of the control and
scenario period into account.

Since the ratio when applying the multiplicative variant can result in extreme
factors, a maximum scaling factor of 10 is set. This can be changed by passing
the desired value to the hidden ``max_scaling_factor`` argument.

The Quantile Delta Mapping technique implemented here is based on the equations
of Tong, Y., Gao, X., Han, Z. et al. (2021) *"Bias correction of temperature and
precipitation over China for RCM simulations using the QM and QDM methods"*.
Clim Dyn 57, 1425-1443 (https://doi.org/10.1007/s00382-020-05447-4). In the
following the additive and multiplicative variant are shown.

**Additive**:

    **(1.1)** In the first step the quantile value of the time step :math:`i` to adjust is stored in
    :math:`\varepsilon(i)`.

    .. math::

        \varepsilon(i) = F_{sim,p}\left[X_{sim,p}(i)\right], \hspace{1em} \varepsilon(i)\in\{0,1\}

    **(1.2)** The bias corrected value at time step :math:`i` is now determined
    by inserting the quantile value into the inverse cumulative distribution
    function of the reference data of the control period. This results in a bias
    corrected value for time step :math:`i` but still without taking the change
    in modeled data into account.

    .. math::

        X^{QDM(1)}_{sim,p}(i) = F^{-1}_{obs,h}\left[\varepsilon(i)\right]

    **(1.3)** The :math:`\Delta(i)` represents the absolute change in quantiles
    between the modeled value in the control and scenario period.

    .. math::

            \Delta(i) & = F^{-1}_{sim,p}\left[\varepsilon(i)\right] - F^{-1}_{sim,h}\left[\varepsilon(i)\right] \\[1pt]
                    & = X_{sim,p}(i) - F^{-1}_{sim,h}\left\{F^{}_{sim,p}\left[X_{sim,p}(i)\right]\right\}

    **(1.4)** Finally the previously calculated change can be added to the
    bias-corrected value.

    .. math::

        X^{*QDM}_{sim,p}(i) = X^{QDM(1)}_{sim,p}(i) + \Delta(i)

**Multiplicative**:

    The first two steps of the multiplicative Quantile Delta Mapping bias
    correction technique are the same as for the additive variant.

    **(2.3)** The :math:`\Delta(i)` in the multiplicative Quantile Delta Mapping
    is calculated like the additive variant, but using the relative than the
    absolute change.

        .. math::

            \Delta(i) & = \frac{ F^{-1}_{sim,p}\left[\varepsilon(i)\right] }{ F^{-1}_{sim,h}\left[\varepsilon(i)\right] } \\[1pt]
                        & = \frac{ X_{sim,p}(i) }{ F^{-1}_{sim,h}\left\{F_{sim,p}\left[X_{sim,p}(i)\right]\right\} }

    **(2.4)** The relative change between the modeled data of the control and
    scenario period is than multiplied with the bias-corrected value (see
    **1.2**).

        .. math::

            X^{*QDM}_{sim,p}(i) = X^{QDM(1)}_{sim,p}(i) \cdot \Delta(i)

.. code-block:: python
    :linenos:
    :caption: Example: Quantile Delta Mapping

    >>> import xarray as xr
    >>> from cmethods import adjust

    >>> # Note: The data sets must contain the dimension "time"
    >>> #       for the respective variable.
    >>> obsh = xr.open_dataset("path/to/reference_data-control_period.nc")
    >>> simh = xr.open_dataset("path/to/modeled_data-control_period.nc")
    >>> simp = xr.open_dataset("path/to/the_dataset_to_adjust-scenario_period.nc")
    >>> variable = "tas" # temperatures
    >>> qdm_adjusted = adjust(
    ...     method="quantile_delta_mapping",
    ...     obs=obsh[variable],
    ...     simh=simh[variable],
    ...     simp=simp[variable],
    ...     n_quantiles=250,
    ...     kind="+"
    ... )
