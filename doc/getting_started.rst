Getting Started
===============

Installation
------------

The `python-cmethods`_ module can be installed using the package manager pip:

.. code-block:: bash

    python3 -m pip install python-cmethods


Command-Line Interface
----------------------

The python-cmethods package provides a command-line interface for applying
various bias correction methods out of the box.

Listing the parameters and their requirements is available by passing the
``--help`` option:

.. code-block:: bash

    cmethods --help

Applying the cmethods tool on the provided example data using the linear scaling
approach is shown below:

.. code-block:: bash
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

For applying a distribution-based bias correction technique, the following
example may help:

.. code-block:: bash
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


API Usage and Examples
------------------

The `python-cmethods`_ module can be imported and applied as showing in the
following examples. For more detailed description of the methods, please have a
look at the method specific documentation.

.. code-block:: python
    :linenos:
    :caption: Apply the Linear Scaling bias correction technique on 1-dimensional data

    import xarray as xr
    from cmethods import adjust

    obsh = xr.open_dataset("input_data/observations.nc")
    simh = xr.open_dataset("input_data/control.nc")
    simp = xr.open_dataset("input_data/scenario.nc")

    ls_result = adjust(
        method="linear_scaling",
        obs=obsh["tas"][:, 0, 0],
        simh=simh["tas"][:, 0, 0],
        simp=simp["tas"][:, 0, 0],
        kind="+",
    )

.. code-block:: python
    :linenos:
    :caption: Apply the Quantile Delta Mapping bias correction technique on 3-dimensional data

    import xarray as xr
    from cmethods import adjust

    obsh = xr.open_dataset("input_data/observations.nc")
    simh = xr.open_dataset("input_data/control.nc")
    simp = xr.open_dataset("input_data/scenario.nc")

    qdm_result = adjust(
        method="quantile_delta_mapping",
        obs=obsh["tas"],
        simh=simh["tas"],
        simp=simp["tas"],
        n_quaniles=1000,
        kind="+",
    )


Advanced Usage
--------------

In some cases the time dimension of input data sets have different sizes. In
such case, the hidden parameter ``input_core_dims`` must be passed to the
``adjust`` call.

It defines the dimension names of the input data sets, i.e. if the time
dimensions of ``obs`` and ``simp`` have the length, but the time dimension of
``simh`` is somewhat smaller, you have to define this as follows:


.. code-block:: python
    :linenos:
    :caption: Bias Adjustments for data sets with different time dimension lengths pt. 1

    from cmethods import adjust
    import xarray as xr

    obs = xr.open_dataset("examples/input_data/observations.nc")["tas"]
    simp = xr.open_dataset("examples/input_data/control.nc")["tas"]
    simh = simp.copy(deep=True)[3650:]

    bc = adjust(
        method="quantile_mapping",
        obs=obs,
        simh=simh.rename({"time": "t_simh"}),
        simp=simh,
        kind="+",
        input_core_dims={"obs": "time", "simh": "t_simh", "simp": "time"}
    )

In case you are applying a scaling based technique using grouping, you have to
adjust the group names accordingly to the time dimension names.

.. code-block:: python
    :linenos:
    :caption: Bias Adjustments for data sets with different time dimension lengths pt. 2

    from cmethods import adjust
    import xarray as xr

    obs = xr.open_dataset("examples/input_data/observations.nc")["tas"]
    simp = xr.open_dataset("examples/input_data/control.nc")["tas"]
    simh = simp.copy(deep=True)[3650:]

    bc = adjust(
        method="linear_scaling",
        obs=obs,
        simh=simh.rename({"time": "t_simh"}),
        simp=simh,
        kind="+",
        group={"obs": "time.month", "simh": "t_simh.month", "simp": "time.month"},
        input_core_dims={"obs": "time", "simh": "t_simh", "simp": "time"}
    )
