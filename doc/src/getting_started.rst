Getting Started
===============

Installation
------------

The `python-cmethods`_ module can be installed using the package manager pip:

.. code-block:: bash

    python3 -m pip install python-cmethods


Usage and Examples
------------------

The `python-cmethods`_ module can be imported and applied as showing in the following examples.
For more detailed description of the methods, please have a look at the
method specific documentation.

.. code-block:: python
    :linenos:
    :caption: Apply the Linear Scaling bias correction technique on 1-dimensional data

    import xarray as xr
    from cmethods import CMethods as cm

    obsh = xr.open_dataset("input_data/observations.nc")
    simh = xr.open_dataset("input_data/control.nc")
    simp = xr.open_dataset("input_data/scenario.nc")

    ls_result = cm.adjust(
        mathod="linear_scaling",
        obs=obsh["tas"][:, 0, 0],
        simh=simh["tas"][:, 0, 0],
        simp=simp["tas"][:, 0, 0],
        kind="+",
    )

.. code-block:: python
    :linenos:
    :caption: Apply the Quantile Delta Mapping bias correction technique on 3-dimensional data

    import xarray as xr
    from cmethods import CMethods as cm

    obsh = xr.open_dataset("input_data/observations.nc")
    simh = xr.open_dataset("input_data/control.nc")
    simp = xr.open_dataset("input_data/scenario.nc")

    qdm_result = cm.adjust(
        method="quantile_delta_mapping",
        obs=obsh["tas"],
        simh=simh["tas"],
        simp=simp["tas"],
        n_quaniles=1000,
        kind="+",
    )
