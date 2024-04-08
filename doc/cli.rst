Command-Line Interface
======================

The command-line interface provides the following help instructions

.. code-block:: bash

    cmethods --help

    Usage: cmethods [OPTIONS]

      Command line tool to apply bias adjustment procedures to climate data.

    Scaling-Based Adjustment Options:
      [all required if --method="linear_scaling" and --method="variance_scaling" and
      --method="delta_method"]
      --group TEXT                Temporal grouping

    Distribution-Based Adjustment Options:
      [all required if --method="quantile_mapping" and
      --method="quantile_delta_mapping"]
      --quantiles INTEGER         Quantiles to respect

    Other options:
      --version                   Show the version and exit.
      --obs, --observations PATH  Reference data set (control period)  [required]
      --simh, --simulated-historical PATH
                                Modeled data set (control period)  [required]
      --simp, --simulated-scenario PATH
                                Modeled data set (scenario period)  [required]
      --method [linear_scaling|variance_scaling|delta_method|quantile_mapping|quantile_delta_mapping]
                                Bias adjustment method to apply  [required]
      --kind [add|mult]           Kind of adjustment  [required]
      --variable TEXT             Variable of interest  [required]
      -o, --output TEXT           Output file name  [required]
      -h, --help                  Show this message and exit.
