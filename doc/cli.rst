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
