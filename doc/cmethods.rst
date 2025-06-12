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

Classes and Functions
=====================

In past versions of the python-cmethods package (v1.x) there was a "CMethods"
class that implemented the bias correction methods. This class was removed in
version v2.0.0. Since then, the ``cmethods.adjust`` function is used to apply
the implemented techniques except for detrended quantile mapping.

.. autofunction:: cmethods.adjust
.. automethod:: cmethods.distribution.detrended_quantile_mapping

Some additional methods
-----------------------

.. automethod:: cmethods.utils.get_pdf
.. automethod:: cmethods.utils.get_cdf
.. automethod:: cmethods.utils.get_inverse_of_cdf
