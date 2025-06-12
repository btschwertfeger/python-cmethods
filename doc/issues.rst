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

Known Issues
============

-  Since the scaling methods implemented so far scale by default over the mean
   values of the respective months, unrealistic long-term mean values may occur
   at the month transitions. This can be prevented either by selecting
   ``group='time.dayofyear'``. Alternatively, it is possible not to scale using
   long-term mean values, but using a 31-day interval, which takes the 31
   surrounding values over all years as the basis for calculating the mean
   values. This is not yet implemented in this module, but is available in the
   command-line tool `BiasAdjustCXX`_.
