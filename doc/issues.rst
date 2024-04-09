.. -*- coding: utf-8 -*-
.. Copyright (C) 2023 Benjamin Thomas Schwertfeger
.. GitHub: https://github.com/btschwertfeger
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
