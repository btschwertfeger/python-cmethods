Known Issues
============

- Since the scaling methods implemented so far scale by default over the mean
  values of the respective months, unrealistic long-term mean values may occur
  at the month transitions. This can be prevented either by selecting
  ``group='time.dayofyear'```. Alternatively, it is possible not to scale using
  long-term mean values, but using a 31-day interval, which takes the 31
  surrounding values over all years as the basis for calculating the mean
  values. This is not yet implemented in this module, but is available in the
  command-line tool `BiasAdjustCXX`_.
- Python can be very slow when applying mathematical calculations on large data
  sets or when iterating over high resolution data. Using this module or
  especially Python to apply bias correction techniques on large data sets can
  be a very time-consuming task. So this module is more about showing how to
  apply different methods on climate data and maybe even to bias-correct small
  data sets. When it comes to large ensenblse it is preferred to use the way
  more efficient tool `BiasAdjustCXX`_. A speed comparison between
  `python-cmethods`_, `BiasAdjustCXX`_, and `xclim`_ was made this `tool
  comparison`_.
