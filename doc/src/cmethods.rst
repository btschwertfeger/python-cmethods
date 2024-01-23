
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
