Unit tests for python-cmethods
##############################

The input data sets are generated before the tests are executed. They are based
on simple equations to simulate temperatures and precipitation for different
locations. The bias correction methods are then applied to those to to then
validate if the methods improved the data.

Additionally some input data is saved as .zarr, so the dask compatibility can be
tested as well.
