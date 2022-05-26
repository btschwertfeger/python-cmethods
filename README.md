# Bias-Adjustment-Python

Collection of different scale- and distribution-based bias adjustments for climatic research. This methods are part of the bachelor thesis of Benjamin T. Schwertfeger.
During this thesis, many of these methods have also been implemented in C++.
This can be found here: [https://github.com/btschwertfeger/Bias-Adjustment-Cpp](https://github.com/btschwertfeger/Bias-Adjustment-Cpp).

There is also a Jupyter Notebook that serves as example.
____
## Run adjustment:
```bash
python3 do_bias_correction.py   \
    --obs input_data/obs.nc     \
    --contr input_data/contr.nc \
    --scen input_data/scen.nc   \
    --method linear_scaling     \
    --variable tas              \
    --unit '°C'                 \
    --group time.month          \
    --kind + 
```
____
## Methods implemented by Benjamin T. Schwertfeger:
|Method| `--method` parameter|
|-----|-----|
|Linear Scaling| linear_scaling|
|Variance Scaling|variance_scaling|
|Delta Method|delta_method|
|Quantile Mapping|quantile_mapping|
|Quantile Delta Mapping|quantile_delta_mapping|

## Methods adapted from [xclim](https://xclim.readthedocs.io/en/stable/sdba.html):
|Method| `--method` parameter|
|-----|-----|
|Empirical Quantile Mapping|xclim_eqm|
|Detrended Quantile Mapping|xclim_dqm|
|Quantile Delta Mapping|xclime_qdm|


____
# Notes:
- Linear and variance, as well as delta change method require `--group time.month` as argument.
- Adjustment methods that apply changes in distributional biasses (QM. QDM, DQM; EQM, ...) need the `--nquantiles` argument set to some integer.
- Data sets should have the same spatial resolutions.
- Computation in Python takes some time, so this is only for demonstration. When adjusting large datasets, its best to the C++ implementation mentioned above.