# Bias-Adjustment-Python

[![Generic badge](https://img.shields.io/badge/license-MIT-green.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/python-3.7+-blue.svg)](https://shields.io/)
[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/btschwertfeger/Bias-Adjustment-Python)

Collection of different scale- and distribution-based bias adjustment techniques for climatic research. (see `examples.ipynb` for help)

Bias adjustment procedures in Python are very slow, so they should not be used on large data sets.
A C++ implementation that works way faster can be found here: [https://github.com/btschwertfeger/Bias-Adjustment-Cpp](https://github.com/btschwertfeger/Bias-Adjustment-Cpp).

## Available methods
- Linear Scaling
- Variance Scaling 
- Delta (Change) Method
- Quantile Mapping
- Quantile Delta Mapping
____
## Usage

### Installation
```bash
python3 -m pip install python-cmethods
```
### Import and application
```python 
from cmethods.CMethods import CMethods
cm = CMethods()

obsh = xr.open_dataset('input_data/obs.nc')
simh = xr.open_dataset('input_data/contr.nc')
simp = xr.open_dataset('input_data/scen.nc')

ls_result = cm.linear_scaling(
    method = 'quantile_delta_mapping',
    obs = obsh['tas'][:,0,0],
    simh = simh['tas'][:,0,0],
    simp = simp['tas'][:,0,0],
    kind = '+' # *
)

qdm_result = cm.adjust_2d(
    method = 'quantile_delta_mapping',
    obs = obsh['tas'],
    simh = simh['tas'],
    simp = simp['tas'],
    n_quaniles = 1000,
    kind = '+' # *
)
# * to calculate the relative rather than the absolute change, '*' can be used instead of '+' (this is prefered when adjusting precipitation)
```

____
## Examples (see repository on [GitHub](https://github.com/btschwertfeger/Bias-Adjustment-Python))

`/examples/examples.ipynb`: Notebook containing different methods and plots

`/examples/do_bias_correctino.py`: Example script for adjusting climate data
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

- Linear and variance, as well as delta change method require `--group time.month` as argument.
- Adjustment methods that apply changes in distributional biasses (QM. QDM, DQM; EQM, ...) need the `--nquantiles` argument set to some integer.
- Data sets should have the same spatial resolutions.
____
## Notes:
- Computation in Python takes some time, so this is only for demonstration. When adjusting large datasets, its best to the C++ implementation mentioned above.