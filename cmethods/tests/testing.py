import numpy as np
import xarray as xr
import random

import logging

import sys
sys.path.append('../../')
from cmethods.CMethods import CMethods

formatter = logging.Formatter(
        fmt='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S' # %I:%M:%S %p AM|PM format
    )

logging.getLogger().setLevel(logging.INFO)
screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
logging.getLogger().addHandler(screen_handler)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

def main() -> None:
    np.random.seed(0)
    random.seed(0)

    logging.info('Prepare data sets.')
    historical_time = xr.cftime_range('1971-01-01', '2000-12-31', freq='D', calendar='noleap')
    future_time = xr.cftime_range('2001-01-01', '2030-12-31', freq='D', calendar='noleap')

    get_hist_temp_for_lat = lambda val: 273.15 - (val * np.cos(2 * np.pi * historical_time.dayofyear / 365) + 2 * np.random.random_sample((historical_time.size,)) + 273.15 + .1 * (historical_time - historical_time[0]).days / 365)
    get_rand = lambda: np.random.rand() if np.random.rand() > .5 else  -np.random.rand()
    latitudes = np.arange(23,27,1)
    some_data = [get_hist_temp_for_lat(val) for val in latitudes]
    data = np.array([some_data, np.array(some_data)+1])

    def get_dataset(data) -> xr.Dataset:
        return xr.DataArray(
            data,
            dims=('lon', 'lat', 'time'),
            coords={'time': historical_time, 'lat': latitudes, 'lon': [0,1]},
            attrs={'units': '°C'},
        ).transpose('time','lat','lon').to_dataset(name='tas')


    obsh = get_dataset(data)
    simh = get_dataset(data-2)
    simp = get_dataset(data-1)

    cm = CMethods()

    logging.info('Testing 1d-methods ...')
    assert type(cm.linear_scaling(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        kind = '+'
    )) == xr.DataArray, 'Invalid return type!'
    logging.info('Linear Scaling done!')
    assert type(cm.variance_scaling(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        kind = '+'
    )) == xr.DataArray, 'Invalid return type!'
    logging.info('Variance Scaling done!')

    assert type(cm.delta_method(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        kind = '+'
    )) == xr.DataArray, 'Invalid return type!'
    logging.info('Delta Method done!')

    assert type(cm.quantile_mapping(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        n_quantiles=100,
        kind='+'
    )) == xr.DataArray, 'Invalid return type!'

    assert type(cm.quantile_delta_mapping(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
         n_quantiles=100,
         kind='+'
    )) == xr.DataArray, 'Invalid return type!'
    logging.info('Quantile Delta Mapping done!')


    logging.info('Testing 3d-methods')

    for method in cm.SCALING_METHODS:
        logging.info(f'Testing {method} ...') 
        assert type(cm.adjust_3d(
            method = method, 
            obs = obsh['tas'], 
            simh = simh['tas'], 
            simp = simp['tas'],
            kind='+',
            goup='time.month'
        )) == xr.DataArray
        logging.info(f'{method} - success!') 

    for method in cm.DISTRIBUTION_METHODS:
        logging.info(f'Testing {method} ...') 
        assert type(cm.adjust_3d(
            method = method, 
            obs = obsh['tas'], 
            simh = simh['tas'], 
            simp = simp['tas'],
            n_quantiles=100
        )) == xr.DataArray
        logging.info(f'{method} - success!') 
    


if __name__ == '__main__':
    main()
    logging.info('Everything passed!')