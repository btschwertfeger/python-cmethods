'''Module to to test the bias adjustment methods'''
import sys
import logging
from typing import List
import numpy as np
import xarray as xr
from sklearn.metrics import mean_squared_error

try:
    from cmethods.CMethods import CMethods
except ModuleNotFoundError:
    print('Using local module')
    sys.path.append('/Users/benjamin/repositories/awi-workspace/Bias-Adjustment-Python')
    from cmethods.CMethods import CMethods

logging.basicConfig(
    format='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO
)

def get_datasets():
    historical_time = xr.cftime_range('1971-01-01', '2000-12-31', freq='D', calendar='noleap')
    future_time = xr.cftime_range('2001-01-01', '2030-12-31', freq='D', calendar='noleap')

    def get_hist_temp_for_lat(lat: int) -> List[float]:
        '''Returns a fake time seires by latitude value'''
        return 273.15 - (
            lat * np.cos(
                2 * np.pi * historical_time.dayofyear / 365
            ) + 2 * np.random.random_sample(
                (historical_time.size,)
            ) + 273.15 + .1 * (
                historical_time - historical_time[0]
            ).days / 365
        )

    latitudes = np.arange(23,27,1)
    some_data = [get_hist_temp_for_lat(val) for val in latitudes]
    data = np.array([some_data, np.array(some_data)+1])

    def get_dataset(data, time):
        '''Returns a data set by data and time'''
        return xr.DataArray(
            data,
            dims=('lon', 'lat', 'time'),
            coords={'time': time, 'lat': latitudes, 'lon': [0,1]},
            attrs={'units': '°C'},
        ).transpose('time','lat','lon').to_dataset(name='tas')


    obsh = get_dataset(data, historical_time)
    obsp = get_dataset(data+1, historical_time)
    simh = get_dataset(data-2, historical_time)
    simp = get_dataset(data-1, future_time)
    return obsh, obsp, simh, simp


def test_linear_scaling() -> None:
    '''Tests the linear scaling method'''
    obsh, obsp, simh, simp = get_datasets()
    logging.info('Testing 1d-methods ...')
    ls_result = CMethods().linear_scaling(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        kind = '+'
    )
    assert isinstance(ls_result, xr.core.dataarray.DataArray)
    assert mean_squared_error(ls_result, obsp['tas'][:,0,0], squared=False) < mean_squared_error(simp['tas'][:,0,0], obsp['tas'][:,0,0], squared=False)
    logging.info('Linear Scaling done!')


def test_variance_scaling() -> None:
    '''Tests the variance scaling method'''
    obsh, obsp, simh, simp = get_datasets()
    vs_result = CMethods().variance_scaling(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        kind = '+'
    )
    assert isinstance(vs_result, xr.core.dataarray.DataArray)
    assert mean_squared_error(
            vs_result, obsp['tas'][:,0,0], squared=False
        ) < mean_squared_error(
            simp['tas'][:,0,0], obsp['tas'][:,0,0], squared=False
        )
    logging.info('Variance Scaling done!')

def test_delta_method() -> None:
    '''Tests the delta method'''
    obsh, obsp, simh, simp = get_datasets()
    dm_result = CMethods().delta_method(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        kind = '+'
    )
    assert isinstance(dm_result, xr.core.dataarray.DataArray)
    assert mean_squared_error(
            dm_result, obsp['tas'][:,0,0], squared=False
        ) < mean_squared_error(
            simp['tas'][:,0,0], obsp['tas'][:,0,0], squared=False
        )
    logging.info('Delta Method done!')

def test_quantile_mapping() -> None:
    '''Tests the quantile mapping method'''
    obsh, obsp, simh, simp = get_datasets()
    qm_result = CMethods().quantile_mapping(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        n_quantiles=100,
        kind='+'
    )
    assert isinstance(qm_result, xr.core.dataarray.DataArray)
    assert mean_squared_error(
            qm_result, obsp['tas'][:,0,0], squared=False
        ) < mean_squared_error(
            simp['tas'][:,0,0], obsp['tas'][:,0,0], squared=False
        )
    logging.info('Quantile Mapping done!')

def test_quantile_delta_mapping() -> None:
    '''Tests the quantile delta mapping method'''
    obsh, obsp, simh, simp = get_datasets()
    qdm_result = CMethods().quantile_delta_mapping(
        obs = obsh['tas'][:,0,0],
        simh = simh['tas'][:,0,0],
        simp = simp['tas'][:,0,0],
        n_quantiles=100,
        kind='+'
    )

    assert isinstance(qdm_result, xr.core.dataarray.DataArray)
    assert mean_squared_error(
            qdm_result, obsp['tas'][:,0,0], squared=False
        ) < mean_squared_error(
            simp['tas'][:,0,0], obsp['tas'][:,0,0], squared=False
        )
    logging.info('Quantile Delta Mapping done!')


def test_3d_sclaing_methods() -> None:
    '''Tests the scaling based methods for 3-dimentsional data sets'''
    obsh, obsp, simh, simp = get_datasets()
    for method in CMethods().SCALING_METHODS:
        logging.info(f'Testing {method} ...')
        result = CMethods().adjust_3d(
            method = method,
            obs = obsh['tas'],
            simh = simh['tas'],
            simp = simp['tas'],
            kind='+',
            goup='time.month'
        )
        assert isinstance(result, xr.core.dataarray.DataArray)
        for lat in range(len(obsh.lat)):
            for lon in range(len(obsh.lon)):
                assert mean_squared_error(
                    result[:,lat,lon], obsp['tas'][:,lat,lon], squared=False
                ) < mean_squared_error(
                    simp['tas'][:,lat,lon], obsp['tas'][:,lat,lon], squared=False
                )
        logging.info(f'3d {method} - success!')

def test_3d_distribution_methods() -> None:
    '''Tests the distribution based methods for 3-dimentsional data sets'''
    obsh, obsp, simh, simp = get_datasets()
    for method in CMethods().DISTRIBUTION_METHODS:
        logging.info(f'Testing {method} ...')
        result = CMethods().adjust_3d(
            method = method,
            obs = obsh['tas'],
            simh = simh['tas'],
            simp = simp['tas'],
            n_quantiles=100
        )
        assert isinstance(result, xr.core.dataarray.DataArray)
        for lat in range(len(obsh.lat)):
            for lon in range(len(obsh.lon)):
                assert mean_squared_error(
                    result[:,lat,lon], obsp['tas'][:,lat,lon], squared=False
                ) < mean_squared_error(
                    simp['tas'][:,lat,lon], obsp['tas'][:,lat,lon], squared=False
                )
        logging.info(f'3d {method} - success!')

def main() -> None:
    '''Main'''

    test_linear_scaling()
    test_variance_scaling()
    test_delta_method()
    test_quantile_mapping()
    test_quantile_delta_mapping()
    test_3d_sclaing_methods()
    test_3d_distribution_methods()

if __name__ == '__main__':
    main()
    logging.info('Everything passed!')
