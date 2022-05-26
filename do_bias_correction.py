#!/bin/python3

__descrption__ = 'Script to adjust climate biases in 3D Climate data'
__author__ = 'Benjamin Thomas Schwertfeger'
__copyright__ = __author__
__email__ = 'development@b-schwertfeger.de'
__link__ = 'https://b-schwertfeger.de'
__github__ = 'https://github.com/btschwertfeger/Bias-Adjustment-Python';

import argparse
import logging, sys
import xarray as xr

from CMethods import CMethods

# * ----- L O G G I N G -----
formatter = logging.Formatter(
    fmt='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' 
)

log = logging.getLogger()
log.setLevel(logging.INFO)
screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
logging.getLogger().addHandler(screen_handler)

# * ----- I N P U T - H A N D L I N G -----
parser = argparse.ArgumentParser(description='Adjust climate data based on bias correction algorithms and magic.')
parser.add_argument('--obs', '--observation', dest='obs_fpath', type=str, help='Observation dataset')
parser.add_argument('--contr', '--control', dest='contr_fpath', type=str, help='Control dataset')
parser.add_argument('--scen', '--scenario', dest='scen_fpath', type=str, help='Scenario dataset (data to adjust)')

parser.add_argument('-m', '--method', dest='method', type=str, help='Correction method')
parser.add_argument('-v', '--variable', dest='var', type=str, default='tas', help='Variable to adjust')
parser.add_argument('-u', '--unit', dest='unit', type=str, default='°C', help='Unit of the varible')

parser.add_argument('-g', '--group', dest='group', type=str, default=None, help='Value grouping, default: time, (options: time.month, time.dayofyear, time.year')
parser.add_argument('-k', '--kind', dest='kind', type=str, default='+', help='+ or *, default: +')
parser.add_argument('-w', '--window', dest='window', type=int, default=1, help='Window of grouping')
parser.add_argument('-n', '--nquantiles', dest='n_quantiles', type=int, default=100, help='Nr. of Quantiles to use')

parser.add_argument('-p', '--processes', dest='p', type=int, default=1, help='Multiprocessing with n processes, default: 1')
params = vars(parser.parse_args())

obs_fpath = params['obs_fpath']
contr_fpath = params['contr_fpath']
scen_fpath = params['scen_fpath']

method = params['method']
var = params['var']
unit = params['unit']
group = params['group']
kind = params['kind']
window = params['window']
n_quantiles = params['n_quantiles']
n_jobs = params['p']

# * ----- ----- -----M A I N ----- ----- -----
def main() -> None:
    cm = CMethods()

    if method not in cm.get_available_methods(): raise ValueError(f'Unknown method {method}. Available methods: {cm.get_available_methods()}')

    ds_obs = xr.open_dataset(obs_fpath)[var]
    ds_simh = xr.open_dataset(contr_fpath)[var]
    ds_simp = xr.open_dataset(scen_fpath)[var]
    log.info('Data Loaded')

    ds_obs.attrs['unit'] = unit
    ds_simh.attrs['unit'] = unit
    ds_simp.attrs['unit'] = unit

    start_date: str = ds_simp['time'][0].dt.strftime('%Y%m%d').values.ravel()[0]
    end_date: str = ds_simp['time'][-1].dt.strftime('%Y%m%d').values.ravel()[0]

    descr1, descr2 = '', ''
    if method in cm.XCLIM_SDBA_METHODS or method in ['quantile_mapping', 'quantile_delta_mapping']:
        descr1 = f'_quantiles-{n_quantiles}'
        descr2 = f'_window-{window}'

    # ----- Adjustment -----
    log.info(f'Starting {method} adjustment')
    result = cm.adjust_2d(
        method = method,
        obs = ds_obs,
        simh = ds_simh,
        simp = ds_simp,
        n_quantiles = n_quantiles,
        kind = kind,
        group = group,
        window = window,
        n_jobs = n_jobs
    )
    log.info('Saving now')
    result.name = var
    result['time'] = ds_simp['time']
    result.to_netcdf(f'{method}_result_var-{var}{descr1}_kind-{kind}_group-{group}{descr2}_{start_date}_{end_date}.nc')
    log.info('Done')


if __name__ == '__main__':
    main()


# * ----- ----- E O F ----- -----