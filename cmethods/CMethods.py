#!/bin/python3

import multiprocessing
import xarray as xr
import numpy as np
from tqdm import tqdm

__descrption__ = 'Script to adjust climate bias in climate data'
__author__ = 'Benjamin Thomas Schwertfeger'
__copyright__ = __author__
__email__ = 'development@b-schwertfeger.de'
__link__ = 'https://b-schwertfeger.de'
__github__ = 'https://github.com/btschwertfeger/Bias-Adjustment-Python'
__description__ = '                                                                     \
    Class / Script / Methods to apply bias corrections on temperature climate data      \
                                                                                        \
    T = Temperatures ($T$)                                                              \
    X = Some climate variable ($X$)                                                     \
    h = historical                                                                      \
    p = scenario; future; predicted                                                     \
    obs = observed data ($T_{obs,h}$)                                                   \
    simh = modeled data with same time period as obs ($T_{sim,h}$)                      \
    simp = data to correct (predicted simulated data) ($T_{sim,p}$)                     \
                                                                                        \
                                                                                        \
    F = Cummulative Distribution Function                                               \
    \mu = mean                                                                          \
    \sigma = standard deviation                                                         \
    i = index                                                                           \
    _{m} = long-term monthly interval                                                   \
'

class CMethods(object):
    '''Class used to adjust timeseries climate data.'''
    
    class UnknownMethodError(Exception):
        '''Exception raised for errors if unknown method called in CMethods class.

        ----- P A R A M E T E R S -----
            method (str): Input method name which caused the error
            available_methods (str): List of available methods
        '''

        def __init__(self, method: str, available_methods: list):
            super().__init__(f'Unknown method "{method}"! Available methods: {available_methods}')
    
    CUSTOM_METHODS = [
        'linear_scaling', 'variance_scaling', 'delta_method', 
        'quantile_mapping', 'quantile_delta_mapping'
    ]
    METHODS = CUSTOM_METHODS #+ XCLIM_SDBA_METHODS
    
    def __init__(self):
        pass
    
    @classmethod
    def get_available_methods(cls) -> list:
        '''Function to return the available adjustment methods'''
        return cls.METHODS
    
    @classmethod
    def get_function(cls, method: str):
        if method == 'linear_scaling': return cls.linear_scaling        
        elif method == 'variance_scaling': return cls.variance_scaling        
        elif method == 'delta_method':  return cls.delta_method
        elif method == 'quantile_mapping': return cls.quantile_mapping
        elif method == 'empirical_quantile_mapping': return cls.empirical_quantile_mapping
        elif method == 'quantile_delta_mapping': return cls.quantile_delta_mapping
        else: raise UnknownMethodError(method, cls.METHODS)

    @classmethod
    def adjust_2d(cls,
        method: str,
        obs: xr.core.dataarray.DataArray, 
        simh: xr.core.dataarray.DataArray, 
        simp: xr.core.dataarray.DataArray,
        tslice_adjust: slice=None,
        n_quantiles: int=100, 
        kind: str='+', 
        group: str=None, 
        window: int=1,
        n_jobs: int=1,
        **kwargs
    ) -> xr.core.dataarray.Dataset:
        '''Function to adjust 3 dimensional climate data
        
        Note: obs, simh and simp has to be in the format (time, lat, lon)
        
        ----- P A R A M E T E R S -----
        
            method (str): adjustment method (see available methods by calling classmethod "get_available_methods")
            
            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data to adjust
            
            tslice_adjust (slice): Timespan to adjust, default: None
            n_quantiles (int): Number of quantiles to involve
            kind (str): Kind of adjustment ('+' or '*'), default: '+' (always use '+' for temperature)
            group (str): Group data by (e.g.: 'time.month', 'time.dayofyear')
            window (int): Grouping window, default: 1
            n_jobs (int): Use n processes, default: 1
            
        ----- R E T U R N -----
            
            xarray.core.dataarray.Dataset: Adjusted dataset
            
        ----- E X A M P L E -----
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simp = xarray.open_dataset('path/to/simulated_future/data.nc') 
            > variable = 'temperature'
            
            > adjusted_data = CMethods().adjust_2d(
                method = 'quantile_delta_mapping', 
                obs = obs[variable], 
                simh = simh[variable], 
                simp = simp[variable], 
                tslice_adjust = slice('2071-01-01', '2100-12-31'),
                n_quantiles = 100, 
                group = 'time.month', 
                n_jobs = 4
            )            
        '''
        
        obs = obs.transpose('lat', 'lon', 'time')
        simh = simh.transpose('lat', 'lon', 'time')
        simp = simp.transpose('lat', 'lon', 'time')
        
        result = simp.copy(deep=True).load()
        len_lat, len_lon = len(obs.lat), len(obs.lon)
        
        if method in cls.CUSTOM_METHODS:
            if n_jobs == 1:
                method = cls.get_function(method)
                for lat in tqdm(range(len_lat)):
                    for lon in range(len_lon):
                        result[lat,lon] = method(
                            obs = obs[lat,lon], 
                            simh = simh[lat,lon],
                            simp = simp[lat,lon],
                            group = group,
                            kind = kind,
                            n_quantiles = n_quantiles,
                            **kwargs
                        ) 
            else:
                with multiprocessing.Pool(processes=n_jobs) as pool:
                    params: [dict] = [{
                        'method': method, 
                        'obs': obs[lat], 
                        'simh': simh[lat], 
                        'simp': simp[lat], 
                        'group': group,
                        'kind': kind,
                        'n_quaniles': n_quantiles,
                        'kwargs': kwargs
                    } for lat in range(len_lat)]
                    for lat, corrected in enumerate(pool.map(cls.pool_adjust, params)):
                        result[lat] = corrected
            return result.transpose('time', 'lat', 'lon')        
        else: raise UnknownMethodError(methodm, cls.METHODS)
                
    @classmethod
    def pool_adjust(cls, params) -> xr.core.dataarray.DataArray:
        ''' Adjustment along longitude for one specific latitude
            used by cls.adjust_2d as callbackfunction for multiprocessing.Pool            
        '''        
        
        method = params['method']
        obs = params['obs']
        simh = params['simh']
        simp = params['simp']
        tslice_adjust = params.get('tslice_adjust', None) 
        n_quantiles = params.get('n_quantiles', 100)
        kind = params.get('kind', '+')
        group = params.get('group', None)
        window = params.get('window', 1)
        save_model = params.get('save_model', None)
        kwargs = params.get('kwargs', {})
        
        result = simp.copy(deep=True).load()
        len_lon = len(obs.lon)
        
        func_adjustment = None
        if method in cls.CUSTOM_METHODS:
            func_adjustment = cls.get_function(method)                
            kwargs['n_quantiles'] = n_quantiles
            kwargs['kind'] = kind
            for lon in range(len_lon):
                result[lon] = func_adjustment(
                    obs=obs[lon], simh=simh[lon], simp=simp[lon], group=group, **kwargs
                )
            return result
        
        else: raise UnknownMethodError(method, cls.METHODS)       
                
    @classmethod 
    def grouped_correction(cls,
        method: str,
        obs: xr.core.dataarray.DataArray, 
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        group: str,
        kind: str='+',
        **kwargs
    ) -> xr.core.dataarray.DataArray:    
        '''Method to adjust 1 dimensional climate data while respecting adjustment groups.
        
        ----- P A R A M E T E R S -----
    
            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            method (str): Scaling method (e.g.: 'linear_scaling')
            group (str): [optional]  Group / Period (either: 'time.month', 'time.year' or 'time.dayofyear')
            
        ----- R E T U R N -----
            
            xarray.core.dataarray.DataArray: Adjusted data 
            
        '''

        func_adjustment = cls.get_function(method)
        result = simp.copy(deep=True).load()
        groups = simh.groupby(group).groups

        for month in groups.keys():
            m_obs, m_simh, m_simp = [], [], []

            for i in groups[month]:
                m_obs.append(obs[i])
                m_simh.append(simh[i])
                m_simp.append(simp[i])           
                
            computed_result = func_adjustment(obs=m_obs, simh=m_simh, simp=m_simp, kind=kind, group=None, **kwargs)
            for i, index in enumerate(groups[month]): result[index] = computed_result[i]

        return result
                  
    # ? -----========= L I N E A R - S C A L I N G =========------ 
    @classmethod
    def linear_scaling(cls,
        obs: xr.core.dataarray.DataArray, 
        simh: xr.core.dataarray.DataArray,
        simp: xr.core.dataarray.DataArray,
        group: str='time.month',
        kind: str='+',
        **kwargs
    ) -> xr.core.dataarray.DataArray:
        '''Method to adjust 1 dimensional climate data by the linear scaling method.
        
        ----- P A R A M E T E R S -----
    
            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            group (str): [optional] Group / Period (e.g.: 'time.month')
            method (str): '+' or '*', default: '+'
            
        ----- R E T U R N -----
            
            xarray.core.dataarray.DataArray: Adjusted data 
            
        ----- E X A M P L E -----
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > simp = xarray.open_dataset('path/to/predicted/data.nc')
            > variable = 'temperature'
            
            > result = CMethods().linear_scaling(
            >    obs=obs[variable], 
            >    simh=simh[variable], 
            >    simp=simp[variable],
            >    group='time.month'
            >)  
            
        ----- E Q U A T I O N S -----
            T = temperature; d = daily; m = monthly 
            Add ('+'):
            (1.)    X^{*LS}_{sim,p}(i) = X_{sim,p}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))
            Mult ('*'):
            (2.)    X^{*LS}_{sim,h}(i) = X_{sim,h}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))
        
        ----- R E F E R E N C E S -----

            Based on the equations of Teutschbein, Claudia and Seibert, Jan (2012) Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods
            https://doi.org/10.1016/j.jhydrol.2012.05.052
            
        '''       
        
        if group != None: return cls.grouped_correction(method='linear_scaling', obs=obs, simh=simh, simp=simp, group=group, kind=kind, **kwargs)
        else:
            if kind == '+': return np.array(simp) + (np.nanmean(obs) - np.nanmean(simh)) # Eq. 1
            elif kind == '*': return np.array(simp) * (np.nanmean(obs) / np.nanmean(simh)) # Eq. 2
            else: raise ValueError('Scaling type invalid. Valid options for param kind: "+" and "*"')

    # ? -----========= V A R I A N C E - S C A L I N G =========------ 
    @classmethod
    def variance_scaling(cls,
        obs: xr.core.dataarray.DataArray, 
        simh: xr.core.dataarray.DataArray, 
        simp: xr.core.dataarray.DataArray, 
        group: str='time.month',
        kind: str='+',
        **kwargs
    ) -> xr.core.dataarray.DataArray:
        '''Method to adjust 1 dimensional climate data by variance scaling method.
        
        ----- P A R A M E T E R S -----
    
            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            group (str): [optional] Group / Period (e.g.: 'time.month')
            
        ----- R E T U R N -----
            
            xarray.core.dataarray.DataArray: Adjusted data 
            
        ----- E X A M P L E -----
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > simp = xarray.open_dataset('path/to/predicted/data.nc')
            > variable = 'temperature'
            
            > result = CMethods().variance_scaling(obs=obs[variable], simh=simh[variable], simp=simp[variable] group='time.dayofyear')   
            
        ------ E Q U A T I O N S -----
        
            T = temperature; d = daily; m = monthly 

            (1.) X^{*LS}_{sim,h}(i) = X_{sim,h}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))
            (2.) X^{*LS}_{sim,p}(i) = X_{sim,p}(i) + \mu_{m}(X_{obs,h}(i)) - \mu_{m}(X_{sim,h}(i))

            (3.) X^{VS(1)}_{sim,h}(i) = X^{*LS}_{sim,h}(i) - \mu_{m}(X^{*LS}_{sim,h}(i))
            (4.) X^{VS(1)}_{sim,p}(i) = X^{*LS}_{sim,p}(i) - \mu_{m}(X^{*LS}_{sim,p}(i))

            (5.) X^{VS(2)}_{sim,p}(i) = X^{VS(1)}_{sim,p}(i) \cdot \left[\frac{\sigma_{m}(X_{obs,h}(i))}{\sigma_{m}(X^{VS(1)}_{sim,h}(i))}\right]

            (6.) X^{*VS}_{sim,p}(i) = X^{VS(2)}_{sim,p}(i) + \mu_{m}(X^{*LS}_{sim,p}(i))
        
        ----- R E F E R E N C E S -----

            Based on the equations of Teutschbein, Claudia and Seibert, Jan (2012) Bias correction of regional climate model simulations for hydrological climate-change impact studies: Review and evaluation of different methods
            https://doi.org/10.1016/j.jhydrol.2012.05.052
        '''
        if group != None: return cls.grouped_correction(method='variance_scaling', obs=obs, simh=simh, simp=simp, group=group, kind=kind, **kwargs)
        else:
            LS_simh = cls.linear_scaling(obs, simh, simh)               # Eq. 1
            LS_simp = cls.linear_scaling(obs, simh, simp)               # Eq. 2

            VS_1_simh = LS_simh - np.nanmean(LS_simh)                   # Eq. 3 
            VS_1_simp = LS_simp - np.nanmean(LS_simp)                   # Eq. 4

            VS_2_simp = VS_1_simp * (np.std(obs) / np.std(VS_1_simh))   # Eq. 5

            return VS_2_simp + np.nanmean(LS_simp)                      # Eq. 6
            
    # ? -----========= D E L T A - M E T H O D =========------ 
    @classmethod
    def delta_method(cls,
        obs: xr.core.dataarray.DataArray, 
        simh: xr.core.dataarray.DataArray, 
        simp: xr.core.dataarray.DataArray, 
        group: str='time.month',
        kind: str='+',
        **kwargs
    ) -> xr.core.dataarray.DataArray: 
        '''Method to adjust 1 dimensional climate data by delta method.
        
        ----- P A R A M E T E R S -----
    
            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            group (str): [optional] Group / Period (e.g.: 'time.month')
            
        ----- R E T U R N -----
            
            xarray.core.dataarray.DataArray: Adjusted data 
            
        ----- E X A M P L E -----
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > simp = xarray.open_dataset('path/to/predicted/data.nc')
            > variable = 'temperature'
            
            > result = CMethods().delta_method(obs=obs[variable], simh=simh[variable], group='time.month')   
            
        ------ E Q U A T I O N S -----

            Add (+):
                (1.) X^{*DM}_{sim,p}(i) = X_{obs,h}(i) + (\mu_{m}(X_{sim,p}(i)) - \mu_{m}(X_{sim,h}(i)))
            Mult (*):
                (2.) X^{*DM}_{sim,p}(i) = X_{obs,h}(i) \cdot \frac{ \mu_{m}(X_{sim,p}(i)) }{ \mu_{m}(X_{sim,h}(i))}
        
        ----- R E F E R E N C E S -----
            https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py
        
        '''
        if group != None: return cls.grouped_correction(method='delta_method', obs=obs, simh=simh, simp=simp, group=group, kind=kind, **kwargs)
        else: 
            if kind == '+': return np.array(obs) + (np.nanmean(simp) - np.nanmean(simh))     # Eq. 1
            elif kind == '*': return np.array(obs) * (np.nanmean(simp) / np.nanmean(simh))   # Eq. 2
            else: raise ValueError(f'{kind} not implemented! Use "+" or "*" instead.')

    
    # ? -----========= Q U A N T I L E - M A P P I N G =========------ 
    @classmethod
    def quantile_mapping(cls,
        obs: xr.core.dataarray.DataArray, 
        simh: xr.core.dataarray.DataArray, 
        simp: xr.core.dataarray.DataArray, 
        n_quantiles: int,
        group: str=None,
        kind: str='+',
        **kwargs
    ) -> xr.core.dataarray.DataArray:
        ''' Quantile Mapping Bias Correction
                
                ------ E Q U A T I O N S -----
                Add (+):
                    (1.) X^{*QM}_{sim,p}(i) = F^{-1}_{obs,h} \left\{F_{sim,h}\left[X_{sim,p}(i)\right]\right\}
                Mult (*):
                    (1.) --//--

                ----- R E F E R E N C E S -----
                
                Multiplicative implementeation by 'deleted profile' OR Adrian Tompkins tompkins@ictp.it, posted on November 8, 2016 at 
                https://www.researchgate.net/post/Does-anyone-know-about-bias-correction-and-quantile-mapping-in-PYTHON
        '''

        if group != None: return cls.grouped_correction(method='quantile_mapping', obs=obs, simh=simh, simp=simp, group=group, n_quantiles=n_quantiles, kind=kind, **kwargs)
        elif kind == '+':
            # create np.array to achieve higher accuracy (idk why)
            obs, simh, simp = np.array(obs), np.array(simh), np.array(simp)
            
            global_max = max(np.amax(obs), np.amax(simh))
            global_min = min(np.amin(obs), np.amin(simh)) 
            wide = abs(global_max - global_min) / n_quantiles 
            xbins = np.arange(global_min, global_max + wide, wide)
            
            cdf_obs = cls.get_cdf(obs, xbins)
            cdf_simh = cls.get_cdf(simh, xbins)
            epsilon = np.interp(simp, xbins, cdf_simh)                                 # Eq. 1            
            return cls.get_inverse_of_cdf(cdf_obs, epsilon, xbins)                     # Eq. 1

        elif kind == '*': 
            ''' Inspired by Adrian Tompkins tompkins@ictp.it posted here: 
                https://www.researchgate.net/post/Does-anyone-know-about-bias-correction-and-quantile-mapping-in-PYTHON
            '''

            obs, simh, simp = np.sort(obs), np.sort(simh), np.array(simp)
            global_max = max(np.amax(obs), np.amax(simh))
            wide = global_max / n_quantiles
            xbins = np.arange(0.0, global_max + wide, wide)
            
            pdf_obs, bins = np.histogram(obs, bins=xbins)
            pdf_simh, bins = np.histogram(simh, bins=xbins)
            
            cdf_obs = np.insert(np.cumsum(pdf_obs), 0, 0.0)
            cdf_simh = np.insert(np.cumsum(pdf_simh), 0, 0.0)
            
            epsilon = np.interp(simp, xbins, cdf_simh, left=0.0, right=999.0)
            return np.interp(epsilon, cdf_obs, xbins, left=0.0, right=-999.0)

        else: raise ValueError('Not implemented!')

    # ? -----========= E M P I R I C A L - Q U A N T I L E - M A P P I N G =========------ 
    @classmethod
    def empirical_quantile_mapping(cls,
        obs: xr.core.dataarray.DataArray, 
        simh: xr.core.dataarray.DataArray, 
        simp: xr.core.dataarray.DataArray, 
        n_quantiles: int=10,
        extrapolate: str=None,
        group: str=None,
        **kwargs
    ) -> xr.core.dataarray.DataArray:
        ''' Method to adjust 1 dimensional climate data by empirical quantile mapping
        
        ----- P A R A M E T E R S -----
    
            obs (xarray.core.dataarray.DataArray): observed / obserence Data
            simh (xarray.core.dataarray.DataArray): simulated historical Data
            simp (xarray.core.dataarray.DataArray): future simulated Data
            n_quantiles (int): Quantiles to involve, default: 10
            extrapolate (str): [optional] 'linear' or 'constant', default: None
            group (str): [optional] Group / Period (e.g.: 'time.month')
            
        ----- R E T U R N -----
            
            xarray.core.dataarray.DataArray: Adjusted data 
            
        ----- E X A M P L E -----
        
            > obs = xarray.open_dataset('path/to/observed/data.nc')
            > simh = xarray.open_dataset('path/to/simulated/data.nc')
            > simp = xarray.open_dataset('path/to/future/data.nc')
            > variable = 'temperature'
            > result = CMethods().empirical_quantile_mapping(
            >    obs=obs[variable], 
            >    simh=simh[variable], 
            >    simp=simp[variable], 
            >    group='time.dayofyear'
            >)      
            
        ----- R E F E R E N C E S -----
        
            (April 2022) Taken from:
            https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py
            
        '''
        raise ValueError('idk if it is allowed to use this so please have a look at: https://svn.oss.deltares.nl/repos/openearthtools/trunk/python/applications/hydrotools/hydrotools/statistics/bias_correction.py ')
        # if group != None:
        #     return cls.grouped_correction(
        #         method = 'empirical_quantile_mapping', 
        #         obs = obs, 
        #         simh = simh, 
        #         simp = simp,
        #         group = group,
        #         n_quantiles = n_quantiles,
        #         extrapolate = extrapolate
        #     )
        # else:
        #     obs, simh, simp = np.array(obs), np.array(simh), np.array(simp)
        #     return corrected
    
    # ? -----========= Q U A N T I L E - D E L T A - M A P P I N G =========------ 
    @classmethod
    def quantile_delta_mapping(cls,
        obs: xr.core.dataarray.DataArray, 
        simh: xr.core.dataarray.DataArray, 
        simp: xr.core.dataarray.DataArray, 
        n_quantiles: int,
        group: str=None,
        kind: str='+',
        **kwargs
    ) -> xr.core.dataarray.DataArray:
        ''' Quantile Delta Mapping bias adjustment
               
            ------ E Q U A T I O N S -----
            
            Add (+):
                (1.1) \varepsilon(i) = F_{sim,p}\left[X_{sim,p}(i)\right], \hspace{1em} \varepsilon(i)\in\{0,1\}

                (1.2) X^{QDM(1)}_{sim,p}(i) = F^{-1}_{obs,h}\left[\varepsilon(i)\right]

                (1.3) \Delta(i) & = F^{-1}_{sim,p}\left[\varepsilon(i)\right] - F^{-1}_{sim,h}\left[\varepsilon(i)\right] \\[1pt]
                              & = X_{sim,p}(i) - F^{-1}_{sim,h}\left\{F^{}_{sim,p}\left[X_{sim,p}(i)\right]\right\}

                (1.4) X^{*QDM}_{sim,p}(i) = X^{QDM(1)}_{sim,p}(i) + \Delta(i)
            
            Mult (*):
                (1.1) --//--

                (1.2) --//--

                (2.3) \Delta(i) & = \frac{ F^{-1}_{sim,p}\left[\varepsilon(i)\right] }{ F^{-1}_{sim,h}\left[\varepsilon(i)\right] } \\[1pt]
                              & = \frac{ X_{sim,p}(i) }{ F^{-1}_{sim,h}\left\{F^{}_{sim,p}\left[X_{sim,p}(i)\right]\right\} }

                (2.4) X^{*QDM}_{sim,p}(i) = X^{QDM(1)}_{sim,p}(i) \cdot \Delta(i)

            ----- R E F E R E N C E S -----
                Tong, Y., Gao, X., Han, Z. et al. Bias correction of temperature and precipitation over China for RCM simulations using the QM and QDM methods. Clim Dyn 57, 1425–1443 (2021). 
                https://doi.org/10.1007/s00382-020-05447-4

            ----- N O T E S -----

            @param global_min: float | this parameter can be set when kind == '*' to define a custom lower limit. Otherwise 0.0 is used.
            
        '''
        
        if group != None: return cls.grouped_correction(method='quantile_delta_mapping', obs=obs, simh=simh, simp=simp, group=group, n_quantiles=n_quantiles, kind=kind, **kwargs)
        elif kind == '+':
            obs, simh, simp = np.array(obs), np.array(simh), np.array(simp) # to achieve higher accuracy
            global_max = max(np.amax(obs), np.amax(simh))
            global_min = min(np.amin(obs), np.amin(simh))
            wide = abs(global_max - global_min) / n_quantiles    
            xbins = np.arange(global_min, global_max + wide, wide)

            cdf_obs = cls.get_cdf(obs,xbins) 
            cdf_simh = cls.get_cdf(simh,xbins)
            cdf_simp = cls.get_cdf(simp, xbins)                                 

            # calculate exact cdf values of $F_{sim,p}[T_{sim,p}(t)]$
            epsilon = np.interp(simp, xbins, cdf_simp)                            # Eq. 1.1            
            QDM1 = cls.get_inverse_of_cdf(cdf_obs, epsilon, xbins)                # Eq. 1.2
            delta = simp - cls.get_inverse_of_cdf(cdf_simh, epsilon, xbins)       # Eq. 1.3
            return QDM1 + delta                                                   # Eq. 1.4

        elif kind == '*': 
            obs, simh, simp = np.array(obs), np.array(simh), np.array(simp) 
            global_max = max(np.amax(obs), np.amax(simh))
            wide = global_max / n_quantiles    
            xbins = np.arange(kwargs.get('global_min', 0.0), global_max + wide, wide)

            cdf_obs = cls.get_cdf(obs,xbins) 
            cdf_simh = cls.get_cdf(simh,xbins)
            cdf_simp = cls.get_cdf(simp, xbins)           

            epsilon = np.interp(simp, xbins, cdf_simp)                            # Eq. 1.1
            QDM1 = cls.get_inverse_of_cdf(cdf_obs, epsilon, xbins)                # Eq. 1.2
            delta = simp / cls.get_inverse_of_cdf(cdf_simh, epsilon, xbins)       # Eq. 2.3
            return QDM1 * delta                                                   # Eq. 2.4
            
        else: raise ValueError(f'Unknown kind {kind}!')
            
    # * -----========= G E N E R A L  =========------ 
    @staticmethod
    def get_pdf(a, xbins: list) -> np.array:
        ''' returns the probability density function of a based on xbins ($P(x)$)'''
        pdf, _ = np.histogram(a,xbins)
        return pdf 

    @staticmethod
    def get_cdf(a, xbins: list) -> np.array:
        ''' returns the cummulative distribution function of a based on xbins ($F_{a}$)'''
        pdf, _ = np.histogram(a, xbins)
        return np.insert(np.cumsum(pdf), 0, 0.0)
    
    @staticmethod    
    def get_inverse_of_cdf(base_cdf, insert_cdf, xbins) -> np.array:
        ''' returns the inverse cummulative distribution function of base_cdf ($F_{base_cdf}\left[insert_cdf\right])$'''
        return np.interp(insert_cdf, base_cdf, xbins)

    @staticmethod
    def load_data(
        obs_fpath: str, 
        contr_fpath: str, 
        scen_fpath: str,
        use_cftime: bool=False, 
        chunks=None
    ) -> (xr.core.dataarray.Dataset, xr.core.dataarray.Dataset, xr.core.dataarray.Dataset):
        '''Load and return loaded netcdf datasets'''
        obs = xr.open_dataset(obs_fpath, use_cftime=use_cftime, chunks=chunks)
        simh = xr.open_dataset(contr_fpath, use_cftime=use_cftime, chunks=chunks)
        simp = xr.open_dataset(scen_fpath, use_cftime=use_cftime, chunks=chunks)
        
        return obs, simh, simp