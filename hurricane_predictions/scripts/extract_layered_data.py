import sys
import os
import numpy as np
from noisy_hurricane import run_inference
from load_data import *
from config import *
from track_hurricane import index_to_lat, index_to_lon

json_datapath = "hurricane_predictions/data/layered_data.json"
layered_data = {}

def array_to_time_lat_lon_dict(arr):
    """
    Convert (time, lat, lon) NumPy array to nested dict:
    time_index -> lat_value -> lon_value -> float
    """
    out = {}
    timesteps, n_lat, n_lon = arr.shape

    for t in range(timesteps):
        lat_dict = {}
        for i_lat in range(n_lat):
            lat_val = index_to_lat(i_lat + y_min)
            lon_dict = {}
            for i_lon in range(n_lon):
                lon_val = index_to_lon(i_lon + x_min)
                lon_dict[lon_val] = float(arr[t, i_lat, i_lon])
            lat_dict[lat_val] = lon_dict
        out[t] = lat_dict

    return out

error_log = load_json(ERROR_LOG_PATH)
median_seeds = {}
for noise_pct, seed2error in error_log.items():
    total_errors = [(seed, errors[-1]) for seed, errors in seed2error.items()]
    total_errors.sort(key = lambda x: x[1])
    median_seeds[noise_pct] = total_errors[len(total_errors) // 2][0]

noise_levels = [str(noise_pct) for noise_pct in NOISE_PCTS] + ['real']
print(noise_levels)
for noise_pct in noise_levels:
    if noise_pct not in layered_data:
        layered_data[noise_pct] = {}

    if noise_pct != 'real':
        seed = median_seeds[float(noise_pct)]
        run_inference(MODEL_DIR, TRUE_PATH, PRED_PATH, ERA5_STATS_DIR, TIMESTEPS, float(noise_pct), seed, verbose=False)
        ds = load_dataset(PRED_PATH)
    else:
        ds = load_dataset(TRUE_PATH)
    
    for pressure_level in ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']:
        if pressure_level not in layered_data[noise_pct]:
            layered_data[noise_pct][pressure_level] = {}
        
        zonal_wind = limit_data(get_channel_data(ds, 'u'+pressure_level), TIMESTEPS, x_min, x_max, y_min, y_max)
        meridional_wind = limit_data(get_channel_data(ds, 'v'+pressure_level), TIMESTEPS, x_min, x_max, y_min, y_max)
        temperature = limit_data(get_channel_data(ds, 't'+pressure_level), TIMESTEPS, x_min, x_max, y_min, y_max)

        wind_speeds = np.sqrt(zonal_wind**2. + meridional_wind**2.)

        layered_data[noise_pct][pressure_level]['temperature'] = array_to_time_lat_lon_dict(temperature)
        layered_data[noise_pct][pressure_level]['wind speed'] = array_to_time_lat_lon_dict(wind_speeds)
    ds.close()

update_json(json_datapath, layered_data)
        
        
        

