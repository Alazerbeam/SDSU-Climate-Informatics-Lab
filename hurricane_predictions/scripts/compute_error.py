import numpy as np
import xarray as xr
from geopy.distance import geodesic
from load_data import *
from track_hurricane import *
from config import *

# find dist between true and pred at each timestep
def compute_per_timestep_error(true_lats, true_lons, pred_lats, pred_lons):
    return [
        geodesic((true_lat, true_lon), (pred_lat, pred_lon)).km
        for true_lat, true_lon, pred_lat, pred_lon in zip(true_lats, true_lons, pred_lats, pred_lons)
    ]

# find total error between true and pred at each timestep
def compute_cumulative_error(true_lats, true_lons, pred_lats, pred_lons):
    return np.cumsum(compute_per_timestep_error(true_lats, true_lons, pred_lats, pred_lons))

# update error_log
def update_error_log(error_log, noise_pct, seed, true_datapath, pred_datapath, timesteps):
    ds_true = load_dataset(true_datapath)
    ds_pred = load_dataset(pred_datapath)
    
    pred_msl_data = get_channel_data(ds_pred, 'msl')
    msl_pred_limited = limit_data(pred_msl_data, timesteps, x_min, x_max, y_min, y_max)
    true_msl_data = get_channel_data(ds_true, 'msl')
    msl_true_limited = limit_data(true_msl_data, timesteps, x_min, x_max, y_min, y_max)

    track_true_x, track_true_y, track_pred_x, track_pred_y = track_true_and_pred_locs(msl_true_limited, msl_pred_limited)

    # convert indices of min msl positions to lat and lon
    track_true_y = index_to_lat_vec(np.array(track_true_y) + y_min)
    track_true_x = index_to_lon_vec(np.array(track_true_x) + x_min)
    track_pred_y = index_to_lat_vec(np.array(track_pred_y) + y_min)
    track_pred_x = index_to_lon_vec(np.array(track_pred_x) + x_min)
    
    if noise_pct not in error_log:
        error_log[noise_pct] = {}
    if seed not in error_log[noise_pct]:
        error_log[noise_pct][seed] = compute_cumulative_error(track_true_y, track_true_x, track_pred_y, track_pred_x).tolist()

# given noise level, calculates mean and std over all seeds per timestep (for hurricane trajectory cumulative distance error)
def compute_error_distribution(error_log, noise_pct):
    all_errors = np.array(list(error_log[noise_pct].values()))
    mean_error = np.mean(all_errors, axis=0)
    std_error = np.std(all_errors, axis=0)
    timesteps = np.arange(len(mean_error))
    return timesteps, mean_error, std_error

# creates netcdf file of pred - true error globally
def generate_global_error_dataset(error_datapath, true_datapath, pred_datapath, timesteps=15):
    ds_true = load_dataset(true_datapath)
    true_vals = ds_true['forecast'].values[:timesteps]
    ds_pred = load_dataset(pred_datapath)
    pred_vals = ds_pred['forecast'].values[:timesteps]
    error_vals = pred_vals - true_vals
    dims = ['time', 'channel', 'lat', 'lon']
    error_xr = xr.DataArray(
        error_vals,
        dims = dims,
        coords = {k: ds_pred.coords[k] for k in dims},
        name = "forecast"
    )
    error_ds = error_xr.to_dataset()
    error_ds.to_netcdf(error_datapath)
    error_ds.close()
