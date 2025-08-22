import numpy as np
import xarray as xr
import json
import os

# retrieve dataset from netcdf file
def load_dataset(dataset_path):
    return xr.open_dataset(dataset_path, engine='netcdf4')

# find the index of a specified channel
def get_channel_index(ds, channel_name):
    return np.where(ds.channel.values == channel_name)[0][0]

# retrieve the data from a specified channel
def get_channel_data(ds, channel_name):
    return ds['forecast'].isel(channel=get_channel_index(ds, channel_name))

def get_all_data_values(data_path):
    return load_dataset(data_path)['forecast'].values

def limit_data(data, timesteps, x_min, x_max, y_min, y_max, stride = 1):
    return data[:timesteps, y_min:y_max:stride, x_min:x_max:stride].values

# load the means and stds of era5 dataset for all channels
def load_era5_stats(era5_stats_dir):
    era5_means = np.load(os.path.join(era5_stats_dir, "global_means.npy"))
    era5_stds = np.load(os.path.join(era5_stats_dir, "global_stds.npy"))
    return era5_means, era5_stds

# retrieve the mean and std from era5 for a specific channel
def get_era5_channel_stats(ds, channel_name, era5_stats_dir):
    channel_idx = get_channel_index(ds, channel_name)
    era5_means, era5_stds = load_era5_stats(era5_stats_dir)
    return era5_means[0,channel_idx,0,0], era5_stds[0,channel_idx,0,0]

def load_json(json_path):
    try:
        with open(json_path, "r") as f:
            temp = json.load(f)
        error_log = {
            float(noise_pct): {
                int(seed): errors for seed, errors in seed_dict.items()
            } for noise_pct, seed_dict in temp.items()
        }
    except (FileNotFoundError, json.JSONDecodeError):
        error_log = {}

    return error_log

def update_json(json_path, error_log):
    with open(json_path, "w") as f:
        json.dump(error_log, f, indent=2)