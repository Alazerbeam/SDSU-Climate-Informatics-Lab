import numpy as np
import xarray as xr
import json

# retrieve dataset from netcdf file
def load_dataset(dataset_path):
    return xr.open_dataset(dataset_path, engine='netcdf4')

# find the index of a specified channel
def get_channel_index(ds, channel_name):
    return np.where(ds.channel.values == channel_name)[0][0]

# retrieve the data from a specified channel
def get_channel_data(ds, channel_name):
    return ds['forecast'].isel(channel=get_channel_index(ds, channel_name))

def limit_data(data, timesteps, x_min, x_max, y_min, y_max, stride = 1):
    return data[:timesteps, y_min:y_max:stride, x_min:x_max:stride].values

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