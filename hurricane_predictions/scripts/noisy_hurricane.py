import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import os
import argparse
from earth2mip.networks import get_model

def randomize_seeds(num_seeds):
    seed_list = np.arange(num_seeds)
    np.random.shuffle(seed_list)
    seed_list = [int(seed) for seed in seed_list]
    return seed_list

# set all random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# load the means and stds of era5 dataset for all channels
def load_era5_stats(era5_stats_dir):
    era5_means = np.load(os.path.join(era5_stats_dir, "global_means.npy"))
    era5_stds = np.load(os.path.join(era5_stats_dir, "global_stds.npy"))
    return era5_means, era5_stds

# load model, run inference, save forecast to NetCDF file
def run_inference(
    model_path: str, input_path: str, output_path: str, era5_stats_dir: str, steps: int, noise_prop: float, seed: float = None, verbose: bool = True
    ):
    # set seed for reproducibility
    if seed is None: 
        set_seed(42)
    else:
        set_seed(seed)
    
    # use gpu is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # save path to modified files
    pkg_dir = Path(model_path).resolve()
    
    # load the model
    model = get_model(f"file://{pkg_dir}")
    model = model.to(device)  # move to gpu if available
    
    if verbose:
        print("Successfully loaded model!")

    data = xr.open_dataset(input_path, engine='netcdf4')['forecast'].values
    init_conds = data[0].reshape((1, 1, 73, 721, 1440))
    init_conds = torch.from_numpy(init_conds).float()
    
    _, era5_stds = load_era5_stats(era5_stats_dir)
    stds = era5_stds[0, :, 0, 0].reshape(1, 1, 73, 1, 1)
    base_noise = torch.randn_like(init_conds) # N(0,1)
    init_conds_noise = init_conds + base_noise * noise_prop * stds
    init_conds_noise = init_conds_noise.to(device)
    
    if verbose:
        print("Successfully generated random initial conditions!")
    
    # run inference
    time = datetime(2018, 9, 13, 0, 0)
    predictions = []
    times = []
    
    iterator = model(time, init_conds_noise)
    for _ in tqdm(range(steps), desc="Generating forecast"):
        temp_time, temp_output, _ = next(iterator)
        predictions.append(temp_output[0].cpu().numpy())
        times.append(temp_time)

    if verbose:
        print("Successfully ran inference!")
    
    predictions = np.stack(predictions)     # shape: (steps, channels, lat, lon)

    # convert to xr.Dataset
    data_arr = xr.DataArray(
        predictions,
        dims = ["time", "channel", "lat", "lon"],
        coords = {
            "time": times,
            "channel": model.out_channel_names,
            "lat": model.grid.lat,
            "lon": model.grid.lon
        },
        name = "forecast"
    )
    ds = data_arr.to_dataset()
    
    if verbose:
        print("Successfully converted to xr.Dataset!")
    
    # save
    ds.to_netcdf(output_path)

    if verbose:
        print(f"Saved forecast to {output_path}!")

# run_inference(
#     model_path = args.model_dir,
#     input_path = args.input,
#     output_path = args.output,
#     era5_stats_dir = args.era5_stats_dir,
#     steps = args.timesteps,
#     verbose = True,
#     noise_prop = args.noise
# )
