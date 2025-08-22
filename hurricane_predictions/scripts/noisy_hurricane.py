import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path
import xarray as xr
from tqdm import tqdm
import os
from earth2mip.networks import get_model
from load_data import load_era5_stats, load_dataset, get_all_data_values
from utils import set_seed

# load model, run inference, save forecast to NetCDF file
def run_inference(
    model_path: str, data_path: str, output_path: str, era5_stats_dir: str, steps: int, noise_prop: float, seed: float = None, verbose: bool = True
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

    data = get_all_data_values(data_path)
    init_conds = data[0].reshape((1, 1, 73, 721, 1440))
    init_conds = torch.from_numpy(init_conds).float()
    
    _, era5_stds = load_era5_stats(era5_stats_dir)
    stds = era5_stds[0, :, 0, 0].reshape(1, 1, 73, 1, 1)
    base_noise = torch.randn_like(init_conds) # N(0,1)
    init_conds_noise = init_conds + base_noise * noise_prop * stds # init_conds + N(0,noise_prop*stds)
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
    ds.to_netcdf(output_path, mode='w')
    ds.close()

    if verbose:
        print(f"Saved forecast to {output_path}!")
