import torch
import numpy as np
import argparse
import os
from datetime import datetime
from pathlib import Path
from torch.distributions import Chi2, LogNormal
import xarray as xr
from tqdm import tqdm
from earth2mip.networks import get_model

# define constants
BATCH_SIZE = 1  # define batch size
CHANNELS = 73   # define number of channels

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--distribution", choices=["normal", "uniform", "chi-sq", "lognormal"], required=True)
parser.add_argument("--forecast_filepath", type=str, required=True)
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--era5_stats_dir", type=str, required=True)
parser.add_argument("--timesteps", type=int, required=True)
parser.add_argument("--mean", type=float, default=0)
parser.add_argument("--std", type=float, default=1)
parser.add_argument("--df", type=float, default=1)
parser.add_argument("--a", type=float, default=0)
parser.add_argument("--b", type=float, default=1)
args = parser.parse_args()

# Set means and stds to match your synthetic distributions
def create_synthetic_stats(output_dir, mean=0, std=1):
    global_means = np.ones((BATCH_SIZE, CHANNELS, 1, 1)) * mean
    global_stds = np.ones((BATCH_SIZE, CHANNELS, 1, 1)) * std
    np.save(f"{output_dir}/global_means.npy", global_means)
    np.save(f"{output_dir}/global_stds.npy", global_stds)

# Calculate the mean and std from given distribution with given parameters
def calc_mean_and_std_from_distr(distribution='normal', mean=0, std=1, df=1, a=0, b=1):
    if distribution == 'normal':
        return mean, std
    elif distribution == 'chi-sq':
        return df, np.sqrt(2*df)
    elif distribution == 'uniform':
        return (a+b)/2., np.sqrt((b-a)**2./12.)
    elif distribution == 'lognormal':
        return np.exp(mean + std**2. / 2.), np.sqrt((np.exp(std**2.) - 1.) * np.exp(2. * mean + std**2.))
    else:
        print("Unknown distribution. Defaulting to N(0,1).")
        return 0., 1.

# generate random initial conditions from uniform, normal, Chi Sq, LogNormal.
def generate_rand_init_conds(
    batch_size, time_steps, channels, height, width, distribution='normal', mean=0, std=1, df=1, a=0, b=1
    ):
    # Gaussian distribution with specified mean and std (default 0, 1)
    if distribution == 'normal':
        return torch.normal(mean=mean, std=std, size=(batch_size, time_steps, channels, height, width))
    # chi squared distribution with specified df (default 1)
    elif distribution == 'chi-sq':
        return Chi2(df).sample((batch_size, time_steps, channels, height, width))
    # uniform distribution with specified range (default [0, 1))
    elif distribution == 'uniform':
        return a + (b - a) * torch.rand(batch_size, time_steps, channels, height, width)
    # lognormal distribution with specified mean and std (default 0, 1)
    elif distribution == 'lognormal':
        return LogNormal(mean, std).sample((batch_size, time_steps, channels, height, width))
    # otherwise use default normal distribution
    else:
        print('Unknown distribution. Defaulting to N(0,1).')
        return torch.randn(batch_size, time_steps, channels, height, width)

# rescale the output of FCNv2 to match the era5 distribution for each channel
def rescale_fcnv2_output(output_array, distr_mean, distr_std):
    era5_means = np.load(os.path.join(args.era5_stats_dir, "global_means.npy"))
    era5_stds = np.load(os.path.join(args.era5_stats_dir, "global_stds.npy"))
    
    distr_means = np.ones((BATCH_SIZE, CHANNELS, 1, 1)) * distr_mean
    distr_stds = np.ones((BATCH_SIZE, CHANNELS, 1, 1)) * distr_std
    
    normalized = (output_array - distr_means) / distr_stds
    rescaled = normalized * era5_stds + era5_means
    
    return rescaled.astype(np.float32)

# load model, run inference, save forecast to NetCDF file
def run_inference(
    output_path: str, steps: int, 
    distribution: str = 'normal', mean=0, std=1, df=1, a=0, b=1, 
    verbose: bool = True
    ):

    # use gpu is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # pre-calculate mean and std of chosen distribution
    distr_mean, distr_std = calc_mean_and_std_from_distr(distribution, mean, std, df, a, b)
    # replace npy files
    create_synthetic_stats(args.model_dir, distr_mean, distr_std)
    # save path to modified files
    pkg_dir = Path(args.model_dir).resolve()
    
    # load the model
    model = get_model(f"file://{pkg_dir}")
    model = model.to(device)  # move to gpu if available
    
    if verbose:
        print("Successfully loaded model!")
    
    # generate random input
    height, width = model.grid.shape
    
    rand_input = generate_rand_init_conds(
        batch_size = BATCH_SIZE,
        time_steps = model.n_history + 1,
        channels = CHANNELS,
        height = height,
        width = width,
        distribution = distribution,
        mean = mean,
        std = std,
        df = df,
        a = a,
        b = b
    ).to(device)
    
    if verbose:
        print("Successfully generated random initial conditions!")
    
    # run inference
    time = datetime(2023, 1, 1)
    predictions = []
    times = []
    
    iterator = model(time, rand_input)
    for _ in tqdm(range(steps), desc="Generating forecast"):
        temp_time, temp_output, _ = next(iterator)
        predictions.append(temp_output[0].cpu().numpy())
        times.append(temp_time)

    if verbose:
        print("Successfully ran inference!")
    
    predictions = np.stack(predictions)     # shape: (steps, channels, lat, lon)

    # rescale output to match era5 distributions
    rescaled = rescale_fcnv2_output(predictions, distr_mean, distr_std)

    # convert to xr.Dataset
    data_arr = xr.DataArray(
        rescaled,
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
    print(f"Saved forecast to {output_path}!")

run_inference(
    output_path = args.forecast_filepath,
    steps = args.timesteps,
    distribution = args.distribution,
    mean = args.mean,
    std = args.std,
    df = args.df,
    a = args.a,
    b = args.b,
    verbose = True
)