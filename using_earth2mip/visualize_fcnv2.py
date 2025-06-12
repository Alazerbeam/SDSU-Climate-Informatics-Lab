import cfgrib
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
from tqdm import tqdm

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--distribution", choices=["normal", "uniform", "chi-sq", "lognormal"], required=True)
parser.add_argument("--distr_name", type=str, required=True)
parser.add_argument("--forecast_filepath", type=str, required=True)
parser.add_argument("--timesteps", type=int, required=True)
parser.add_argument("--visual_dir", type=str, required=True)
parser.add_argument("--era5_stats_dir", type=str, required=True)
parser.add_argument("--variables", nargs='+', required=True, help="List of variable names to visualize")
args = parser.parse_args()

# load the forecast NetCDF and display it
def load_dataset():
    ds = xr.open_dataset(args.forecast_filepath, engine='netcdf4')
    print(ds)
    print("Available channels:", ds.channel.values)
    return ds

# load the means and stds of era5 dataset for all channels
def load_era5_stats():
    era5_means = np.load(os.path.join(args.era5_stats_dir, "global_means.npy"))
    era5_stds = np.load(os.path.join(args.era5_stats_dir, "global_stds.npy"))
    return era5_means, era5_stds

# find the index of a specified channel
def get_channel_index(ds, channel_name):
    return np.where(ds.channel.values == channel_name)[0][0]

# retrieve the data from a specified channel
def get_channel_data(ds, channel_name):
    return ds['forecast'].isel(channel=get_channel_index(ds, channel_name))

# retrieve the mean and std from era5 for a specific channel
def get_era5_channel_stats(ds, channel_name, era5_means, era5_stds):
    channel_idx = get_channel_index(ds, channel_name)
    return era5_means[0,channel_idx,0,0], era5_stds[0,channel_idx,0,0]

# plot the results from a specified channel over all timesteps, scaled accordingly
def robinson_predictions_plot(
    ds, channel, num_steps, output_image_dir, era5_means, era5_stds, 
    distribution='normal', distr_name='$N(\mu=0,\sigma=1)$'
):
    data = get_channel_data(ds, channel)

    # use era5 stats as bounds
    era5_ch_mean, era5_ch_std = get_era5_channel_stats(ds, channel, era5_means, era5_stds)
    vmin = era5_ch_mean - 2.25 * era5_ch_std
    vmax = era5_ch_mean + 2.25 * era5_ch_std
    tick_vals = [era5_ch_mean + i * era5_ch_std for i in range(-2, 3)]
    
    # Create subplots with the Robinson projection centered on the Pacific (central_longitude=180)
    central_longitude = 180
    projection = ccrs.Robinson(central_longitude=central_longitude)

    # Define the extent of the map (in degrees)
    extent = (-180, 180, -90, 90)

    for t in tqdm(range(num_steps), desc=f"Visualizing {channel}"):
        dat = data.isel(time=t)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), subplot_kw={'projection': projection})
    
        # Plot the prediction data
        ax.set_global()
        im1 = ax.imshow(np.roll(dat, shift=dat.shape[-1]//2, axis=-1), 
                           transform=ccrs.PlateCarree(central_longitude=0), 
                           cmap="jet", extent=extent, origin='upper',
                           vmin=vmin, vmax=vmax)
        ax.coastlines()

        ax.set_title(f"FourCastNetv2 prediction from {distr_name} - {channel} - {t * 6} hours from initialization")
        
        # Add colorbar
        cbar = fig.colorbar(im1, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
        cbar.set_ticks(tick_vals) # set tick values to match stds away from mean
    
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False

        filename = f"{distribution}_{channel}_t{t:02d}.png"

        plt.savefig(f"{output_image_dir}/{filename}")
        plt.close(fig)

# only load all necessary files once
ds = load_dataset()
era5_means, era5_stds = load_era5_stats()

for var in args.variables:   # list of variables you want to visualize
    output_dir = os.path.join(args.visual_dir, args.distribution, var, "frames")
    os.makedirs(output_dir, exist_ok=True)  # create directory if does not already exist
    robinson_predictions_plot(
        ds, var, args.timesteps, output_dir, era5_means, era5_stds, 
        distribution=args.distribution, distr_name=args.distr_name
    )
