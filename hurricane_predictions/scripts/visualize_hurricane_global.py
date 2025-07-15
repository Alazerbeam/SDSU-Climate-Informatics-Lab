import cfgrib
import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import argparse
from tqdm import tqdm

NOISE_PCT = 50

# load the forecast NetCDF and display it
def load_dataset():
    ds = xr.open_dataset(f"/home/jovyan/hurricane_predictions/data/florence_{NOISE_PCT}noise.nc", engine='netcdf4')
    print(ds)
    print("Available channels:", ds.channel.values)
    return ds

# load the means and stds of era5 dataset for all channels
def load_era5_stats():
    era5_means = np.load("/home/jovyan/era5_stats/global_means.npy")
    era5_stds = np.load("/home/jovyan/era5_stats/global_stds.npy")
    return era5_means, era5_stds

# return full name of channel including units to display on plot
def full_name(channel_name):
    if channel_name == "sp": return "Surface Pressure (Pa)"
    elif channel_name == "msl": return "Mean Sea Level Pressure (Pa)"
    elif channel_name == "tcwv": return "Total Column Water Vapor ($kg/m^2$)"
    elif channel_name[-1] == "m":  # if last character is m, represents meters above surface
        if channel_name[0] == "u":
            return f"East/West Wind {channel_name[1:-1]}m Above Surface (m/s)"
        elif channel_name[0] == "v":
            return f"North/South Wind {channel_name[1:-1]}m Above Surface (m/s)"
        elif channel_name[0] == "t":
            return f"Temperature {channel_name[1:-1]}m Above Surface (K)"
    elif channel_name[0] == "u": return f"East/West Wind at {channel_name[1:]} hPa (m/s)"
    elif channel_name[0] == "v": return f"North/South Wind at {channel_name[1:]} hPa (m/s)"
    elif channel_name[0] == "z": return f"Geopotential at {channel_name[1:]} hPa ($m^2/s^2$)"
    elif channel_name[0] == "t": return f"Temperature at {channel_name[1:]} hPa (K)"
    elif channel_name[0] == "r": return f"Relative Humidity at {channel_name[1:]} hPa (%)"
    elif channel_name[0] == "q": return f"Specific Humidity at {channel_name[1:]} hPa (kg/kg)"
    else: return "Unknown channel."

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
    ds, channel, num_steps, output_image_dir, era5_means, era5_stds
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

        ax.set_title(f"FourCastNetv2 Hurricane Florence Prediction - {NOISE_PCT}% noise - {channel} - {t * 6} hours from initialization")
        
        # Add colorbar
        cbar = fig.colorbar(im1, ax=ax, orientation='horizontal', fraction=0.046, pad=0.08)
        cbar.set_ticks(tick_vals) # set tick values to match stds away from mean
        cbar.set_label(full_name(channel))
    
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False

        filename = f"{channel}_noise{NOISE_PCT}_t{t:02d}.png"

        plt.savefig(f"{output_image_dir}/{filename}")
        plt.close(fig)

# only load all necessary files once
ds = load_dataset()
era5_means, era5_stds = load_era5_stats()

for var in ["u10m", "v10m", "msl"]:   # list of variables you want to visualize
    output_dir = f"/home/jovyan/hurricane_predictions/plots/global/noise{NOISE_PCT}/{var}/frames"
    os.makedirs(output_dir, exist_ok=True)  # create directory if does not already exist
    robinson_predictions_plot(
        ds, var, 15, output_dir, era5_means, era5_stds
    )

### animate frames ###

from PIL import Image

# compile all plots from a channel over all timesteps into an animated gif
def sequence2gif(images_path, channel, num_frames):
    filenames = [f"{images_path}/frames/{channel}_noise{NOISE_PCT}_t{t:02d}.png" for t in range(num_frames)]
    images = [Image.open(f) for f in filenames if os.path.exists(f)]
    
    output_path = os.path.join(images_path, f"animated_{channel}_noise{NOISE_PCT}.gif")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=500,  # duration per frame in milliseconds
        loop=0
    )

for var in ["u10m", "v10m", "msl"]:
    sequence2gif(f"/home/jovyan/hurricane_predictions/plots/global/noise{NOISE_PCT}/{var}", var, 15)