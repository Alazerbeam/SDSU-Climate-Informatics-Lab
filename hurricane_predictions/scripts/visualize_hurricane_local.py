import numpy as np
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from tqdm import tqdm
from PIL import Image
from load_data import *
from track_hurricane import *
from config import *

def visualize_florence(plot_dir, true_data, pred_data, noise_pct, timesteps):   
    # load the data
    ds_pred = load_dataset(pred_data)
    ds_true = load_dataset(true_data)
    
    pred_msl_data = get_channel_data(ds_pred, 'msl')
    msl_pred_limited = limit_data(pred_msl_data, timesteps, x_min, x_max, y_min, y_max)
    true_msl_data = get_channel_data(ds_true, 'msl')
    msl_true_limited = limit_data(true_msl_data, timesteps, x_min, x_max, y_min, y_max)
    
    u10m_true_data = get_channel_data(ds_true, 'u10m')
    u10m_true_limited = limit_data(u10m_true_data, timesteps, x_min, x_max, y_min, y_max, stride = 2)
    u10m_pred_data = get_channel_data(ds_pred, 'u10m')
    u10m_pred_limited = limit_data(u10m_pred_data, timesteps, x_min, x_max, y_min, y_max, stride = 2)
    
    v10m_true_data = get_channel_data(ds_true, 'v10m')
    v10m_true_limited = limit_data(v10m_true_data, timesteps, x_min, x_max, y_min, y_max, stride = 2)
    v10m_pred_data = get_channel_data(ds_pred, 'v10m')
    v10m_pred_limited = limit_data(v10m_pred_data, timesteps, x_min, x_max, y_min, y_max, stride = 2)
    
    windvec_bases_x = np.arange(-90, -70, 0.5)
    windvec_bases_y = np.arange(30, 40, 0.5)

    track_true_x, track_true_y, track_pred_x, track_pred_y = track_true_and_pred_locs(msl_true_limited, msl_pred_limited)

    # convert indices of min msl positions to lat and lon
    track_true_y = index_to_lat_vec(np.array(track_true_y) + y_min)
    track_true_x = index_to_lon_vec(np.array(track_true_x) + x_min)
    track_pred_y = index_to_lat_vec(np.array(track_pred_y) + y_min)
    track_pred_x = index_to_lon_vec(np.array(track_pred_x) + x_min)
    
    # prepare to plot
    projection = ccrs.PlateCarree(central_longitude = 180)
    dttime = datetime(2023, 9, 13, 0, 0)
    
    nc_extent = [index_to_lon(x_min), index_to_lon(x_max),
                 index_to_lat(y_max), index_to_lat(y_min)]  # y_max is bottom in image
    titles = ['Original Data', f'FourCastNetv2 Prediction ({noise_pct * 100.}% noise)']
    
    os.makedirs(plot_dir, exist_ok=True)
    
    for idx in tqdm(range(timesteps), desc="Visualizing trajectory"):
        # set up figure
        fig = plt.figure(figsize=(13, 14))
        gs = gridspec.GridSpec(2, 2, width_ratios=[20, 0.5], height_ratios=[1, 1], wspace=0.06, hspace=0.12)
        axs = [fig.add_subplot(gs[i, 0], projection=projection) for i in range(2)]
        fig.suptitle(f"Hurricane Florence True vs. Predicted MSL - {dttime + timedelta(hours = idx * 6)}", fontsize=20, y=0.92)
    
        #set up colorbar
        cbar_ax = fig.add_subplot(gs[:, 1])  # spans both rows in 2nd column
    
        # set up data
        msl_fields = [msl_true_limited[idx], msl_pred_limited[idx]]
        u10m_fields = [u10m_true_limited[idx], u10m_pred_limited[idx]]
        v10m_fields = [v10m_true_limited[idx], v10m_pred_limited[idx]]
    
        # plot both true and pred msl vals
        for ax, title, msl_data, u10m_data, v10m_data in zip(axs, titles, msl_fields, u10m_fields, v10m_fields):
            ax.set_extent(nc_extent, crs = ccrs.PlateCarree())
            ax.coastlines()
            img = ax.imshow(msl_data / 100, cmap='jet', extent = nc_extent, 
                            transform = ccrs.PlateCarree(), vmin = 977.5, vmax = 1022.5)
        
            # plot data
            ax.scatter(track_true_x, track_true_y, s = 30, marker = 's', c = '#32CD32', transform = ccrs.PlateCarree())
            ax.plot(track_true_x, track_true_y, c = '#32CD32', label = 'True', transform = ccrs.PlateCarree())
            ax.scatter(track_pred_x, track_pred_y, s = 30, marker = 'o', c = '#1E90FF', transform = ccrs.PlateCarree())
            ax.plot(track_pred_x, track_pred_y, c = '#1E90FF', label = 'Forecast', transform = ccrs.PlateCarree())
            ax.legend(fontsize=16)
    
            ax.set_title(title, fontsize=16)
    
            # plot time labels (just 1st and last)
            text_props = dict(boxstyle = 'round,pad=0.3', edgecolor = 'black', facecolor = 'white', alpha = 0.7)
            timestamp = dttime + timedelta(hours = 0)
            time_label = timestamp.strftime('%m/%d, %H:00')
            ax.text(track_pred_x[0], track_pred_y[0] + .35, time_label, transform = ccrs.PlateCarree(),
                    fontsize = 12, color = 'black', rotation = 45, bbox = text_props)
            timestamp = dttime + timedelta(hours = 6 * (timesteps - 1))
            time_label = timestamp.strftime('%m/%d, %H:00')
            if track_true_y[-1] > track_pred_y[-1]:
                ax.text(track_pred_x[-1] - 1.8, track_pred_y[-1] - 2.1, time_label, transform = ccrs.PlateCarree(),
                        fontsize = 12, color = 'black', rotation = 45, bbox = text_props)
                ax.text(track_true_x[-1], track_true_y[-1] + .35, time_label, transform = ccrs.PlateCarree(),
                        fontsize = 12, color = 'black', rotation = 45, bbox = text_props)
            else:
                ax.text(track_true_x[-1] - 1.8, track_true_y[-1] - 2.1, time_label, transform = ccrs.PlateCarree(),
                        fontsize = 12, color = 'black', rotation = 45, bbox = text_props)
                ax.text(track_pred_x[-1], track_pred_y[-1] + .35, time_label, transform = ccrs.PlateCarree(),
                        fontsize = 12, color = 'black', rotation = 45, bbox = text_props)
            
            # plot wind vectors
            q = ax.quiver(windvec_bases_x, windvec_bases_y, u10m_data, v10m_data, transform = ccrs.PlateCarree(), 
                          scale=58, scale_units='inches', width=0.0025, alpha=0.7)
            
            # Add lat/lon gridlines
            gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color='gray')
            
            # Customize gridline labels
            gl.top_labels = False   # Hide top labels
            gl.right_labels = False # Hide right labels
            gl.xlabel_style = {'size': 12, 'color': 'black'}
            gl.ylabel_style = {'size': 12, 'color': 'black'}
    
        # finish setting up colorbar
        cbar = fig.colorbar(img, cax=cbar_ax)
        cbar.set_label("Mean Sea Level Pressure (hPa)", fontsize = 16, labelpad = 10)
        cbar.set_ticks([980 + i * 5 for i in range(9)])
    
        # save figure
        folder = os.path.join(plot_dir, "frames")
        os.makedirs(folder, exist_ok = True)
        plt.savefig(os.path.join(folder, f'noise{int(noise_pct * 100.):02d}_t{idx:02d}.png'))
        plt.close()

# compile all plots from a channel over all timesteps into an animated gif
def animate_florence(plot_dir, noise_pct, num_frames):
    filenames = [os.path.join(plot_dir, f'frames/noise{int(noise_pct * 100.):02d}_t{idx:02d}.png') for idx in range(num_frames)]
    images = [Image.open(f) for f in filenames if os.path.exists(f)]
    
    output_path = os.path.join(plot_dir, f'animated_noise{int(noise_pct * 100.):02d}.gif')
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=1000,  # duration per frame in milliseconds
        loop=0
    )
