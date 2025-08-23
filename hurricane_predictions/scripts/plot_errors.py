import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from compute_error import *
from utils import units
from load_data import load_dataset, get_channel_data
from scipy.stats import skew, kurtosis
from config import x_min, x_max, y_min, y_max

def set_plot_parameters():
    plt.rcParams.update({
        "font.size": 21,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

def plot_tracking_error_vs_time_given_noise(plot_dir, error_log, noise_pct):
    plt.figure(figsize=(10, 6))
    
    timesteps, mean_error, std_error = compute_error_distribution(error_log, noise_pct)

    # Plot with vertical error bars
    plt.errorbar(
        timesteps * 6, mean_error, yerr=std_error,
        capsize=3,  # little caps on error bars
        elinewidth=1,
        marker='o',  # dots at points
        markersize=4,
        linestyle=''
    )
    
    plt.xlabel("Time (Hours)")
    plt.ylabel("Cumulative Error (km)")
    plt.title(f"Cumulative Trajectory Error Over Time for {int(noise_pct * 100.) : 02d}% Noise")
    os.makedirs(plot_dir, exist_ok = True)
    plt.savefig(os.path.join(plot_dir, f"noise{int(noise_pct * 100.):02d}_cumulative_error_vs_time.png"))
    plt.close()

def plot_tracking_error_vs_time_all_noise(plot_dir, error_log):
    for noise_pct in error_log:
        plot_tracking_error_vs_time_given_noise(plot_dir, error_log, noise_pct)

def plot_tracking_error_vs_noise(plot_dir, error_log):
    plt.figure(figsize=(10, 6))
    noise_pcts = np.array(list(error_log.keys())) * 100.
    mean_error = np.array([])
    std_error = np.array([])
    for noise_pct, seed2errors in error_log.items():
        _, mean_error_temp, std_error_temp = compute_error_distribution(error_log, noise_pct)
        mean_error = np.append(mean_error, mean_error_temp[-1])
        std_error = np.append(std_error, std_error_temp[-1])
    
    plt.errorbar(
        noise_pcts, mean_error, yerr=std_error,
        fmt = 'o', markersize=10,
        color='black',
        ecolor='black',
        elinewidth=2, capsize=5,
        markeredgecolor='black',
        markerfacecolor='#87CEEB',
        linestyle='--'
    )
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.xlabel("Noise (%)")
    plt.ylabel("Total Error (km)")
    plt.title("Total Trajectory Error Per Noise Level")
    os.makedirs(plot_dir, exist_ok = True)
    plt.savefig(os.path.join(plot_dir, "total_error_vs_noise.png"))
    plt.close()

def plot_pixelwise_error_hists(plot_dir, error_datapath, channel, num_steps, title_prefix, filename_prefix, mode='global'):
    os.makedirs(plot_dir, exist_ok = True)
    ds = load_dataset(error_datapath)
    data = get_channel_data(ds, channel)

    # set boundaries
    data_np = data.values
    max_abs = np.percentile(np.abs(data_np), 99)
    hist_range = (-max_abs, max_abs)

    # only plot 1st and last timesteps
    for t in [0, num_steps-1]:
        errors_t = data.isel(time=t).values
        if mode.lower()[0] == 'l':
            errors_t = errors_t[y_min:y_max, x_min:x_max]
        errors_t = errors_t.flatten()

        # plot histogram
        plt.figure(figsize=(10,6))
        plt.hist(errors_t, bins=100, range=hist_range, color="#87CEEB", edgecolor="black")
        plt.xlabel(f"Prediction Error ({units(channel)})")
        plt.ylabel("Pixel Count")
        # plt.title(title_prefix + f" - {t*6} hours from initialization")
        plt.title(title_prefix)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # save figure
        filename = filename_prefix + f"_t{t:02d}.png"
        plt.savefig(os.path.join(plot_dir, filename))
        plt.close()

# create plot of summary statistics (max, 95th percentile, ..., min) for a given experiment over all timesteps
def plot_pixelwise_error_summary(plot_dir, error_datapath, channel, num_steps, title, filename, mode='global'):
    os.makedirs(plot_dir, exist_ok = True)
    ds = load_dataset(error_datapath)
    data = get_channel_data(ds, channel)
    
    max_vals = []
    p95_vals = []
    p75_vals = []
    median_vals = []
    p25_vals = []
    p5_vals = []
    min_vals = []
    for t in range(num_steps):
        errors_t = data.isel(time=t).values
        if mode.lower()[0] == 'l': # if in local mode, limit the data to atlantic region
            errors_t = errors_t[y_min:y_max, x_min:x_max]
        errors_t = errors_t.flatten()
        
        max_vals.append(np.max(errors_t))
        p95_vals.append(np.percentile(errors_t, 95))
        p75_vals.append(np.percentile(errors_t, 75))
        median_vals.append(np.percentile(errors_t, 50))
        p25_vals.append(np.percentile(errors_t, 25))
        p5_vals.append(np.percentile(errors_t, 5))
        min_vals.append(np.min(errors_t))

    x_axis = np.arange(num_steps) * 6
    plt.figure(figsize=(10,6))
    plt.plot(
        x_axis, max_vals, color="#690101", linestyle='--', linewidth=2, marker='o', 
        markersize=8, markerfacecolor="#690101", markeredgecolor="black", markeredgewidth=1, label="Max Value"
    )
    plt.plot(
        x_axis, p95_vals, color="#b30e12", linestyle='--', linewidth=2, marker='o', 
        markersize=8, markerfacecolor="#b30e12", markeredgecolor="black", markeredgewidth=1, label="95th Percentile"
    )
    plt.plot(
        x_axis, p75_vals, color="#f7651b", linestyle='--', linewidth=2, marker='o', 
        markersize=8, markerfacecolor="#f7651b", markeredgecolor="black", markeredgewidth=1, label="75th Percentile"
    )
    plt.plot(
        x_axis, median_vals, color="#efc2fc", linestyle='--', linewidth=2, marker='o', 
        markersize=8, markerfacecolor="#efc2fc", markeredgecolor="black", markeredgewidth=1, label="Median Value"
    )
    plt.plot(
        x_axis, p25_vals, color="#87ceeb", linestyle='--', linewidth=2, marker='o', 
        markersize=8, markerfacecolor="#87ceeb", markeredgecolor="black", markeredgewidth=1, label="25th Percentile"
    )
    plt.plot(
        x_axis, p5_vals, color="#3376bd", linestyle='--', linewidth=2, marker='o', 
        markersize=8, markerfacecolor="#3376bd", markeredgecolor="black", markeredgewidth=1, label="5th Percentile"
    )
    plt.plot(
        x_axis, min_vals, color="#181ced", linestyle='--', linewidth=2, marker='o', 
        markersize=8, markerfacecolor="#181ced", markeredgecolor="black", markeredgewidth=1, label="Min Value"
    )
    plt.xlabel("Time (hours)")
    plt.ylabel(f"Error ({units(channel)})")
    plt.title(title)
    #plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.xticks(x_axis)
    plt.grid(True, linestyle='--', alpha=0.5)

    if filename.split(".")[-1] != "png":
        filename += ".png"
    
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    plt.close()

# create plots of mean, std, skew, kurtosis and save to plot_dir
def plot_pixelwise_error_moments(plot_dir, error_datapath, channel, num_steps, title, filename, mode='global'):
    os.makedirs(plot_dir, exist_ok = True)
    ds = load_dataset(error_datapath)
    data = get_channel_data(ds, channel)
    
    mean_vals = []
    std_vals = []
    skew_vals = []
    kurt_vals = []
    for t in range(num_steps):
        errors_t = data.isel(time=t).values
        if mode.lower()[0] == 'l':
            errors_t = errors_t[y_min:y_max, x_min:x_max]
        errors_t = errors_t.flatten()
        
        mean_vals.append(np.mean(errors_t))
        std_vals.append(np.std(errors_t))
        skew_vals.append(skew(errors_t))
        kurt_vals.append(kurtosis(errors_t))
    
    fig, axs = plt.subplots(4, 1, figsize=(8,8), sharex=True)
    x_axis = np.arange(num_steps) * 6

    for ax, data, label in zip(axs, [mean_vals, std_vals, skew_vals, kurt_vals], ["Mean", "SD", "Skew", "Kurt."]):
        ax.plot(x_axis, data, linestyle='--', linewidth=2, color="black", marker='o', markersize=8, markerfacecolor="#87CEEB", markeredgecolor="black", markeredgewidth=1)
        ax.set_ylabel(label)
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4))
        ax.grid(True, linestyle='--', alpha=0.5)

    axs[-1].set_xticks(x_axis)
    axs[-1].set_xlabel("Time (hours)")
    axs[0].set_title(title)

    if filename.split(".")[-1] != "png":
        filename += ".png"

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.subplots_adjust(hspace=0.25)
    plt.savefig(os.path.join(plot_dir, filename), bbox_inches='tight')
    plt.close()
    