import matplotlib.pyplot as plt
import numpy as np
import os
from compute_error import compute_error_distribution

def plot_error_vs_time_given_noise(plot_dir, error_log, noise_pct):
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
    plt.savefig(os.path.join(plot_dir, f"noise{int(noise_pct * 100.):02d}_cumulative_error_vs_time.png"))
    plt.close()

def plot_error_vs_time_all_noise(plot_dir, error_log):
    for noise_pct in error_log:
        plot_error_vs_time_given_noise(plot_dir, error_log, noise_pct)

def plot_error_vs_noise(plot_dir, error_log):
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
        capsize=3,
        elinewidth=1,
        marker='o',
        markersize=3,
        linestyle=''
    )
    
    plt.xlabel("Noise (%)")
    plt.ylabel("Total Error (km)")
    plt.title("Total Trajectory Error Per Noise Level")
    plt.savefig(os.path.join(plot_dir, "total_error_vs_noise.png"))
    plt.close()
