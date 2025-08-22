import sys
import os
import numpy as np
from noisy_hurricane import run_inference
from compute_error import update_error_log, generate_global_error_dataset
from visualize_hurricane import *
from plot_errors import *
from load_data import load_dataset
from config import *

plot_dir = os.path.join(GLOBAL_ERR_PLOT_DIR, "random", "uniform")

generate_global_error_dataset(ERROR_DATAPATH, TRUE_PATH, PRED_PATH)

for channel in ["msl", "u10m", "v10m"]:
    visualize_global(
        plot_dir, ERROR_DATAPATH, channel, TIMESTEPS, ERA5_STATS_DIR, 
        title_prefix = f"FourCastNetv2 Hurricane Florence Prediction Error - Uniform Distribution - {channel}", 
        filename_prefix = f"{channel}_error_uniform",
        is_error_plot = True
    )
    animate_frames(os.path.join(plot_dir, channel), animation_name=f"animated_error_{channel}_uniform.gif")

set_plot_parameters()
for channel in ["msl", "u10m", "v10m"]:
    plot_pixelwise_error_hists(
        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS,
        title_prefix = f"Uniform",
        filename_prefix = f"{channel}_error_hist_uniform"
    )
    plot_pixelwise_error_summary(
        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS, 
        title = f"Uniform", 
        filename = f"{channel}_error_summary_uniform"
    )
    plot_pixelwise_error_moments(
        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS, 
        title = f"Uniform", 
        filename = f"{channel}_error_moments_uniform"
    )