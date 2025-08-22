import sys
import os
import numpy as np
from noisy_hurricane import run_inference
from compute_error import update_error_log, generate_global_error_dataset
from visualize_hurricane import *
from plot_errors import *
from load_data import load_dataset
from config import *

error_log = {}

if PREDICT:
    seed_idx = 0
    for noise_pct in NOISE_PCTS:
        for _ in range(NUM_EXPERIMENTS):
            seed = SEEDS[seed_idx]
            print(f"Generating forecast with noise {noise_pct} and seed {seed}...")
            run_inference(MODEL_DIR, TRUE_PATH, PRED_PATH, ERA5_STATS_DIR, TIMESTEPS, noise_pct, seed, verbose = False)
            print("Updating error log...")
            update_error_log(error_log, noise_pct, seed, TRUE_PATH, PRED_PATH, TIMESTEPS)
            seed_idx += 1
    if not CLEAN_UP:
        update_json(ERROR_LOG_PATH, error_log)
else:
    print("Loading error log...")
    error_log = load_json(ERROR_LOG_PATH)

if not error_log:
    print("error log empty.")
    sys.exit(1)

worst_seeds = {}
median_seeds = {}
best_seeds = {}

set_plot_parameters()

if PLOT_ERROR_LOCAL:
    plot_tracking_error_vs_noise(LOCAL_ERR_PLOT_DIR, error_log)
    # plot_tracking_error_vs_time_all_noise(LOCAL_ERR_PLOT_DIR, error_log)

for noise_pct, seed2error in error_log.items():
    total_errors = [(seed, errors[-1]) for seed, errors in seed2error.items()]
    total_errors.sort(key = lambda x: x[1])
    
    best_seeds[noise_pct] = total_errors[0][0]
    median_seeds[noise_pct] = total_errors[len(total_errors) // 2][0]
    worst_seeds[noise_pct] = total_errors[-1][0]

# if VISUALIZE_GLOBAL:
#     # global visualization of real data
#     print("Visualizing original data globally...")
#     original_global_plot_dir = os.path.join(GLOBAL_PRED_PLOT_DIR, "original")
#     for channel in ["msl", "u10m", "v10m"]:
#         visualize_global(
#             original_global_plot_dir, TRUE_PATH, channel, TIMESTEPS, ERA5_STATS_DIR, 
#             title_prefix = f"FourCastNetv2 Hurricane Florence Prediction - Original Data - {channel}", 
#             filename_prefix = f"{channel}_original"
#         )
#         animate_frames(
#             os.path.join(original_global_plot_dir, channel), 
#             animation_name = f"animated_{channel}_original.gif"
#         )

if VISUALIZE_LOCAL or VISUALIZE_GLOBAL or PLOT_ERROR_LOCAL or PLOT_ERROR_GLOBAL:
    # local and global visualizations of predicted data and errors
    for noise_pct in NOISE_PCTS:
        for label, seed_dict in [("best", best_seeds), ("median", median_seeds), ("worst", worst_seeds)]:
            noise_str = f"noise{int(noise_pct*100):02d}"
            seed = seed_dict[noise_pct]
            
            print(f"Regenerating forecast with noise {noise_pct} and {label} seed {seed}...")
            run_inference(MODEL_DIR, TRUE_PATH, PRED_PATH, ERA5_STATS_DIR, TIMESTEPS, noise_pct, seed, verbose=False)

            if VISUALIZE_LOCAL:
                print(f"Visualizing {label} seed locally...")
                plot_dir = os.path.join(LOCAL_PRED_PLOT_DIR, noise_str, label)
                visualize_local(plot_dir, TRUE_PATH, PRED_PATH, noise_pct, TIMESTEPS)
                animate_frames(plot_dir, animation_name=f"animated_{label}_{noise_str}.gif", duration=1000)
                visualize_local_trajectories_only(
                    plot_dir, TRUE_PATH, PRED_PATH, TIMESTEPS,
                    title = f"Hurricane Florence True vs. Predicted Trajectories ({noise_pct * 100:.0f}% noise)",
                    filename = f"traj_only_{label}_{noise_str}.png"
                )

            if VISUALIZE_GLOBAL:
                print(f"Visualizing {label} seed globally...")
                plot_dir = os.path.join(GLOBAL_PRED_PLOT_DIR, noise_str, label)
                for channel in ["msl", "u10m", "v10m"]:
                    visualize_global(
                        plot_dir, PRED_PATH, channel, TIMESTEPS, ERA5_STATS_DIR, 
                        title_prefix = f"FourCastNetv2 Hurricane Florence Prediction - {noise_pct * 100.}% noise - {channel}", 
                        filename_prefix = f"{channel}_{noise_str}"
                    )
                    animate_frames(os.path.join(plot_dir, channel), animation_name=f"animated_{channel}_{label}_{noise_str}.gif")

            generate_global_error_dataset(ERROR_DATAPATH, TRUE_PATH, PRED_PATH)
            
            if PLOT_ERROR_LOCAL:
                print(f"Plotting {label} local errors...")
                plot_dir = os.path.join(LOCAL_ERR_PLOT_DIR, noise_str, label)
                for channel in ["msl", "u10m", "v10m"]:
                    plot_pixelwise_error_hists(
                        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS,
                        title_prefix = f"{int(noise_pct*100)}% Noise Added", 
                        filename_prefix = f"{channel}_error_hist_{label}_{noise_str}",
                        mode = 'local'
                    )
                    plot_pixelwise_error_summary(
                        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS, 
                        title = f"{int(noise_pct*100)}% Noise Added", 
                        filename = f"{channel}_error_summary_{label}_{noise_str}",
                        mode = 'local'
                    )
                    plot_pixelwise_error_moments(
                        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS, 
                        title = f"{int(noise_pct*100)}% Noise Added", 
                        filename = f"{channel}_error_moments_{label}_{noise_str}",
                        mode = 'local'
                    )
                    

            if PLOT_ERROR_GLOBAL:
                print(f"Plotting {label} global errors...")
                plot_dir = os.path.join(GLOBAL_ERR_PLOT_DIR, noise_str, label)
                for channel in ["msl", "u10m", "v10m"]:
                    # visualize_global(
                    #     plot_dir, ERROR_DATAPATH, channel, TIMESTEPS, ERA5_STATS_DIR, 
                    #     title_prefix = f"FourCastNetv2 Hurricane Florence Prediction Error - {noise_pct * 100.}% noise - {channel}", 
                    #     filename_prefix = f"{channel}_error_{noise_str}",
                    #     is_error_plot = True
                    # )
                    # animate_frames(os.path.join(plot_dir, channel), animation_name=f"animated_error_{channel}_{label}_{noise_str}.gif")
                    plot_pixelwise_error_hists(
                        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS,
                        title_prefix = f"{int(noise_pct*100)}% Noise Added", 
                        filename_prefix = f"{channel}_error_hist_{label}_{noise_str}"
                    )
                    plot_pixelwise_error_summary(
                        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS, 
                        title = f"{int(noise_pct*100)}% Noise Added", 
                        filename = f"{channel}_error_summary_{label}_{noise_str}"
                    )
                    plot_pixelwise_error_moments(
                        os.path.join(plot_dir, channel), ERROR_DATAPATH, channel, TIMESTEPS, 
                        title = f"{int(noise_pct*100)}% Noise Added", 
                        filename = f"{channel}_error_moments_{label}_{noise_str}"
                    )

if CLEAN_UP:
    if os.path.exists(PRED_PATH):
        print("Deleting forecast...")
        os.remove(PRED_PATH)
    if os.path.exists(ERROR_LOG_PATH):
        print("Deleting error log...")
        os.remove(ERROR_LOG_PATH)
