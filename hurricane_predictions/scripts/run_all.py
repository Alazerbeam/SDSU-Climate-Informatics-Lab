import sys
import os
import numpy as np
from noisy_hurricane import run_inference, randomize_seeds
from compute_error import update_error_log
from visualize_hurricane_local import *
from plot_errors import *

# choose which steps to take
PREDICT = True
PLOT_ERROR = True
VISUALIZE = True
CLEAN_UP = False

# choose which noise levels to test + how many timesteps
NOISE_PCTS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
# NOISE_PCTS = [0.20, 0.35]
NUM_EXPERIMENTS = 30
SEEDS = randomize_seeds(len(NOISE_PCTS) * NUM_EXPERIMENTS)
TIMESTEPS = 15

# choose the directories
PRED_PATH = "/home/jovyan/hurricane_predictions/data/hurricane_run.nc"
TRUE_PATH = "/home/jovyan/hurricane_predictions/data/fcnv2_input.nc"
ERROR_LOG_PATH = "/home/jovyan/hurricane_predictions/data/error_log.json"
MODEL_DIR = "/home/jovyan/fcnv2_params"
ERA5_STATS_DIR = "/home/jovyan/era5_stats"
TRAJ_PLOT_DIR = "/home/jovyan/hurricane_predictions/plots/local"
ERROR_PLOT_DIR = "/home/jovyan/hurricane_predictions/plots/errors"

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

# print(error_log)

worst_seeds = {}
median_seeds = {}
best_seeds = {}

if PLOT_ERROR:
    plot_error_vs_noise(ERROR_PLOT_DIR, error_log)

for noise_pct, seed2error in error_log.items():
    if PLOT_ERROR:
        plot_error_vs_time_given_noise(ERROR_PLOT_DIR, error_log, noise_pct)

    if VISUALIZE:
        total_errors = [(seed, errors[-1]) for seed, errors in seed2error.items()]
        total_errors.sort(key = lambda x: x[1])
        
        best_seeds[noise_pct] = total_errors[0][0]
        median_seeds[noise_pct] = total_errors[len(total_errors) // 2][0]
        worst_seeds[noise_pct] = total_errors[-1][0]

# print(worst_seeds)
# print(median_seeds)
# print(best_seeds)

if VISUALIZE:
    for noise_pct in NOISE_PCTS:
        print(f"Regenerating forecast with noise {noise_pct} and best seed {best_seeds[noise_pct]}...")
        run_inference(MODEL_DIR, TRUE_PATH, PRED_PATH, ERA5_STATS_DIR, TIMESTEPS, noise_pct, best_seeds[noise_pct], verbose = False)
        print("Visualizing best seed...")
        best_traj_plot_dir = os.path.join(TRAJ_PLOT_DIR, f"noise{int(noise_pct * 100.):02d}", "best")
        visualize_florence(best_traj_plot_dir, TRUE_PATH, PRED_PATH, noise_pct, TIMESTEPS)
        animate_florence(best_traj_plot_dir, noise_pct, TIMESTEPS)
    
        print(f"Regenerating forecast with noise {noise_pct} and median seed {median_seeds[noise_pct]}...")
        run_inference(MODEL_DIR, TRUE_PATH, PRED_PATH, ERA5_STATS_DIR, TIMESTEPS, noise_pct, median_seeds[noise_pct], verbose = False)
        print("Visualizing median seed...")
        median_traj_plot_dir = os.path.join(TRAJ_PLOT_DIR, f"noise{int(noise_pct * 100.):02d}", "median")
        visualize_florence(median_traj_plot_dir, TRUE_PATH, PRED_PATH, noise_pct, TIMESTEPS)
        animate_florence(median_traj_plot_dir, noise_pct, TIMESTEPS)
    
        print(f"Regenerating forecast with noise {noise_pct} and worst seed {worst_seeds[noise_pct]}...")
        run_inference(MODEL_DIR, TRUE_PATH, PRED_PATH, ERA5_STATS_DIR, TIMESTEPS, noise_pct, worst_seeds[noise_pct], verbose = False)
        print("Visualizing worst seed...")
        worst_traj_plot_dir = os.path.join(TRAJ_PLOT_DIR, f"noise{int(noise_pct * 100.):02d}", "worst")
        visualize_florence(worst_traj_plot_dir, TRUE_PATH, PRED_PATH, noise_pct, TIMESTEPS)
        animate_florence(worst_traj_plot_dir, noise_pct, TIMESTEPS)

if CLEAN_UP:
    if os.path.exists(PRED_PATH):
        print("Deleting forecast...")
        os.remove(PRED_PATH)
    if os.path.exists(ERROR_LOG_PATH):
        print("Deleting error log...")
        os.remove(ERROR_LOG_PATH)
