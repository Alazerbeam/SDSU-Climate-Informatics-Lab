import subprocess
import os
from glob import glob

# === USER CONFIGURATION ===

# choosing which steps to take
predict = True          # determines whether a prediction will be generated
visualize = True        # determines whether the data will be visualized
animate = True          # determines whether the visualized data will be animated
clear_forecasts = True  # determines whether the forecasts will be deleted (recommended to free up space)

# choosing the distribution
distributions = ["normal", "chi-sq", "lognormal", "uniform"]  # these are the choices
distr_names = ["$N(\mu=0,\sigma=1)$", "$\chi^2(df=1)$", "$LogNormal(\mu=0,\sigma=1)$", "$Unif(a=0,b=1)$"] # name of distribution to display on plots (can use LaTeX)
mean = 0                    # parameters of the chosen distribution
std = 1                     # mean/std for normal/lognormal
a = 0                       # a/b for uniform
b = 1
df = 1                      # df for chi-sq

# choosing the directories
visual_dir = "/your/path/to/generated_visuals"  # directory to store generated images, will be organized by distribution and variable
model_dir = "/your/path/to/fcnv2_params"        # both should contain global_means.npy, global_stds.npy, weights.tar, metadata.json
era5_stats_dir = "/your/path/to/era5_stats"     # should contain the original global_means.npy and global_stds.npy

# choosing prediction/plotting settings
timesteps = 21                     # number of predictions to run (including initial conditions)
variables = ["t2m", "u10m", "v10m", "sp", "msl", "u100m", "v100m", "t100", "tcwv", "r100", "z100"] # which variables to visualize

for distribution, distr_name in zip(distributions, distr_names):
    print(f"Generating from {distribution} distribution...")
    
    forecast_filepath = f"/your/path/to/forecasts/forecast_{distribution}.nc"  # filepath to the forecast NetCDF file     
    
    # === STEP 1: Generate new forecast ===
    if predict:
        print("Running random initial condition forecast generation...")
        
        subprocess.run([
            "python3", "/home/jovyan/using_earth2mip/randinitconds_fcnv2.py",
            "--distribution", distribution,
            "--timesteps", str(timesteps),
            "--forecast_filepath", forecast_filepath,
            "--model_dir", model_dir,
            "--era5_stats_dir", era5_stats_dir,
            "--mean", str(mean),
            "--std", str(std),
            "--a", str(a),
            "--b", str(b),
            "--df", str(df)
        ], check=True)
    
    # === STEP 2: Visualize output ===
    if visualize:
        print("Visualizing output...")
        
        subprocess.run([
            "python3", "/home/jovyan/using_earth2mip/visualize_fcnv2.py",
            "--forecast_filepath", forecast_filepath,
            "--visual_dir", visual_dir,
            "--era5_stats_dir", era5_stats_dir,
            "--variables", *variables,
            "--distribution", distribution,
            "--distr_name", distr_name,
            "--timesteps", str(timesteps),
        ], check=True)
    
    # === STEP 3: Animate results ===
    if animate:
        print("Animating results...")
        subprocess.run([
            "python3", "/home/jovyan/using_earth2mip/animate_sequence.py",
            "--visual_dir", visual_dir,
            "--timesteps", str(timesteps),
            "--distribution", distribution,
            "--variables", *variables,
        ], check=True)

    # === STEP 4: Clean up forecasts ===
    forecast_files = glob("/home/jovyan/forecasts/*")
    if clear_forecasts and forecast_files:
        print(f"Deleting {len(forecast_files)} forecasts...")
        for file in forecast_files:
            os.remove(file)

print("All steps completed successfully.")