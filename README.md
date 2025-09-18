# SDSU Robustness Test for FourCastNetv2 AI Weather Forecast Models
This repository contains the supporting materials for my paper entitled "Robustness Test for AI Forecasting of Hurricane Florence Using FourCastNetv2 and Random Perturbations of the Initial Condition" authored by Adam Lizerbram, Shane Stevenson, Iman Khadir, Matthew Tu, and Samuel S. P. Shen.

# Summary of this Project

# Sample Figures for this Project

# Computer Code for Reproduction for the Figures and Animations of the Paper

# Data Needed for the Reproduction of the Figures

# Computer Code for this Research Project

# (contact me, tutoral, etc)

# References (2 or 3, at most 5)

# Tasks
This repo contains the tasks that I was assigned to complete for the SDSU Climate Informatics Lab (SCIL). My specific tasks included testing NVIDIA’s model FourCastNetv2 (FCNv2) with added randomness to the model’s input in various settings. I was able to provide custom input to FCNv2 by using NVIDIA’s Earth2MIP github repository. I have included instructions on how to download this repo and install all dependencies to be able to utilize it right away.

## Random Forecast

My first task was to generate a forecast using FCNv2 from a completely random input generated from several different distributions, including normal, lognormal, χ², and uniform distributions. This task involved using NVIDIA’s Earth2MIP repo to input randomly generated data into the FourCastNetv2 model. I found that no matter the initial distribution used to generate the random initial conditions, the results are relatively consistent among each variable. For some variables such as zonal wind (u) and meridional wind (v), the model “forgets” the randomness after a few timesteps and produces a smooth forecast. For other variables such as surface pressure (sp), the randomness is seemingly never forgotten, with the forecast continuing to look like random noise for many timesteps; after a while, some patterns in the forecast begin to emerge, but the randomness is still prevalent in many areas. 

### Folders

In the generated_visuals folder, each subfolder corresponds to the distribution that was used to generate the initial conditions. Within each of these subfolders, there is a folder for each variable that has been visualized. For example, there are folders msl (mean sea level), sp (sea pressure), and u10m (wind speed 10m above sea level traveling east/west). These folders contain a gif of the full forecast (120 hours, 1 frame for initial conditions + 20 timesteps) as well as each frame in the animation.

The using_earth2mip folder contains the scripts that I used to generate the predictions, visualize them, and animate them, as well as a few test scripts to make sure everything is working properly. There is also a script which serves as a pipeline, calling all of the other scripts so all steps are performed automatically. There is a pdf in this folder which goes into more detail about setting up your environment to use the Earth2MIP repo and how to use the provided scripts.

## Hurricane Prediction

My second task was to generate hurricane trajectory predictions of hurricane Florence with increasing amounts of noise added to the initial conditions. This task involved again using Earth2MIP to add randomness to the input of FCNv2. However, this time I started with real data of hurricane Florence (9/13/2018) and added increasing amounts of Gaussian noise to see how that would affect the predicted hurricane trajectory. For each noise level, 30 experiments were performed to track the distribution of errors. Error was calculated as the cumulative sum of distances between true and predicted hurricane positions over all timesteps, measured in kilometers. I tracked the hurricane trajectory by finding the position of the lowest mean sea level pressure (msl) at each timestep, which represents the eye of the hurricane. I found that with more noise, the hurricane predictions were still relatively close to the real path in the early stages, but with more timesteps the predictions tended to drift further away from the true trajectory. I noticed that no matter the amount of noise, the generated forecasts from FCNv2 tended to predict the hurricane petering out earlier than the true data, and by the end of the forecast the hurricane was predicted to be almost completely gone despite the true data showing the hurricane still visible by the end of the timeframe. Additionally, from the Random Forecast task, msl seemed to be one of the variables that “forgot” the randomness after a few timesteps and was able to produce a cohesive forecast on a global scale; however, on a local scale, we can observe relatively significant differences as the noise increases.

### Folders

The hurricane_predictions folder contains plots and animations of my results from the noisy hurricane task. Within the plots folder, there are “local,” “global,” and “errors” subfolders. 

The figures within the “local” subfolder show a zoomed in comparison between the predicted hurricane path and the actual hurricane path, showing the mean sea level pressure (msl) values represented by a heatmap and the wind speeds and directions represented by arrows. The calculated location of the hurricane at each timestep is determined by the pixel with the lowest msl value. Each of the figures and animations show the actual data on the top (msl, u10m, v10m) and the predicted data on the bottom. Additionally, the prediction plot is labeled with an amount of noise added to the data. This percentage X represents adding to each lat/lon pair within each channel a random value from a normal distribution with mean 0 and standard deviation X percent of the corresponding channel’s standard deviation. In other words, the noise scales with each channel’s distribution to ensure each channel is proportionally affected equally. Over all experiments for each noise level, the best, median, and worst instances were chosen to be plotted, and can be found in the corresponding subfolders.

The figures within the “global” subfolder show a global view of the forecasts with the added noise, similar to the Random Forecast task, but instead of the input data being 100% random from a specific distribution, a certain amount of random noise is added to real data.

The figures within the “errors” subfolder contain plots of distributions of accumulated error over time for each noise level individually, as well as a plot of distribution of total accumulated error for each noise level. 

The scripts folder contains the scripts that I used to generate, visualize, and animate the predictions, alongside other scripts for testing and for fetching data from the climate data store. Here is a list of each script and its purpose:

| Script                      | Purpose                                                                                                  |
|----------------------------|----------------------------------------------------------------------------------------------------------|
| `compute_error.py`         | Provide helper functions to compute error between predicted and true hurricane trajectories and to update the error log. |
| `config.py`                | Define constants that other scripts use.                                                                 |
| `load_data.py`             | Provide helper functions to load and write data to the disk for other scripts to utilize.               |
| `merge_data.py`            | Combine the fetched data from the climate data store into one dataset which matches the output of FCNv2. |
| `noisy_hurricane.py`       | Provide functions to randomize seeds and generate a forecast with added noise.                          |
| `plot_errors.py`           | Provide functions to plot the calculated errors, both in accumulated error over time for each noise level and total error vs noise level. |
| `retrieve_era5_data.py`    | Fetch data from the climate data store and save to disk.                                                |
| `run_all.py`               | Creates a pipeline to automate generating forecasts, calculating error, and plotting results.            |
| `track_hurricane.py`       | Provides helper functions to track the true and predicted positions of the hurricane and convert indices to latitude and longitude. |
| `visualize_hurricane_global.py` | Plots and animates the data on a global scale.                                                       |
| `visualize_hurricane_local.py` | Plots and animates the data on a local scale, with plotted hurricane trajectory lines for true and predicted data. |

The data folder contains a json file of logged error for each noise level. The architecture of this file is {noise level : {seed : [accumulated error over time]}}. 30 experiments were performed for each noise level with randomized seeds to generate different random noise each time and track the distribution of errors.

# Variables
FourCastNetv2 includes 73 variables in its forecast predictions. Here is a quick guide to what they mean. 
- uXm: Zonal wind at X meters above sea level  in m/s (“u” is east-west wind)
- vXm: Meridional wind at X meters above sea level in m/s (“v” is north-south wind)
- tXm: Temperature at X meters above sea level in K
- sp: Surface pressure in Pa
- msl: Mean sea level pressure in Pa
- tcwv: Total column water vapor, vertically integrated in kg/m²
- uX: Zonal wind at X hPa in m/s
- vX: Meridional wind at X hPa in m/s
- zX: Geopotential height at X hPa 
- tX: Temperature at X hPa in K
- rX: Relative humidity at X hPa as percentage
Note that hPa is the unit to essentially determine the height of the measurement. Here is a table of common pressure levels.

| Pressure (hPa) | Approx Altitude (km) |
|---------------|-----------------------|
| 1000          | 0.1                   |
| 925           | 0.8                   |
| 850           | 1.5                   |
| 700           | 3.0                   |
| 600           | 4.5                   |
| 500           | 5.5                   |
| 400           | 7.0                   |
| 300           | 9.0                   |
| 250           | 10.5                  |
| 200           | 12.0                  |
| 150           | 13.5                  |
| 100           | 16.0                  |
| 50            | 20.0                  |

References:
- https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/neural_operators/fourcastnet.html
- https://docs.api.nvidia.com/nim/reference/nvidia-fourcastnet
