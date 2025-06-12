import numpy as np

era5_means = np.load("/home/jovyan/era5_stats/global_means.npy")
era5_stds = np.load("/home/jovyan/era5_stats/global_stds.npy")
print("means shape:", era5_means.shape)
print("stds shape:", era5_stds.shape)