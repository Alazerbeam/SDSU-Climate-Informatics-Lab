from utils import randomize_seeds
import os

# choose which steps to take
PREDICT = False
PLOT_ERROR_LOCAL = True
PLOT_ERROR_GLOBAL = True
VISUALIZE_LOCAL = False
VISUALIZE_GLOBAL = False
CLEAN_UP = False

# choose which noise levels to test, how many experiments per noise level, and how many timesteps
NOISE_PCTS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
# NOISE_PCTS = [0.35]
NUM_EXPERIMENTS = 30
SEEDS = randomize_seeds(len(NOISE_PCTS) * NUM_EXPERIMENTS)
TIMESTEPS = 15

# choose the directories
PRED_PATH = "/home/jovyan/hurricane_predictions/data/hurricane_run.nc"
TRUE_PATH = "/home/jovyan/hurricane_predictions/data/fcnv2_input.nc"
ERROR_LOG_PATH = "/home/jovyan/hurricane_predictions/data/error_log.json"
ERROR_DATAPATH = "/home/jovyan/hurricane_predictions/data/error_ds.nc"
MODEL_DIR = "/home/jovyan/fcnv2_params"
ERA5_STATS_DIR = "/home/jovyan/era5_stats"
PLOT_DIR = "/home/jovyan/hurricane_predictions/plots"
PRED_PLOT_DIR = os.path.join(PLOT_DIR, "predictions")
ERR_PLOR_DIR = os.path.join(PLOT_DIR, "errors")
LOCAL_PRED_PLOT_DIR = os.path.join(PRED_PLOT_DIR, "local")
GLOBAL_PRED_PLOT_DIR = os.path.join(PRED_PLOT_DIR, "global")
LOCAL_ERR_PLOT_DIR = os.path.join(ERR_PLOR_DIR, "local")
GLOBAL_ERR_PLOT_DIR = os.path.join(ERR_PLOR_DIR, "global")

# set limits on area of local data
x_min = 1080
x_max = 1160
y_min = 200
y_max = 240