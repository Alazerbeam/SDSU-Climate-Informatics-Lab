import numpy as np

# convert indices to lat/lon
def index_to_lon(x):
    return x/4-360

def index_to_lat(y):
    return 90 - y/4

index_to_lat_vec = np.vectorize(index_to_lat)
index_to_lon_vec = np.vectorize(index_to_lon)

# find the x and y coord of min pressure
def track_min_pressure(msl_vals, track_x, track_y):
    min_pressure = np.min(msl_vals)
    y_loc = np.where(msl_vals == min_pressure)[0][0]
    x_loc = np.where(msl_vals == min_pressure)[1][0]
    track_x.append(x_loc)
    track_y.append(y_loc)

# create list of indices of min msl positions in true data and predictions
def track_true_and_pred_locs(true_msl_vals, pred_msl_vals):
    track_pred_x = []
    track_pred_y = []
    track_true_x = []
    track_true_y = []
    
    for true_vals, pred_vals in zip(true_msl_vals, pred_msl_vals):
        track_min_pressure(true_vals, track_true_x, track_true_y)
        track_min_pressure(pred_vals, track_pred_x, track_pred_y)
    
    return track_true_x, track_true_y, track_pred_x, track_pred_y