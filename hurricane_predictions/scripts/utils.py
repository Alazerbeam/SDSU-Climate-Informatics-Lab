import numpy as np
import torch
import random

# create a random list of seeds to use for each experiment
def randomize_seeds(num_seeds):
    seed_list = np.arange(num_seeds)
    np.random.shuffle(seed_list)
    seed_list = [int(seed) for seed in seed_list]
    return seed_list

# set all random seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def units(channel_name):
    if channel_name in ["sp", "msl"]: return "Pa"
    elif channel_name == "tcwv": return "$kg/m^2$"
    elif channel_name[0] in ["u", "v"]: return "m/s"
    elif channel_name[0] == "z": return "$m^2/s^2$"
    elif channel_name[0] == "t": return "K"
    elif channel_name[0] == "r": return "%"
    elif channel_name[0] == "q": return "kg/kg"
    else: return "Unknown units."

# return full name of channel including units to display on plot
def full_name(channel_name):
    if channel_name == "sp": return f"Surface Pressure ({units(channel_name)})"
    elif channel_name == "msl": return f"Mean Sea Level Pressure ({units(channel_name)})"
    elif channel_name == "tcwv": return f"Total Column Water Vapor ({units(channel_name)})"
    elif channel_name[-1] == "m":  # if last character is m, represents meters above surface
        if channel_name[0] == "u": return f"East/West Wind {channel_name[1:-1]}m Above Surface ({units(channel_name)})"
        elif channel_name[0] == "v": return f"North/South Wind {channel_name[1:-1]}m Above Surface ({units(channel_name)})"
        elif channel_name[0] == "t": return f"Temperature {channel_name[1:-1]}m Above Surface ({units(channel_name)})"
        else: return "Unknown channel."
    elif channel_name[0] == "u": return f"East/West Wind at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "v": return f"North/South Wind at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "z": return f"Geopotential at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "t": return f"Temperature at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "r": return f"Relative Humidity at {channel_name[1:]} hPa ({units(channel_name)})"
    elif channel_name[0] == "q": return f"Specific Humidity at {channel_name[1:]} hPa ({units(channel_name)})"
    else: return "Unknown channel."