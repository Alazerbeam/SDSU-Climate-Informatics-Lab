import xarray as xr
import numpy as np

surface_ds = xr.open_dataset("/home/jovyan/hurricane_predictions/era5_sfc.nc", engine='netcdf4')
pressure_ds = xr.open_dataset("/home/jovyan/hurricane_predictions/era5_pl.nc", engine='netcdf4')

surface_ds = surface_ds.rename({'valid_time': 'time'})
pressure_ds = pressure_ds.rename({'valid_time': 'time'})

levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

surface_vars = {
    'u10m': surface_ds['u10'],   # or '10u' depending on file
    'v10m': surface_ds['v10'],
    'u100m': surface_ds['u100'],
    'v100m': surface_ds['v100'],
    't2m': surface_ds['t2m'],
    'sp': surface_ds['sp'],
    'msl': surface_ds['msl'],
    'tcwv': surface_ds['tcwv'],
}

pressure_vars = {}
for var in ['u', 'v', 'z', 't', 'r']:
    arr = pressure_ds[var].sel(pressure_level=levels)
    for lev in levels:
        single_level = arr.sel(pressure_level=lev).assign_coords(pressure_level=None).expand_dims("channel")
        pressure_vars[f"{var}{lev}"] = single_level

ordered_vars = [
    surface_vars['u10m'],
    surface_vars['v10m'],
    surface_vars['u100m'],
    surface_vars['v100m'],
    surface_vars['t2m'],
    surface_vars['sp'],
    surface_vars['msl'],
    surface_vars['tcwv'],
]

for var in ['u', 'v', 'z', 't', 'r']:
    for lev in levels:
        ordered_vars.append(pressure_vars[f"{var}{lev}"])

data_array = xr.concat(ordered_vars, dim='channel')
data_array = data_array.transpose('time', 'channel', 'latitude', 'longitude')  # ensure correct order
tensor = data_array.values  # (channel, time, lat, lon) -> (time, channel, lat, lon)
print("tensor shape:", tensor.shape)
print("lat:", data_array.latitude.shape)
print("lon:", data_array.longitude.shape)

channel_names = [
    'u10m', 'v10m', 'u100m', 'v100m', 't2m', 'sp', 'msl', 'tcwv'
] + [f"{var}{lev}" for var in ['u', 'v', 'z', 't', 'r'] for lev in levels]

data_xr = xr.DataArray(
    tensor,
    dims=["time", "channel", "latitude", "longitude"],
    coords={
        "time": surface_ds.time,  # use from your original dataset
        "channel": channel_names,
        "latitude": surface_ds.latitude,
        "longitude": surface_ds.longitude,
    },
    name="fcnv2_input"
)

ds = data_xr.to_dataset()
print(ds)
print("Available channels:", ds.channel.values)

ds.to_netcdf("/home/jovyan/hurricane_predictions/fcnv2_merged_input.nc")
np.save("/home/jovyan/hurricane_predictions/fcnv2_input.npy", tensor)