import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-complete',
    {
        'class': 'ea',
        'date': '2018-09-13/to/2018-09-16',
        'expver': '1',
        'levelist': '1000/925/850/700/600/500/400/300/250/200/150/100/50',  # Pressure levels in hPa
        'levtype': 'pl',  # pressure levels
        'param': '129/130/131/132/157',  # u, v, z, t, r 
        'stream': 'oper',
        'type': 'an',  # analysis
        'grid': '0.25/0.25',  # 0.25-degree resolution
        'time': [
            '00:00', '06:00', '12:00', '18:00',
        ],
        'area': [90, 0, -90, 359.75],  # North, West, South, East (approx Atlantic basin)
        'format': 'netcdf',
    },
    '/home/jovyan/hurricane_predictions/era5_pl.nc')

c.retrieve(
    'reanalysis-era5-complete',
    {
        'class': 'ea',
        'date': '2018-09-13/to/2018-09-16',
        'expver': '1',
        'levtype': 'sfc',  # surface variables
        'param': '134/137/151/165/166/167/228246/228247',  # u10m, v10m, u100m, v100m, t2m, sp, msl, tcwv
        'stream': 'oper',
        'type': 'an',  # analysis
        'grid': '0.25/0.25',  # 0.25-degree resolution
        'time': [
            '00:00', '06:00', '12:00', '18:00',
        ],
        'area': [90, 0, -90, 359.75],  # North, West, South, East (approx Atlantic basin)
        'format': 'netcdf',
    },
    '/home/jovyan/hurricane_predictions/era5_sfc.nc')