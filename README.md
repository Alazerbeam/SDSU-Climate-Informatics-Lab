# Purpose
This repository contains my contributions to the SDSU Climate Informatics Lab (SCIL). 

# Tasks
My first task was to generate random initial conditions to use as input to NVIDIA's model FourCastNetv2 (FCNv2) and generate a forecast with it. Normally you would provide real data, 
but we wanted to discover how FCNv2 handles random input and how long it takes to "forget" the randomness and produce a cohesive forecast. We found that for some variables, the model
forgets the randomness within a couple of iterations, and for other variables, seems to never forget the randomness at all. We tested several different distributions to generate random
initial conditions from (normal, lognormal, chi square, uniform), and the results for each variable were the same across all initial distributions.

# Directory Layout
In the generated_visuals folder, you will find gif animations and pngs of each frame. The layout is as follows: generated_visuals/distribution/variable/frames. The animation is saved under
the variable folder, and each frame is saved in the frames folder. In the using_earth2mip folder, you will find some starter scripts and step-by-step instructions on how to install and
utilize the Earth2MIP repository to run FCNv2 predictions using any custom data you want.

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
Pressure (hPa)           Approx Altitude (km)
    1000                         0.1
     925                         0.8
     850                         1.5
     700                         3.0
     600                         4.5
     500                         5.5
     400                         7.0
     300                         9.0
     250                        10.5
     200                        12.0
     150                        13.5
     100                        16.0
      50                        20.0

References:
https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-sym/user_guide/neural_operators/fourcastnet.html
https://docs.api.nvidia.com/nim/reference/nvidia-fourcastnet
