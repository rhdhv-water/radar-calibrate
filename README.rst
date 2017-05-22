radar-calibrate
===============

Test suite for calibration of rainfall radar images using groundstations

## TODO: Integrate IDW function in kriging.py
## TODO: Insert shapefile NL / loc rainstations in plt
## TODO: Smooth grid to 5000*4900 (M*N)
## TODO: Change function to signature: calibrate_using_idw(source_path, target_path, rain_stations), calibrate_using_kriging(source_path, target_path, rain_stations)
## TODO: Test time performance of IDW and KED functions with grid (100m and 1000m)
## TODO: Pykrige can uses 0 OR 3 parameters (i.e. sill, range and nugget) for variogram
## TODO: Get 30 rainfall events (calibrate.h5) for 5min, 1hour, 1d to test various performances of various KED configs
## TODO: Make tools.py met tools for plotting and grid convertion.

