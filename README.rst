radar-calibrate
===============

Test suite for calibration of rainfall radar images using groundstations

To Do:
  - Integrate IDW function in kriging.py
  - Insert shapefile NL / loc rainstations in plt
  - Smooth grid to 5000*4900 (M*N)
  - Change function to signature: calibrate_using_idw(source_path, target_path, rain_stations), calibrate_using_kriging(source_path, target_path, rain_stations)
  - Test time performance of IDW and KED functions with grid (100m and 1000m)
  - Pykrige can uses 0 OR 3 parameters (i.e. sill, range and nugget) for variogram
  - Get 30 rainfall events (calibrate.h5) for 5min, 1hour, 1d to test various performances of various KED configs
  - Make tools.py met tools for plotting and grid convertion.
