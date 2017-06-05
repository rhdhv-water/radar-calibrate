#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

import h5py
import numpy
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from time import time

from calibrate import idw, kriging

file_aggregate = r'..\data\24uur_20170223080000.h5'
file_calibrate = r'..\data\RAD_TF2400_U_20170223080000.h5'

with h5py.File(file_aggregate, 'r') as ds:
    aggregate = numpy.float64(ds['precipitation'][:]).T
    grid_extent = ds.attrs['grid_extent']
    grid_size = ds.attrs['grid_size']
    left, right, top, bottom = grid_extent
    pixelwidth = (right - left) / grid_size[0]
    pixelheight = (bottom - top) / grid_size[1]
with h5py.File(file_calibrate, 'r') as ds:
    calibrate = numpy.float64(ds['image1/image_data']).T # Index is [x][y]
    coords = numpy.array(ds.attrs['cal_station_coords'])
    x = coords[:, 0]
    y = coords[:, 1]
    z = numpy.array(ds.attrs['cal_station_measurements'])

#==============================================================================
# Prep data as numpy arrays, as the required format of the krige modules
#==============================================================================
radar = numpy.array(kriging.get_radar_for_locations(x, y, grid_extent, aggregate, pixelwidth, pixelheight))
xi, yi = kriging.get_grid(aggregate, grid_extent, pixelwidth, pixelheight)
# Initialize smoothing grid
factor_x = 10
factor_y = 10
pixelwidth_smooth = pixelwidth/factor_x
pixelheight_smooth = pixelheight/factor_y
zi_smooth = numpy.empty(numpy.array(aggregate.shape) * 10)
xi_smooth, yi_smooth = kriging.get_grid(zi_smooth, grid_extent, pixelwidth_smooth, pixelheight_smooth)
# Interpolate grid with inter from scipy. 
vals = numpy.reshape(aggregate, (len(aggregate[:][0]) * len(aggregate[:])))
pts = numpy.array([[i,j] for i in xi[:,0] for j in yi[0,:]] )
zi_smooth = inter.griddata(pts, vals, (xi_smooth, yi_smooth), method='linear')

#==============================================================================
# RUN FUNCTIONS
#==============================================================================
tic = time()
rain_est_idw = idw.idw(x, y, z, xi_smooth, yi_smooth)
print("py:" + str(time() - tic) + "seconds")

plt.figure()
f221 = plt.subplot(2, 1, 1)
plt.imshow(aggregate, cmap='rainbow', vmin=0, vmax=40)
plt.ylabel('y-coordinate')
plt.title('aggregate')
f222 = plt.subplot(2, 1, 2, sharex=f221, sharey=f221)
plt.imshow(rain_est_idw, cmap='rainbow', vmin=0, vmax=40)
plt.title('$calibrate_{IDW}$')