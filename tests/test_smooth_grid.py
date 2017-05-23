#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

import h5py
import numpy
import os
import importlib
from math import floor, ceil, sqrt
import matplotlib.pyplot as plt

import sys
if "kriging" not in sys.modules:
    import kriging
else: 
    importlib.reload(kriging)

def distance(x1, x2, y1, y2):
    dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
    return dist

def smooth_weight(xl, xr, yt, yb, x, y):
    dist_lb = sqrt((xl - x)**2 + (yb - y)**2)
    dist_lt = sqrt((xl - x)**2 + (yt - y)**2)
    dist_rt = sqrt((xr - x)**2 + (yb - y)**2)
    dist_rb = sqrt((xr - x)**2 + (yt - y)**2)
    
    weight = [dist_lb, dist_lt, dist_rt, dist_rb]/sum([dist_lb, dist_lt, dist_rt, dist_rb])
    return weight

root = r'C:\Project_OG\BA8186_NRR\2_technical'
file_aggregate = r'\radar-calibrate\data\24uur_20170223080000.h5'
file_smooth = r'\radar-calibrate\data\24uur_20170223080000_smooth.h5'

with h5py.File(os.path.join(root + file_aggregate), 'r') as ds:
    aggregate = numpy.float64(ds['precipitation'][:]).T # Index is [x][y]
    grid_extent = ds.attrs['grid_extent']
    grid_size = ds.attrs['grid_size']
    left, right, top, bottom = grid_extent
    pixelwidth = (right - left) / grid_size[0]
    pixelheight = (bottom - top) / grid_size[1]
    
#with h5py.File(os.path.join(root + "_smooth_" + file_aggregate), 'w') as ds:
## based on nearest four locations on the grid (each corner)
xi, yi = kriging.get_grid(aggregate, grid_extent, pixelwidth, pixelheight)

# Smoothing factor in both x and y direction
factor_x = 10
factor_y = 10

pixelwidth_smooth = pixelwidth/factor_x
pixelheight_smooth = pixelheight/factor_y
zi_smooth = numpy.empty(numpy.array(aggregate.shape) * 10)
xi_smooth, yi_smooth = kriging.get_grid(zi_smooth, grid_extent, pixelwidth_smooth, pixelheight_smooth)

for i in range(0, len(xi_smooth[:, 1])-10):
    x_left = int(floor(i - factor_x * 0.5) / factor_x)
    x_right = int(ceil((i - factor_x * 0.5) / factor_x))
    for j in range(0, len(xi_smooth[1, :])):
        if j < 0.5 * factor_y: # to treat the boundaries correct
            y_top = 0
            y_bottom = 0
        elif j > len(xi_smooth[1,:]) - 0.5 * factor_y:
            y_top = 489
            y_bottom = 489
        else: 
            y_top = int(floor((j - factor_y * 0.5) / factor_y))
            y_bottom = int(ceil((j - factor_y * 0.5) / factor_y)) 
        
        weight = smooth_weight(xi[x_left, y_bottom], xi[x_right, y_bottom], yi[x_left, y_bottom], yi[x_left, y_top], xi_smooth[i,j], yi_smooth[i,j])
        value = [aggregate[x_left, y_bottom],aggregate[x_left, y_top],aggregate[x_right, y_top],aggregate[x_right, y_bottom]]
        zi_smooth[i,j] = sum(value * weight)
    print(str(i*100/len(xi_smooth[:, 1])) + "%")
zi_smooth_32 = numpy.float32(zi_smooth)

# Write file
with h5py.File(os.path.join(root + file_smooth)) as h5:
    # Precipitation
    dataset = h5.create_dataset(
        'precipitation', zi_smooth_32.shape, numpy.float32, data = zi_smooth_32, fillvalue=-9999,
        compression='gzip', shuffle=True#, chunks=(20, 20, 24)
    )

# Plot radar_calibrate_R
plt.figure()
f221 = plt.subplot(2, 1, 1)
plt.imshow(aggregate, cmap='rainbow', vmin=0, vmax=40)
plt.ylabel('y-coordinate')
plt.title('aggregate')
f222 = plt.subplot(2, 1, 2)
plt.imshow(zi_smooth000, cmap='rainbow', vmin=0, vmax=40)
plt.title('$calibrate_{original}$')