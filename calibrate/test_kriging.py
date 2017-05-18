#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

#==============================================================================
# Main
#==============================================================================
import h5py
import json

import numpy
import matplotlib.pyplot as plt

import os
from time import time
import logging

import kriging
plt.close('all')

#==============================================================================
# INPUT
#==============================================================================
root = r'C:\Project_OG\BA8186_NRR\2_technical'
file_station = r'\radar-calibrate\data\2017_grounddata.json'
file_aggregate = r'\radar-calibrate\data\24uur_20170223080000.h5'
file_calibrate = r'\radar-calibrate\data\RAD_TF2400_U_20170223080000.h5'
date = '2017-02-23T08:00:00'

#==============================================================================
# Read files
#==============================================================================
with open(os.path.join(root + file_station)) as json_data:
    data = json.load(json_data)
with h5py.File(os.path.join(root + file_aggregate), 'r') as ds:
    aggregate = numpy.float64(ds['precipitation'][:]).T # Index is [x][y]
    grid_extent = ds.attrs['grid_extent']
    grid_size = ds.attrs['grid_size']
    left, right, top, bottom = grid_extent
    pixelwidth = (right - left) / grid_size[0]
    pixelheight = (bottom - top) / grid_size[1]
with h5py.File(os.path.join(root + file_calibrate), 'r') as ds:
    calibrate = numpy.float64(ds['image1/image_data']).T # Index is [x][y]
    coords = numpy.array(ds.attrs['cal_station_coords'])
    x = coords[:, 0]
    y = coords[:, 1]
    z = numpy.array(ds.attrs['cal_station_measurements'])

#==============================================================================
# Prep data as numpy arrays, as the required format of the krige modules
#==============================================================================
rainstation = numpy.array(data[date])
radar = kriging.get_radar_for_locations(x, y, grid_extent, aggregate, pixelwidth, pixelheight)
numpy.array(radar)
xi, yi = kriging.get_grid(aggregate, grid_extent, pixelwidth, pixelheight)
zi = aggregate.flatten()

#==============================================================================
# # Run the KED functions
#==============================================================================
tic = time()
rain_est_R = kriging.ked_R(x, y, z, radar, xi, yi, zi, True)
calibrate_R = rain_est_R.reshape(aggregate.shape)
print("R:" + str(time() - tic) + "seconds")

tic = time()
rain_est_py = kriging.ked_py(x, y, z, radar, xi, yi, zi, True)
calibrate_py = rain_est_py.reshape(aggregate.shape)
print("py:" + str(time() - tic) + "seconds")

#==============================================================================
# VISUALISATION
#==============================================================================
#Plot error of rainstations vs radar in (mm)
plt.figure()
plt.scatter(z, radar)
plt.plot([0, 40], [0, 40], 'k')
plt.xlabel('$P_{station}\/(mm)$')
plt.ylabel('$P_{radar}\/(mm)$')
plt.axis([0, 40, 0, 40])
plt.show()

# Plot radar_calibrate_R
plt.figure()
f221 = plt.subplot(2, 2, 1)
plt.imshow(aggregate, cmap='rainbow', vmin=0, vmax=40)
plt.ylabel('y-coordinate')
plt.title('aggregate')
f222 = plt.subplot(2, 2, 2, sharex=f221, sharey=f221)
plt.imshow(calibrate/100, cmap='rainbow', vmin=0, vmax=40)
plt.title('$calibrate_{original}$')
f223 = plt.subplot(2, 2, 3, sharex=f221, sharey=f221)
plt.imshow(calibrate_R, cmap='rainbow', vmin=0, vmax=40)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('$calibrate_R$')
plt.subplot(2, 2, 4, sharex=f221, sharey=f221)
plt.imshow(calibrate_py, cmap='rainbow', vmin=0, vmax=40)
plt.xlabel('x-coordinate')
plt.title('$calibrate_{py}$')
plt.tight_layout()
plt.show()

# Plot histgram error
plt.figure()
plt.subplot(2, 1, 1)
plt.hist(calibrate_R.flatten() - calibrate.flatten(), bins=range(-10, 10, 1))
plt.xlim([-10, 10])
plt.title('$histogram\/\/of\/\/error\/\/ked_R$')
plt.subplot(2, 1, 2)
plt.hist(calibrate_py.flatten() - calibrate.flatten(), bins=range(-10, 10, 1))
plt.xlim([-10, 10])
plt.title('$histogram\/\/of\/\/error\/\/ked_{py}$')
plt.tight_layout()
plt.show()
