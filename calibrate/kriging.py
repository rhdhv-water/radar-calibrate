#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

"""
# IN:
len(x) = 407
len(y) = 407
len(z) = 407
len(radar) = 407
len(xi) = 245000
len(yi) = 245000
len(zi) = 245000

# OUT:
shape(rain_est) = 490*500
"""

#==============================================================================
# FUNCTIONS
#==============================================================================

def get_radar_for_locations(rainstation, aggregate, block=2):
    '''
    Radar "pixel"values for location closest to weather station.
    Returns those pixels that are closest to the rain stations
    '''
    # Changes name size to block as size is a function as well and might confuse
    block = block # use number of surrounding pixels
    x = []
    y = []
    z = []
    radar = []
    for i in range(len(rainstation)):
        x.append(rainstation.ix[i][0][0])
        y.append(rainstation.ix[i][0][1])
        z.append(rainstation.ix[i][2])
        xoff = int((x[i] - grid_extent[0]) / pixelwidth)
        yoff = int((y[i] - grid_extent[3]) / pixelheight) # 700000 moet 0 zijn
        data = aggregate[xoff:xoff + block, yoff:yoff + block] # - half block + half block? (i.e. size)
        radar.append(numpy.median(data))
    return x, y, z, radar

def get_grid(aggregate,grid_extent):
    """
    Return x and y coordinates of cell centers.
    """
#    cellwidth, cellheight = aggregate.get_cellsize()
    left, right, top, bottom = grid_extent

    nx = numpy.size(aggregate,0)
    ny = numpy.size(aggregate,1)
    xmin = left + pixelwidth / 2
    xmax = right - pixelwidth / 2
    ymin = bottom + pixelheight / 2
    ymax = top - pixelheight / 2

    xi, yi = numpy.mgrid[ xmin:xmax:nx * 1j, ymax:ymin:ny * 1j,]
    xi = numpy.float32(xi).flatten()
    yi = numpy.float32(yi).flatten()
    return xi, yi
    
def ked_R(x, y, z, radar, xi, yi, zi):
    """
    Run the kriging method using the R module "gstat".
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long. 
    Inputs xi, yi and zi (radar size) should be equally long.
    """
    
    import rpy2.robjects as robj
    robj.r.library('gstat')
    
    # Modification to prevent singular matrix (Tom)
    radar += (1e-9 * numpy.random.rand(len(radar)))

    # Convert data readible to R        
    radar = robj.FloatVector(radar)
    x, y, z = robj.FloatVector(x), robj.FloatVector(y), robj.FloatVector(z)
    xi, yi, zi = robj.FloatVector(xi), robj.FloatVector(yi), robj.FloatVector(zi)
    rain_radar_frame = robj.DataFrame({'x': x, 'y': y, 'z': z,'radar': radar})
    rain_radar_frame_py = numpy.array(rain_radar_frame)
    radar_frame = robj.DataFrame({'x': xi, 'y': yi, 'radar': zi})
    radar_frame_py = numpy.array(radar_frame)
    
    # Create predictor
    vgm = robj.r.variogram(robj.r("z~radar"), 
                           robj.r('~ x + y'),
                           data=rain_radar_frame,
                           cutoff=50000, 
                           width=5000)
    residual_callable = robj.r('fit.variogram')
    residual = residual_callable(vgm, robj.r.vgm(1, 'Sph', 25000, 1))
    residual_py = numpy.array(residual)
    ked = robj.r('NULL')
    ked = robj.r.gstat(ked, 'raingauge', robj.r("z~radar"),
                       robj.r('~ x + y'),
                       data=rain_radar_frame,
                       model=residual, nmax=40)
    
    # Run predictor
    result = robj.r.predict(ked, radar_frame, nsim=0)
    rain_est = numpy.array(result[2])
    
    # Correction 0's and extreme kriged values
    zi = numpy.array(zi)
    zero_or_no_data = numpy.logical_or(zi == 0, zi == -9999)
    correction_factor = numpy.ones(zi.shape)
    correction_factor[~zero_or_no_data] = (rain_est[~zero_or_no_data] / zi[~zero_or_no_data])
    leave_uncalibrated = numpy.logical_or(correction_factor < 0, correction_factor > 10)
    logging.info('Leaving {} extreme pixels uncalibrated.'.format(leave_uncalibrated.sum(),))
    rain_est[leave_uncalibrated] = zi[leave_uncalibrated]
    
    return rain_est

def ked_py(x, y, z, radar, xi, yi, zi):
    """
    Run the kriging method using the Pykrige module "gstat".
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long. 
    Inputs xi, yi and zi (radar size) should be equally long.
    """
    import pykrige
    # Create predictor
    ked = pykrige.UniversalKriging(x, y, z)
    # Run predictor
    y_pred = ked.execute('points', xi, yi,specified_drift_arrays=radar)
    rain_est = numpy.squeeze(y_pred)[0]
    return rain_est

#==============================================================================
# Main
#==============================================================================

import os
import numpy
import pandas
import h5py
import json 
import matplotlib.pyplot as plt
from datetime import datetime
import logging

#==============================================================================
# INPUT
#==============================================================================
root = r'C:\Project_OG\BA8186_NRR\2_technical\radar-calibrate'
file_station = r'\data\2017_grounddata.json'
file_aggregate = r'\data\24uur_20170223080000.h5'
file_calibrate = r'\data\RAD_TF2400_U_20170223080000.h5'
date = '2017-02-23T08:00:00'

#==============================================================================
# Read files
#==============================================================================
os.chdir(root)
with open(root + file_station) as json_data:
    data = json.load(json_data)
with h5py.File(root + file_aggregate, 'r') as ds:
    aggregate = numpy.float64(ds['precipitation'][:]).T # Index is [x][y]
    grid_extent = ds.attrs['grid_extent']
    grid_size = ds.attrs['grid_size']
    pixelwidth = (grid_extent[1] - grid_extent[0]) / grid_size[0]
    pixelheight = (grid_extent[2] - grid_extent[3]) / grid_size[1]
with h5py.File(root + file_calibrate, 'r') as ds:
    calibrate = numpy.float64(ds['image1/image_data']).T # Index is [x][y]

#==============================================================================
# Prep data as numpy arrays, as the required format of the krige modules
#==============================================================================
rainstation = pandas.DataFrame.from_dict(data[date])
x, y, z, radar = get_radar_for_locations(rainstation, aggregate, block=2)
xi, yi = get_grid(aggregate,grid_extent)
zi = aggregate.flatten()

#==============================================================================
# # Run the KED functions
#==============================================================================
start_time = datetime.now()
rain_est_R = ked_R(x, y, z, radar, xi, yi, zi)
calibrate_R = rain_est_R.reshape([500,490]) # TODO: Make general
end_time = datetime.now()
print("It took R",end_time - start_time, "seconds to complete ked with a grid of 1000")

start_time = datetime.now()
rain_est_py = ked_py(x, y, z, radar, xi, yi, zi)
calibrate_py = rain_est_py.reshape([490,500])
end_time = datetime.now()
print("It took py",end_time - start_time, "seconds to complete ked with a grid of 1000")

#==============================================================================
# VISUALISATION
#==============================================================================
#Plot error of rainstations vs radar in (mm)
plt.figure(0)
plt.scatter(z,radar)
plt.plot([0,40],[0,40],'k')
plt.xlabel('$P_{station}\/(mm)$')
plt.ylabel('$P_{radar}\/(mm)$')
plt.axis([0, 40, 0, 40])
plt.show()

# Plot radar_calibrate_R
plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(aggregate, cmap='rainbow', vmin=0, vmax=40)
plt.ylabel('y-coordinate')
plt.title('aggregate')
plt.subplot(2, 2, 2)
plt.imshow(calibrate/100, cmap='rainbow', vmin=0, vmax=40)
plt.title('$calibrate_{original}$')
plt.subplot(2, 2, 3)
plt.imshow(calibrate_R, cmap='rainbow', vmin=0, vmax=40)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('$calibrate_R$')
plt.subplot(2, 2, 4)
plt.imshow(calibrate_R / (calibrate/100), cmap='rainbow', vmin=0, vmax=40)
plt.xlabel('x-coordinate')
plt.title('$calibrate_{py}$')
plt.tight_layout()
plt.show()