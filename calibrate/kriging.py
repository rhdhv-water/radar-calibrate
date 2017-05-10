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
    radar_frame = robj.DataFrame({'x': xi, 'y': yi, 'radar': zi})
    
    # Create predictor
    vgm = robj.r.variogram(robj.r("z~radar"), 
                           robj.r('~ x + y'),
                           data=rain_radar_frame,
                           cutoff=50000, 
                           width=5000)
    residual_callable = robj.r('fit.variogram')
    residual = residual_callable(vgm, robj.r.vgm(1, 'Sph', 25000, 1))
    ked = robj.r('NULL')
    ked = robj.r.gstat(ked, 'raingauge', robj.r("z~radar"),
                       robj.r('~ x + y'),
                       data=rain_radar_frame,
                       model=residual, nmax=40)
    
    # Run predictor
    result = robj.r.predict(ked, radar_frame, nsim=0)
    rain_est = numpy.array(result[2])
#    zero_or_no_data = numpy.logical_or(aggregate == 0, aggregate == -9999)
    
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

#==============================================================================
# INPUT
#==============================================================================
root = r'C:\Project_OG\BA8186_NRR\2_technical\radar-calibrate'
file_station = r'\data\2017_grounddata.json'
file_aggregate = r'\data\24uur_20170223080000.h5'
file_calibrate = r'\data\RAD_TF2400_U_20170223080000.h5'
date = '2017-02-23T08:00:00'
os.chdir(root)

#==============================================================================
# Read files
#==============================================================================
with open(root + file_station) as json_data:
    data = json.load(json_data)
with h5py.File(root + file_aggregate, 'r') as ds:
    aggregate = numpy.float64(ds['precipitation'][:])
    grid_extent = ds.attrs['grid_extent']
    grid_size = ds.attrs['grid_size']
    xstep = (grid_extent[1] - grid_extent[0]) / grid_size[0]
    ystep = (grid_extent[2] - grid_extent[3]) / grid_size[1]
with h5py.File(root + file_calibrate, 'r') as ds:
    calibrate = numpy.float64(ds['image1/image_data'])

#==============================================================================
# Prep data as numpy arrays, as the required format of the krige modules
#==============================================================================
rainstation = pandas.DataFrame.from_dict(data[date])
x = numpy.empty((len(rainstation),0))
y = numpy.empty((len(rainstation),0))
radar = numpy.empty((len(rainstation),0))
for i in range(len(rainstation)):
    x = numpy.append(x,rainstation.ix[i][0][0])
    y = numpy.append(y,rainstation.ix[i][0][1])
    radar = numpy.append(radar,aggregate
                         [int(round((x[i] - grid_extent[0]) / xstep))]
                         [int(round((y[i] - grid_extent[3]) / ystep))]
                        )
z = rainstation.as_matrix(['value']).T[0]

xi = numpy.tile(numpy.arange(grid_extent[0],grid_extent[1],xstep),len(aggregate[:]))
yi = numpy.repeat(numpy.arange(grid_extent[3],grid_extent[2],ystep),len(aggregate[0]))
zi = aggregate.flatten()

#==============================================================================
# # Run the KED functions
#==============================================================================
start_time = datetime.now()
rain_est_R = ked_R(x, y, z, radar, xi, yi, zi)
calibrate_R = rain_est_R.reshape([490,500]) # TODO: Make general
#calibrate_R = 
end_time = datetime.now()
print("It took R",end_time - start_time, "seconds to complete ked with a grid of",xstep)

start_time = datetime.now()
rain_est_py = ked_py(x, y, z, radar, xi, yi, zi)
calibrate_py = rain_est_py.reshape([490,500])
end_time = datetime.now()
print("It took py",end_time - start_time, "seconds to complete ked with a grid of",xstep)

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
plt.imshow(aggregate, cmap='rainbow', vmin=0, vmax=20)
plt.ylabel('y-coordinate')
plt.title('aggregate')
plt.subplot(2, 2, 2)
plt.imshow(calibrate/100, cmap='rainbow', vmin=0, vmax=20)
plt.title('$calibrate_{original}$')
plt.subplot(2, 2, 3)
plt.imshow(calibrate_R, cmap='rainbow', vmin=0, vmax=20)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.title('$calibrate_R$')
plt.subplot(2, 2, 4)
plt.imshow(calibrate_py, cmap='rainbow', vmin=0, vmax=20)
plt.xlabel('x-coordinate')
plt.title('$calibrate_{py}$')
plt.tight_layout()
plt.show()