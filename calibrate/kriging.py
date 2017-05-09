#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

"""
# IN:
len(x) = 42
len(y) = 42
len(z) = 42
len(radar) = 42
len(xi) = 245000
len(yi) = 245000
len(zi) = 245000

# OUT:
shape(rain_est) = 490*500
"""


import numpy

def ked_R(x, y, z, radar, xi, yi, zi):
    """
    Run the kriging method using the R module "gstat".
    """
    import rpy2.robjects as robj
    robj.r.library('gstat')
    # Convert data readible to R
    radar = robj.FloatVector(radar)
    x, y, z = robj.FloatVector(x), robj.FloatVector(y), robj.FloatVector(z)
    rain_radar_frame = robj.DataFrame({'x': x, 'y': y, 'z': z,'radar': radar})
    radar_frame = robj.DataFrame({'x': xi, 'y': yi, 'radar': rxi})
    
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
    return rain_est

def ked_Py(x, y, z, radar, xi, yi, zi):
    """
    Run the kriging method using the python module "Pykrige".
    """
    import pykrige
    
    ked = pykrige.UniversalKriging(x, y, z)
    y_pred = ked.execute('grid', xi, yi)

    rain_est = numpy.squeeze(y_pred)
    return rain_est

import os
import numpy
import pandas
import h5py
import matplotlib.pyplot as plt
import json 

root = r'C:\Project_OG\BA8186_NRR\2_technical\radar-calibrate'
file_json = r'\data\2017_grounddata.json'
file_h5 = r'\data\24uur_20170223080000.h5'
os.chdir(root)


# Open files 
with open(root + file_json) as json_data:
    data = json.load(json_data)
with h5py.File(root+file_h5) as ds:
    precip = numpy.array(ds['precipitation'][:])

# Prep data as numpy arrays, as the required format of the krige modules
rainstation = pandas.DataFrame.from_dict(data['2017-02-23T08:00:00'])
coords = rainstation.as_matrix(['coords'])
#x = 
#y = 
z = rainstation.as_matrix(['value']).T

#xi = 
#yi = 
zi = precip.ravel()

