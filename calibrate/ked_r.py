#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

"""
# IN:
len(x) = 71
len(y) = 71
len(z) = 71
len(radar) = 71
len(xi) = 245000
len(yi) = 245000
len(zi) = 245000

# OUT:
shape(rain_est) = 490*500
"""


import numpy
import rpy2.robjects as robj

def ked_R(x, y, z, radar, xi, yi, zi):
    """
    Run the kriging method using the R module "gstat".
    """
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