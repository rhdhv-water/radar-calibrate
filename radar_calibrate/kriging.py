# -*- coding: utf-8 -*-
# Royal HaskoningDHV

# import rpy2.robjects as robj
import pykrige

import matplotlib.pyplot as plt
import numpy

import logging


def plot_vgm_R(vgm_py, residual_py):
    figure = plt.figure()
    plt.plot(vgm_py[1,:], vgm_py[0,:], 'r*')
    plt.xlabel('distance')
    plt.ylabel('semivariance')
    plt.title('R variogram')
    return figure


def ked_R(x, y, z, radar, xi, yi, zi, vario=False):
    """
    Run the kriging method using the R module "gstat".
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long.
    Inputs xi, yi and zi (radar size) should be equally long.
    Input vario will display the variogram.

    Returns calibrated grid
    """
    robj.r.library('gstat')

    # Modification to prevent singular matrix (Tom)
    radar += (1e-9 * numpy.random.rand(len(radar)))
    radar = robj.FloatVector(radar)
    x, y, z = robj.FloatVector(x), robj.FloatVector(y), robj.FloatVector(z)
    xi, yi, zi = robj.FloatVector(xi), robj.FloatVector(yi), robj.FloatVector(zi)
    rain_radar_frame = robj.DataFrame({'x': x, 'y': y, 'z': z, 'radar': radar})
    rain_radar_frame_py = numpy.array(rain_radar_frame)
    radar_frame = robj.DataFrame({'x': xi, 'y': yi, 'radar': zi})
    radar_frame_py = numpy.array(radar_frame)

    # Create predictor
    vgm = robj.r.variogram(robj.r("z~radar"),
                           robj.r('~ x + y'),
                           data=rain_radar_frame,
                           cutoff=50000,
                           width=5000)
    vgm_py = numpy.array(vgm)
    residual_callable = robj.r('fit.variogram')
    residual = residual_callable(vgm, robj.r.vgm(1, 'Sph', 25000, 1))
    residual_py = numpy.array(residual)
    if vario == True:
        plot_vgm_R(vgm_py, residual_py).show()

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
#    rain_est[leave_uncalibrated] = zi[leave_uncalibrated]

    leave_uncalibrated = leave_uncalibrated.reshape([500, 490])
    return rain_est

def ked_py_v(x, y, z, radar, xi, yi, zi, vario=False):
    """
    Run the kriging method using the Python module Pykrige using vectorized backend to save time (high memory).
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long and as numpy array.
    Input xi, yi and zi (radar size) should be equally long and as numpy array.
    Input vario will display the variogram.

    Returns calibrated grid
    """
    # Create predictor
    ked = pykrige.UniversalKriging(x, y, z,
                                   drift_terms = ["specified"],
                                   specified_drift = [radar,],
                                   variogram_model = "spherical",
                                   variogram_parameters = {'sill': 80, 'range': 25000, 'nugget': 0},
#                                   variogram_model = 'custom',
#                                   variogram_function([100,50000,0], 5000)
                                   nlags = 10,
                                   verbose = False,

    )
    if vario == True:
        ked.display_variogram_model()
    # Run predictor
    y_pred = ked.execute('points', xi, yi,
                         specified_drift_arrays = [zi,],
                         backend="vectorized",
    )
    rain_est = numpy.squeeze(y_pred)[0]
    return rain_est

def ked_py_l(x, y, z, radar, xi, yi, zi, vario=False):
    """
    Run the kriging method using the Python module Pykrige using loop backend to save memory (long runtime).
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long and as numpy array.
    Input xi, yi and zi (radar size) should be equally long and as numpy array.
    Input vario will display the variogram.

    Returns calibrated grid
    """
    # Create predictor
    ked = pykrige.UniversalKriging(x, y, z,
                                   drift_terms = ["specified"],
                                   specified_drift = [radar,],
                                   variogram_model = "spherical",
                                   variogram_parameters = {'sill': 80, 'range': 25000, 'nugget': 0},
#                                   variogram_model = 'custom',
#                                   variogram_function([100,50000,0], 5000)
                                   nlags = 10,
                                   verbose = False,

    )
    if vario == True:
        ked.display_variogram_model()
    # Run predictor
    y_pred = ked.execute('points', xi, yi,
                         specified_drift_arrays = [zi,],
                         backend="loop",
    )
    rain_est = numpy.squeeze(y_pred)[0]
    return rain_est
