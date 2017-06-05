# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import utils

import numpy as np
import pykrige

import logging

class Interpolator(object):
    def __init__(self, aggregatefile, calibratefile, rainstations):
        # read aggregate

        # geotransform

        # read rainstations
        files.read_rainstations

    def interpolate(self, method, res):
        pass

    def bootstrap(self, method, res):
        pass

    def save(calibratefile):
        pass


def idw(x, y, z, xi, yi, p=2):
    """
    Simple idw function. Slow, but memory efficient implementation.

    Input x, y, z (rainstation size) shoud be equally long.
    Inputs xi, yi (radar size) should be equally long.
    Input p is the power factor.

    Returns calibrated grid (zi)
    """
    sum_of_weights = np.zeros(xi.shape)
    sum_of_weighted_gauges = np.zeros(xi.shape)
    for i in range(x.size):
        distance = np.sqrt((x[i] - xi) ** 2 + (y[i] - yi) ** 2)
        weight = 1.0 / distance ** p
        weighted_gauge = z[i] * weight
        sum_of_weights += weight
        sum_of_weighted_gauges += weighted_gauge
    zi = sum_of_weighted_gauges / sum_of_weights

    return zi


def ked(x, y, z, radar, xi, yi, zi,
    varogram_model="spherical", verbose=False, backend="vectorized"):
    """
    Run the kriging method using the Python module Pykrige using vectorized backend to save time (high memory).
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long and as numpy array.
    Input xi, yi and zi (radar size) should be equally long and as numpy array.
    Input vario will display the variogram.

    Returns calibrated grid
    """
    # create predictor
    ked = pykrige.UniversalKriging(x, y, z,
                                   drift_terms=["specified"],
                                   specified_drift=[radar,],
                                   variogram_model=variogram_model,
                                   verbose=verbose,
                                   )
    # run predictor
    est, sigma = ked.execute('grid', xi, yi,
                         specified_drift_arrays=[zi,],
                         backend=backend,
                         )

    # return prediction result (est) and variance (sigma)
    return est, sigma
