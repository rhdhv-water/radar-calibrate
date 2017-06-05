# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import files

import numpy as np
import pykrige

import logging
import os

class Calibrator(object):
    def __init__(self, aggregatefile, calibratefile=None, rainstations=None):
        # read aggregate
        self.aggregatefile = os.path.basename(aggregatefile)
        self.aggregate, self.basegrid = files.read_aggregate(
            aggregatefile)

        # read calibrate
        if calibratefile is not None:
            self.calibratefile = calibratefile
            self.calibrate, self.rainstations = files.read_calibrate(
                calibratefile)
        else:
            self.calibrate = None
            self.rainstations = None

        # read rainstations
        if rainstations is not None:
            self.rainstations = files.read_rainstations(rainstations)
        else:
            self.rainstations = None

        # initialize result
        self.calibrate = None

    def __repr__(self):
        return 'Calibrator(aggregatefile={aggregatefile:})'.format(
            aggregatefile=self.aggregatefile,
            )

    def interpolate(self, method, res, **interpolate_kwargs):
        logging.info('interpolate using {method.__name__:}'.format(
            method=method))

        # values for rainstations, drop where radar is NaN
        x, y, z, radar = self.get_radar_for_rainstations()

        # interpolation grid at desired resolution
        xi, yi = self.basegrid.get_grid()
        zi = self.aggregate.flatten()
        xi_interp, yi_interp = self.basegrid.get_grid(res)
        zi_interp = gridtools.resample(xi, yi, zi, xi_interp, yi_interp)

        interpolate_kwargs.update({
            'x': x,
            'y': y,
            'z': z,
            'radar': radar,
            'xi': xi_interp,
            'yi': yi_interp,
            'zi': zi_interp,
            })

        # run interpolation method and set result to self
        result = method(**interpolate_kwargs)

        #
        rain_est = self.result[0]
        self.calibrate = self.apply_factor(prediction)


    def bootstrap(self, method, res):
        pass

    def apply_mask(self, mask):
        masked = mask * self.calibrate + (1 - mask) * self.aggregate
        self.calibrate = masked

    def save(calibratefile):
        if self.calibrate is None:
            logging.warning('Calibrator does not contain result, passing')
            pass
        files.save_calibrate(calibratefile, attrs)


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
