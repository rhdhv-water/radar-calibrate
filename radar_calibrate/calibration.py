# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import files
from radar_calibrate import utils

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
        self.result = None

    def __repr__(self):
        return 'Calibrator(aggregatefile={aggregatefile:})'.format(
            aggregatefile=self.aggregatefile,
            )

    def get_radar_for_locations(self):
        # station coordinates and values
        station_coords, station_values = self.rainstations

        # select measurements which are equal or larger than zero
        rain_ge_zero = station_values >= 0.
        station_coords = station_coords[rain_ge_zero, :]
        station_values = station_values[rain_ge_zero]

        # sample aggregate at calibration station coordinates
        radar_values = gridtools.sample_array(
            coords=station_coords,
            array=self.aggregate,
            geotransform=self.basegrid.get_geotransform(),
            )
        radar = np.array([v for v in radar_values])

        # unselect measurement if radar is masked
        radar_nan = np.isnan(radar)
        station_coords = station_coords[~radar_nan, :]
        station_values = station_values[~radar_nan]
        radar = radar[~radar_nan]

        # unpack and rename coordinates and values
        x = station_coords[:, 0]
        y = station_coords[:, 1]
        z = station_values

        return x, y, z, radar

    def interpolate(self, method, resolution=None, factor_bounds=None,
        countrymask=None, **interpolate_kwargs):
        # values for rainstations, drop where radar is NaN
        x, y, z, radar = self.get_radar_for_rainstations()

        # interpolation grid at desired resolution
        xi, yi = self.basegrid.get_grid()
        zi = self.aggregate
        xi_interp, yi_interp = self.basegrid.get_grid(resolution)
        zi_interp = gridtools.resample(xi, yi, zi, xi_interp, yi_interp)

        # create mask
        if countrymask is not None:
            mask = np.logical_or(zi_interp.mask, ~countrymask.astype(np.bool))
        else:
            mask = zi_interp.mask

        interpolate_kwargs.update({
            'x': x,
            'y': y,
            'z': z,
            'radar': radar,
            'xi': xi_interp,
            'yi': yi_interp,
            'zi': zi_interp,
            'mask': mask,
            })

        # run interpolation method
        logging.info('interpolate using {method.__name__:}'.format(
            method=method))
        timed_method = utils.timethis(method)
        try:
            timed_result = timed_method(**interpolate_kwargs)
        except:
            logging.warning('interpolation failed')
            pass
        logging.info('interpolation took {dt:2.f} seconds'.format(
            dt=timed_result.dt,
            ))

        # set result to self
        dt, result = timed_result.dt, timed_result.result
        est, sigma, params = result
        self.result = {
            'method': method.__name__,
            'dt': dt,
            'est': est,
            'sigma': sigma,
            'params': params,
            }

        # apply correction factor
        gt_zero = np.logical_and(~zi_interp.mask, zi_interp > 0.)
        factor = np.ones(est.shape)
        factor[gt_zero] = est[gt_zero] / zi_interp[gt_zero]

        # resample factor to aggregate resolution
        factor = gridtools.resample(xi_interp, yi_interp, factor, xi, yi)

        # apply factor bounds
        min_factor, max_factor = factor_bounds
        leave_uncalibrated = np.logical_or(
            self.aggregate.mask, factor < min_factor, factor > max_factor
        )

        # apply factor to aggregate
        calibrate = np.copy(self.aggregate)
        calibrate[~leave_uncalibrated] = calibrate * factor

        # apply country mask border (gradual)
        calibrate = (
            countrymask * calibrate + (1 - countrymask) * self.aggregate)

        self.result.update({
            'calibrate': calibrate,
            })

    def use_aggregate(self):
        '''use aggregate directly without modification'''
        logging.warning('using aggregate as calibrate')
        self.result = {
            'method': 'use aggregate',
            'calibrate': self.aggregate,
            }

    def save(resultfile, attrs=None, save_sigma=False):
        '''save calibration result to file'''
        if self.result is None:
            logging.warning('Calibrator does not contain result, passing')
            pass

        # update attributes with calibration result
        attrs.update({
            'interpolation_method': self.result.get('method'),
            'interpolation_time_seconds': self.result.get('dt'),
            'interpolation_parameters': self.result.get('params'),
            })

        # save to file
        files.save_result(resultfile, self.calibrate, attrs)

        # save sigma (prediction variance) to separate file?
        if save_sigma:
            raise NotImplementedError('what to do')


def ked(x, y, z, radar, xi, yi, zi,
    mask=None, variogram_model='spherical', nlags=40,
    weight=False, verbose=False, backend='vectorized'):
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
                                   drift_terms=['specified'],
                                   specified_drift=[radar,],
                                   variogram_model=variogram_model,
                                   nlags=nlags,
                                   weight=weight,
                                   verbose=verbose,
                                   )
    # run predictor
    est, sigma = ked.execute('grid', xi, yi, mask,
                         specified_drift_arrays=[zi,],
                         backend=backend,
                         )

    # get optimized parameter values (order: sill, range, nugget)
    params = ked.variogram_model_parameters

    # return prediction result (est), variance (sigma) and parameters (params)
    return est, sigma, params


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
