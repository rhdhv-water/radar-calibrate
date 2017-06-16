# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import files
from radar_calibrate import gridtools
from radar_calibrate import utils

from rasterio.enums import Resampling
import numpy as np
import pykrige

import logging
import os

import pdb

log = logging.getLogger(os.path.basename(__file__))


class Calibrator(object):
    def __init__(self, aggregatefile,
        calibratefile=None, rainstationsfile=None, areamaskfile=None):
        # read aggregate
        self.aggregatefile = os.path.basename(aggregatefile)
        self.aggregate, self.basegrid, self.timestamp = files.read_file(
            aggregatefile)

        # read rainstations from calibrate calibrate
        if calibratefile is not None:
            self.rainstationsfile = os.path.basename(calibratefile)
            self.rainstations = files.rainstations_from_calibrate(
                calibratefile)
        else:
            self.rainstationsfile = None
            self.rainstations = None

        # read rainstations from JSON
        if rainstationsfile is not None:
            self.rainstationsfile = os.path.basename(rainstationsfile)
            self.rainstations = files.read_rainstations(
                rainstationsfile=rainstationsfile,
                timestamp=self.timestamp,
                )

        # read mask
        if areamaskfile is not None:
            self.areamaskfile = os.path.basename(areamaskfile)
            self.areamask = files.read_mask(
                maskfile=areamaskfile,
                )
        else:
            self.areamaskfile = None
            self.areamask = None

        # initialize result
        self.result = None

    def __repr__(self):
        return ('Calibrator('
            'aggregatefile={aggregatefile:}, '
            'rainstationsfile={rainstationsfile:}, '
            'areamaskfile={areamaskfile:})').format(
            aggregatefile=self.aggregatefile,
            rainstationsfile=self.rainstationsfile,
            areamaskfile=self.areamaskfile,
            )

    @property
    def number_of_stations(self):
        return len(self.rainstations[0])

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
            array=self.aggregate.filled(np.nan),
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

    def resample(self, to_cellsize):
        self.aggregate = gridtools.resample(self.aggregate,
            basegrid=self.basegrid,
            to_cellsize=to_cellsize,
            resampling=Resampling.nearest,
            )
        if self.areamask is not None:
            self.areamask = gridtools.resample(self.areamask,
                basegrid=self.basegrid,
                to_cellsize=to_cellsize,
                resampling=Resampling.nearest,
                )
        if (self.result is not None) and ('calibrate' in self.result):
            self.result['calibrate'] = gridtools.resample(
                self.result['calibrate'],
                basegrid=self.basegrid,
                to_cellsize=to_cellsize,
                resampling=Resampling.nearest,
                )
        self.basegrid = self.basegrid.rescale(to_cellsize=to_cellsize)


    def interpolate(self, method,
        to_cellsize=None, factor_bounds=(0., 10.), **interpolate_kwargs):
        # values for rainstations, drop where radar is NaN
        x, y, z, radar = self.get_radar_for_locations()
        
        # interpolation grid at desired to_cellsize
        xi, yi = self.basegrid.get_grid()
        zi = self.aggregate
        if to_cellsize is not None:
            grid_interp = self.basegrid.rescale(to_cellsize=to_cellsize)
            xi_interp, yi_interp = grid_interp.get_grid()
            timed_downsample = utils.timethis(gridtools.resample)
            resampled_aggregate = timed_downsample(zi,
                basegrid=self.basegrid,
                to_cellsize=to_cellsize,
                resampling=Resampling.average,
                )
            log.info(('downsampling aggregate '
                'to cellsize {cellsize[0]:.2f} x {cellsize[1]:.2f} '
                'took {dt:.2f} seconds').format(
                cellsize=to_cellsize, dt=resampled_aggregate.dt))
            zi_interp = resampled_aggregate.result

            if self.areamask is not None:
                resampled_areamask = timed_downsample(zi,
                    basegrid=self.basegrid,
                    to_cellsize=to_cellsize,
                    resampling=Resampling.average,
                    )
                log.info(('downsampling areamask '
                    'to cellsize {cellsize[0]:.2f} x {cellsize[1]:.2f} '
                    'took {dt:.2f} seconds').format(
                    cellsize=to_cellsize, dt=resampled_areamask.dt))
                areamask_interp = resampled_areamask.result
            else:
                areamask_interp = None
        else:
            xi_interp = xi
            yi_interp = yi
            zi_interp = zi
            areamask_interp = self.areamask

        # create mask
        if areamask_interp is not None:
            mask_interp = np.logical_or(
                zi_interp.mask,
                ~areamask_interp.astype(np.bool),
                )
        else:
            mask_interp = zi_interp.mask

        # unmask before interpolation
        zi_interp = np.ma.filled(zi_interp, fill_value=np.nan)

        # collect keyword arguments
        interpolate_kwargs.update({
            'x': x,
            'y': y,
            'z': z,
            'radar': radar,
            'xi': xi_interp,
            'yi': yi_interp,
            'zi': zi_interp,
            'mask': mask_interp,
            })
        
        # run interpolation method
        pdb.set_trace()
        log.info('interpolate using {method.__name__:}'.format(
            method=method))
        timed_method = utils.timethis(method)
        try:
            timed_result = timed_method(**interpolate_kwargs)
        except:
            log.warning('interpolation failed')
            return
        log.info('interpolation took {dt:.2f} seconds'.format(
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

        # re-apply mask to zi
        zi_interp = np.ma.masked_invalid(zi_interp)

        # apply correction factor
        gt_zero = zi_interp > 0.
        factor = np.ma.where(gt_zero, self.result['est'] / zi_interp, 1.)

        # resample factor to aggregate resolution
        if to_cellsize is not None:
            orig_cellsize = self.basegrid.get_cellsize()
            timed_upsample = utils.timethis(gridtools.resample)
            resampled_factor = timed_upsample(factor,
                basegrid=grid_interp,
                to_cellsize=orig_cellsize,
                resampling=Resampling.average,
                )
            log.info(('upsampling factor '
                'to cellsize {cellsize[0]:.2f} x {cellsize[1]:.2f} '
                'took {dt:.2f} seconds').format(
                cellsize=to_cellsize, dt=resampled_factor.dt))
            factor = resampled_factor.result

        # apply mask and factor bounds
        min_factor, max_factor = factor_bounds
        leave_uncalibrated = np.ma.logical_or(
            self.aggregate.mask,
            factor < min_factor,
            factor > max_factor,
            )
        log.info('leaving {:d} extreme pixels uncalibrated'.format(
            leave_uncalibrated.sum(),
        ))

        # apply factor to aggregate
        calibrate = np.ma.where(
            leave_uncalibrated,
            self.aggregate,
            self.aggregate * factor,
            )

        # apply country mask border (gradual)
        if self.areamask is not None:
            calibrate = (
            self.areamask * calibrate + (1 - self.areamask) * self.aggregate
            )

        self.result.update({
            'calibrate': calibrate,
            })

    def use_aggregate(self):
        '''use aggregate directly without modification'''
        log.warning('using aggregate as calibrate')
        self.result = {
            'method': 'use aggregate',
            'calibrate': self.aggregate,
            }

    def save(self, resultfile, attrs=None, add_sigma=False):
        '''save calibration result to file'''
        if self.result is None:
            log.warning('Calibrator does not contain result, passing')
            pass
        log.info('saving result as {file:}'.format(
            file=os.path.basename(resultfile))
            )

        # update attributes with calibration result
        attrs = attrs or {}
        attrs.update({
            'interpolation_method': self.result.get('method'),
            'interpolation_time_seconds': self.result.get('dt'),
            'interpolation_parameters': self.result.get('params'),
            'grid_extent': self.basegrid.extent,
            'grid_size': self.basegrid.size,
            'timestamp_last_composite': (self.timestamp
                .strftime('%Y%m%d%H%M%S')
                .upper()),
            })

        # save to file
        calibrate = self.result['calibrate']
        if add_sigma and 'sigma' in self.result:
            sigma = self.result['sigma']
        else:
            sigma = None
        files.save_result(resultfile, calibrate, sigma, attrs)


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


def idw(x, y, z, radar, xi, yi, zi,
    mask=None, p=2):
    pdb.set_trace()
    """
    Simple idw function. Slow, but memory efficient implementation.

    Input x, y, z (rainstation size) shoud be equally long.
    Inputs xi, yi (radar size) should be equally long.
    Input p is the power factor.

    Returns calibrated grid (zi)
    """
    xi, yi = np.meshgrid(xi, yi)
    sum_of_weights = np.zeros(xi.shape)
    sum_of_weighted_gauges = np.zeros(xi.shape)
    for i in range(x.size):
        distance = np.sqrt((x[i] - xi) ** 2 + (y[i] - yi) ** 2)
        weight = 1.0 / distance ** p
        weighted_gauge = z[i] * weight
        sum_of_weights += weight
        sum_of_weighted_gauges += weighted_gauge
    zi = sum_of_weighted_gauges / sum_of_weights
    est = zi
    sigma = []
    params = []
    return est, sigma, params
