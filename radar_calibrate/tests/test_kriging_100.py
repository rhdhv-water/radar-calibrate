# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import gridtools
from radar_calibrate import files
from radar_calibrate import kriging
from radar_calibrate import utils

import numpy

import logging
import os

@utils.timethis
def krige_py(x, y, z, radar, xi, yi, zi):
    return kriging.ked_py(x, y, z, radar, xi, yi, zi)

def test_compare_grid(plot_comparison=False, timestamp='20170223080000'):
    # test data from files
    aggregatefile = r'data\24uur_{}.h5'.format(timestamp)
    calibratefile = r'data\RAD_TF2400_U_{}.h5'.format(timestamp)
    calibrate_kwargs, aggregate, calibrate = files.get_testdata(
        aggregatefile,
        calibratefile,
        )
    # NaN mask
    nan_mask = numpy.isnan(aggregate)

    # ked using Python 1000*1000
    timedresult = krige_py(**calibrate_kwargs)
    logging.info('ked in python took {dt:.2f} seconds'.format(
        dt=timedresult.dt))
    rain_est, sigma = timedresult.result
    calibrate = utils.apply_countrymask(
        rain_est.reshape(calibrate.shape), aggregate)
    calibrate[nan_mask] = numpy.nan

    # ked using Python 500*500
    calibrate_kwargs, aggregate, calibrate = files.get_testdata(
        aggregatefile,
        calibratefile,
        reshape = 2,
        )
    timedresult_reshape = krige_py(**calibrate_kwargs)
    logging.info('ked in python took {dt:.2f} seconds'.format(
        (timedresult_reshape.dt))
    rain_est_reshape, sigma = timedresult_reshape.result
#    calibrate_py = utils.apply_countrymask(
#        rain_est_py.reshape(calibrate.shape), aggregate)
#    calibrate_py[nan_mask] = numpy.nan

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_compare_grid(plot_comparison=True)



#import scipy.interpolate as inter
## Initialize smoothing grid
#factor_x = 10
#factor_y = 10
#pixelwidth_smooth = pixelwidth/factor_x
#pixelheight_smooth = pixelheight/factor_y
#zi_smooth = numpy.empty(numpy.array(aggregate.shape) * 10)
#xi_smooth, yi_smooth = kriging.get_grid(zi_smooth, grid_extent, pixelwidth_smooth, pixelheight_smooth)
## Interpolate grid with inter from scipy.
#vals = numpy.reshape(aggregate, (len(aggregate[:][0]) * len(aggregate[:])))
#pts = numpy.array([[i,j] for i in xi[:,0] for j in yi[0,:]] )
#zi_smooth = inter.griddata(pts, vals, (xi_smooth, yi_smooth), method='linear')