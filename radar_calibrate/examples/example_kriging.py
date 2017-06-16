# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate.tests import testutils
from radar_calibrate import gridtools
from radar_calibrate import calibration
from radar_calibrate import kriging_r
from radar_calibrate import plot
from radar_calibrate import utils

import numpy as np

import logging
import os

@utils.timethis
def krige_r(x, y, z, radar, xi, yi, zi):
    return kriging_r.ked_r(x, y, z, radar, xi, yi, zi)


@utils.timethis
def krige_py(x, y, z, radar, xi, yi, zi):
    return calibration.ked(x, y, z, radar, xi, yi, zi)


def test_compare_ked(plot_comparison=False, timestamp='20170305080000'):
    # test data from files
    aggregatefile = r'data\24uur_{}.h5'.format(timestamp)
    calibratefile = r'data\RAD_TF2400_U_{}.h5'.format(timestamp)
    calibrate_kwargs, aggregate, calibrate = testutils.get_testdata(
        aggregatefile,
        calibratefile,
        )
    # NaN mask
    nan_mask = np.isnan(aggregate)

    # ked using R
    timedresult_r = krige_r(**calibrate_kwargs)
    logging.info('ked in R took {dt:.2f} seconds'.format(dt=timedresult_r.dt))
    rain_est_r = timedresult_r.result
    calibrate_r = utils.apply_countrymask(
        rain_est_r.reshape(calibrate.shape), aggregate)
    calibrate_r[nan_mask] = np.nan

    # ked using Python
    timedresult_py = krige_py(**calibrate_kwargs)
    logging.info('ked in python took {dt:.2f} seconds'.format(
        dt=timedresult_py.dt))
    rain_est_py, sigma = timedresult_py.result
    calibrate_py = utils.apply_countrymask(
        rain_est_py.reshape(calibrate.shape), aggregate)
    calibrate_py[nan_mask] = np.nan

    # plot
    if plot_comparison:
        imagefile = os.path.join(testconfig.PLOTDIR,
            'compare_ked_{}.png'.format(timestamp))
        plot.compare_ked(aggregate,
            calibrate, calibrate_r, calibrate_py,
            imagefile=imagefile,
            )

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_compare_ked(plot_comparison=True)
