# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import config
from radar_calibrate import gridtools
from radar_calibrate import files
from radar_calibrate import kriging
from radar_calibrate import kriging_r
from radar_calibrate import plot
from radar_calibrate import utils

import numpy

import logging
import os

@utils.timethis
def krige_r(x, y, z, radar, xi, yi, zi):
    return kriging_r.ked(x, y, z, radar, xi, yi, zi)


@utils.timethis
def krige_py(x, y, z, radar, xi, yi, zi):
    return kriging.ked_py_v(x, y, z, radar, xi, yi, zi)


def test_compare_ked(plot_comparison=False, timestamp='20170228080000'):
    # test data from files
    aggregatefile = r'data\24uur_{}.h5'.format(timestamp)
    calibratefile = r'data\RAD_TF2400_U_{}.h5'.format(timestamp)
    calibrate_kwargs, aggregate, calibrate = files.get_testdata(
        aggregatefile,
        calibratefile,
        )
    # NaN mask
    nan_mask = numpy.isnan(aggregate)

    # ked using R
    timedresult_r = krige_r(**calibrate_kwargs)
    logging.info('ked in R took {dt:.2f} seconds'.format(dt=timedresult_r.dt))
    calibrate_r = utils.apply_countrymask(
        timedresult_r.result.reshape(calibrate.shape), aggregate)
    calibrate_r[nan_mask] = numpy.nan

    # ked using Python
    timedresult_py = krige_py(**calibrate_kwargs)
    logging.info('ked in python took {dt:.2f} seconds'.format(
        dt=timedresult_py.dt))
    calibrate_py = utils.apply_countrymask(
        timedresult_py.result.reshape(calibrate.shape), aggregate)
    calibrate_py[nan_mask] = numpy.nan

    # plot
    if plot_comparison:
        imagefile = os.path.join(config.PLOTDIR, 'compare_ked_{}.png'.format(
            timestamp))
        plot.compare_ked(aggregate,
            calibrate, calibrate_r, calibrate_py,
            imagefile=imagefile,
            )

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_compare_ked(plot_comparison=True)
