# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import gridtools
from radar_calibrate import files
from radar_calibrate import kriging
from radar_calibrate import kriging_r
from radar_calibrate import utils
from radar_calibrate import plot
from radar_calibrate import config

import numpy

import logging
import os

@utils.timethis
def krige_r(x, y, z, radar, xi, yi, zi):
    return kriging_r.ked(x, y, z, radar, xi, yi, zi)

@utils.timethis
def krige_py(x, y, z, radar, xi, yi, zi):
    return kriging.ked_py(x, y, z, radar, xi, yi, zi)

def test_compare_grid(plot_comparison=False, timestamp='20170223080000'):
    # test data from files
    aggregatefile = r'data\24uur_{}.h5'.format(timestamp)
    calibratefile = r'data\RAD_TF2400_U_{}.h5'.format(timestamp)

    # initialize variables
    reshapes = [1, 1.5, 2, 4, 5]
    timedresults_py = []
    timedresults_r = []

    for i in range(len(reshapes)):
        # get test data
        calibrate_kwargs, aggregate, calibrate = files.get_testdata(
        aggregatefile,
        calibratefile,
        reshape = reshapes[i],
        )
        # NaN mask
        nan_mask = numpy.isnan(aggregate)

        # ked using Python with a reshaped grid
        timedresult = krige_py(**calibrate_kwargs)
        logging.info('ked in python took {dt:.2f} seconds'.format(
                dt=timedresult.dt))
        timedresults_py.append(timedresult.dt)

        # ked using R with a reshaped grid
        # timedresult = krige_r(**calibrate_kwargs)
        # logging.info('ked in R took {dt:.2f} seconds'.format(
        #         dt=timedresult.dt))
        # timedresults_r.append(timedresult.dt)
        timedresults_r = None

    # plot
    if plot_comparison:
        imagefile = os.path.join(config.PLOTDIR, 'time_ked_{}.png'.format(
            timestamp))
        plot.timedresults(reshapes, timedresults_py, timedresults_r, imagefile=imagefile,)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_compare_grid(plot_comparison=True)
