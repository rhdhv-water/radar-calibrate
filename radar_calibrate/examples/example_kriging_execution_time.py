# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate.tests import testutils
from radar_calibrate import calibration
from radar_calibrate import plot
from radar_calibrate import utils

import numpy as np

from collections import defaultdict
import logging
import csv
import os

@utils.timethis
def krige_r(x, y, z, radar, xi, yi, zi):
    return kriging_r.ked_r(x, y, z, radar, xi, yi, zi)

@utils.timethis
def krige_py(x, y, z, radar, xi, yi, zi):
    return calibration.ked(x, y, z, radar, xi, yi, zi, backend='vectorized')

def compare_time():
    timestamps=['20170223080000', '20170228080000', '20170305080000']
    resizes = [1., 1.5, 2.]

    # initialize variables
    nstations = {}
    timedresults_py = defaultdict(list)
    # timedresults_r = []

    for timestamp in timestamps:
        logging.info('timestamp = {timestamp:}'.format(timestamp=timestamp))
        # test data from files
        aggregatefile = r'24uur_{}.h5'.format(timestamp)
        calibratefile = r'RAD_TF2400_U_{}.h5'.format(timestamp)
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        for resize in resizes:
            logging.info('resize factor = {resize:.1f}'.format(resize=resize))
            # get test data
            calibrate_kwargs, aggregate, calibrate = testutils.get_testdata(
            aggregatefile,
            calibratefile,
            resize=resize,
            )
            logging.info('interpolate using {:d} rainstations'.format(
                len(calibrate_kwargs['x'])))
            nstations[timestamp] = len(calibrate_kwargs['x'])

            # ked using Python with a resized grid
            try:
                timedresult = krige_py(**calibrate_kwargs)
                logging.info('ked in python took {dt:.2f} seconds'.format(
                        dt=timedresult.dt))
                timedresults_py[timestamp].append(timedresult.dt)
            except MemoryError:
                logging.warning('memory error, passing')
                timedresults_py[timestamp].append(np.nan)

            # ked using R with a resized grid
            # timedresult = krige_r(**calibrate_kwargs)
            # logging.info('ked in R took {dt:.2f} seconds'.format(
            #         dt=timedresult.dt))
            # timedresults_r.append(timedresult.dt)

    # plot
    imagefile = os.path.join(testconfig.PLOTDIR, 'time_ked.png'.format())
    results = [timedresults_py, ]
    plot.timedresults(resizes, results, nstations,
        imagefile=imagefile, xlim=[0., 3.], ylim=[1., 1000.])

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    compare_time()
