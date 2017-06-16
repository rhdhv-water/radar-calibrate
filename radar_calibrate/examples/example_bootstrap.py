# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate.tests import testutils
from radar_calibrate.calibration import ked,idw
from radar_calibrate.kriging_r import ked_r
from radar_calibrate.bootstrap import BootStrappedCalibrator

from radar_calibrate import plot

import logging
import os

import pdb

def bootstrap():
    timestamp='20170223080000'

    # initialize variables
    logging.info('timestamp = {timestamp:}'.format(timestamp=timestamp))
    
    # test data from files
    aggregatefile = r'24uur_{}.h5'.format(timestamp)
    calibratefile = r'RAD_TF2400_U_{}.h5'.format(timestamp)
    aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
    calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
    test = BootStrappedCalibrator(aggregatefile=aggregatefile,calibratefile=calibratefile)
    
    # run bootstrap on BootStrappedCalibrator object
    result = test.bootstrap_single(method=ked_r)

    # plot
    imagefile = os.path.join(testconfig.PLOTDIR, 'bootstrap_{ts}.png'.format(ts=timestamp))
    plot.bootstrap(result, imagefile=imagefile, zrange=(-10,10))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    bootstrap()
