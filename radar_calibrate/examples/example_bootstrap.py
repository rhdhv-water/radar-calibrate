# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate.tests import testutils
from radar_calibrate.calibration import ked
from radar_calibrate.kriging_r import ked_r
from radar_calibrate.bootstrap import BootStrappedCalibrator

from radar_calibrate import plot
from radar_calibrate import utils

import numpy as np

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
    
    result = test.bootstrap_single(method=ked)
    pdb.set_trace()
    
    # plot
    
#    imagefile = os.path.join(testconfig.PLOTDIR, 'bootstrap.png'.format())
#    results = [timedresults_py, ]
#    plot.timedresults(resizes, results, nstations,
#        imagefile=imagefile, xlim=[0., 5.], ylim=[1., 1000.])
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    bootstrap()
