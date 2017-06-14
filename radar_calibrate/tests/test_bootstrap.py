# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate.calibration import ked
from radar_calibrate.bootstrap import BootStrappedCalibrator

import numpy as np
import os

aggregatefile = r'24uur_20170223080000.h5'
calibratefile = r'RAD_TF2400_U_20170223080000.h5'
aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
cal = BootStrappedCalibrator(
    aggregatefile=aggregatefile,
    calibratefile=calibratefile,
    )
result = cal.bootstrap_single(ked)
