# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate.calibration import Calibrator

import numpy as np
import os


class TestCalibrator(object):
    def test_rainstations_from_file(self):
        aggregatefile = r'24uur_20170223080000.h5'
        rainstationsfile = r'grounddata.json'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        rainstationsfile = os.path.join(testconfig.DATADIR, rainstationsfile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            rainstationsfile=rainstationsfile,
            )
        assert cal.number_of_stations == 0

    def test_rainstations_from_calibrate(self):
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            )
        assert cal.number_of_stations == 87

    def test_radar_for_locations(self):
        pass
    def test_interpolate(self):
        pass
    def test_save(self):
        pass
