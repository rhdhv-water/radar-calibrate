# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate.calibration import Calibrator, ked, idw
from radar_calibrate.kriging_r import ked_r
from radar_calibrate import gridtools

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
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            )
        x, y, z, radar = cal.get_radar_for_locations()
        assert len(x) == len(y) == len(z) == len(radar) == 87

    def test_interpolate_idw(self):
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            )
        cal.interpolate(method=idw)
        resultfile = os.path.join(testconfig.RESULTDIR,
            'ked_20170223080000.h5')
        
    def test_interpolate_ked(self):
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            )
        cal.interpolate(method=ked)
        resultfile = os.path.join(testconfig.RESULTDIR,
            'ked_20170223080000.h5')
        cal.save(resultfile=resultfile)
        assert (cal.result is not None) and ('calibrate' in cal.result)

    def test_interpolate_ked_r(self):
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            )
        cal.interpolate(method=ked_r)
        resultfile = os.path.join(testconfig.RESULTDIR,
            'ked_r_20170223080000.h5')
        cal.save(resultfile=resultfile)
        assert (cal.result is not None) and ('calibrate' in cal.result)

    def test_interpolate_ked_cellsize_2000(self):
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            )
        cal.interpolate(method=ked, to_cellsize=[2000., 2000.])
        resultfile = os.path.join(testconfig.RESULTDIR,
            'ked_cellsize_2000_20170223080000.h5')
        cal.save(resultfile=resultfile)
        assert (cal.result is not None) and ('calibrate' in cal.result)

    def test_interpolate_ked_r_cellsize_2000(self):
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            )
        cal.interpolate(method=ked_r, to_cellsize=[2000., 2000.])
        resultfile = os.path.join(testconfig.RESULTDIR,
            'ked_r_cellsize_2000_20170223080000.h5')
        cal.save(resultfile=resultfile)
        assert (cal.result is not None) and ('calibrate' in cal.result)

    def test_interpolate_ked_cellsize_1000_from_100(self):
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            areamaskfile=testconfig.AREAMASKFILE,
            )

        # resample to 100 x 100 cellsize
        cal.resample(to_cellsize=[100., 100.])

        # interpolate
        cal.interpolate(method=ked,
            to_cellsize=[1000., 1000.],
            )

        # resample back to 1000 x 1000 cellsize
        cal.resample(to_cellsize=[1000., 1000.])

        # save
        resultfile = os.path.join(testconfig.RESULTDIR,
            'ked_cellsize_1000_from_100_20170223080000.h5')
        cal.save(resultfile=resultfile)
        assert (cal.result is not None) and ('calibrate' in cal.result)

    def test_interpolate_ked_r_cellsize_1000_from_100(self):
        aggregatefile = r'24uur_20170223080000.h5'
        calibratefile = r'RAD_TF2400_U_20170223080000.h5'
        aggregatefile = os.path.join(testconfig.DATADIR, aggregatefile)
        calibratefile = os.path.join(testconfig.DATADIR, calibratefile)
        cal = Calibrator(
            aggregatefile=aggregatefile,
            calibratefile=calibratefile,
            areamaskfile=testconfig.AREAMASKFILE,
            )

        # resample to 100 x 100 cellsize
        cal.resample(to_cellsize=[100., 100.])

        # interpolate
        cal.interpolate(method=ked_r,
            to_cellsize=[1000., 1000.],
            )

        # resample back to 1000 x 1000 cellsize
        cal.resample(to_cellsize=[1000., 1000.])

        # save
        resultfile = os.path.join(testconfig.RESULTDIR,
            'ked_r_cellsize_1000_from_100_20170223080000.h5')
        cal.save(resultfile=resultfile)
        assert (cal.result is not None) and ('calibrate' in cal.result)
