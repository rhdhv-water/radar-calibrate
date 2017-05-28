# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import gridtools
from radar_calibrate import files
from radar_calibrate import kriging
from radar_calibrate import kriging_r
from radar_calibrate import plot
from radar_calibrate import utils

import numpy

import logging


@utils.timethis
def krige_r(x, y, z, radar, xi, yi, zi):
    return kriging_r.ked(x, y, z, radar, xi, yi, zi)


@utils.timethis
def krige_py(x, y, z, radar, xi, yi, zi):
    return kriging.ked_py_v(x, y, z, radar, xi, yi, zi)


def test_ked():
    # test data from files
    aggregatefile = r'data\24uur_20170223080000.h5'
    calibratefile = r'data\RAD_TF2400_U_20170223080000.h5'
    x, y, z, radar, xi, yi, zi, aggregate, calibrate = files.get_testdata(
        aggregatefile,
        calibratefile,
        )

    # ked using R
    timedresult_r = krige_r(x, y, z, radar, xi, yi, zi)
    logging.info('ked in R took {dt:2f} seconds'.format(dt=timedresult_r.dt))
    calibrate_r = timedresult_r.result.reshape(calibrate.shape)

    # ked using Python
    timedresult_py = krige_py(x, y, z, radar, xi, yi, zi)
    logging.info('ked in python took {dt:2f} seconds'.format(
        dt=timedresult_py.dt))
    calibrate_py = timedresult_py.result.reshape(calibrate.shape)

    # plot
    plot.compare_ked(z, radar, aggregate, calibrate, calibrate_r, calibrate_py)


def main():
    test_ked()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
