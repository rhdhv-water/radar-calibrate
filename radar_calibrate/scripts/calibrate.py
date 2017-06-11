# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate.calibration import Calibrator, ked, idw
from radar_calibrate.kriging_r import ked_r

import argparse
import logging
import os

log = logging.getLogger(os.path.basename(__file__))


def get_parser():
    '''get argumentparser and add arguments'''
    parser = argparse.ArgumentParser(
        'calibrate aggregated radar images using rainstations',
        )

    # Command line arguments
    parser.add_argument('aggregatefile', type=str,
        help=('input aggregate path to HDF5 file'))
    parser.add_argument('-cal', '--calibratefile', type=str,
        help=('input pre-existing calibrate file path to HDF5 file'))
    parser.add_argument('-rain', '--rainstationsfile', type=str,
        help=('rainstation coords and values in JSON file'))
    parser.add_argument('-a', '--areamaskfile', type=str,
        default=testconfig.AREAMASKFILE,
        help=('areamask in HDF5 file'))
    parser.add_argument('-m', '--method', type=str,
        choices=['ked', 'ked_r', 'idw'],
        default='ked',
        help=('interpolation method'))
    parser.add_argument('-c', '--cellsize', type=float, nargs=2,
        default=[1000., 1000.],
        help=('grid cellsize for interpolation'))
    parser.add_argument('-b', '--factor_bounds', type=float, nargs=2,
        default=[0., 10.],
        help=('min-max bounds for calibration correction factor'))
    parser.add_argument('resultfile', type=str,
        help=('output calibrate path to HDF5 file'))
    return parser


def calibrate(**kwargs):
    # calibrator instance
    calibrator_kwargs = {
        'aggregatefile': kwargs['aggregatefile'],
        'calibratefile': kwargs['calibratefile'],
        'rainstationsfile': kwargs['rainstationsfile'],
        'areamaskfile': kwargs['areamaskfile'],
        }
    cal = Calibrator(**calibrator_kwargs)

    # perform calibration by interpolation
    interpolation_methods = {
        'ked': ked,
        'ked_r': ked_r,
        'idw': idw,
        }
    interpolate_kwargs = {
        'method': interpolation_methods[kwargs['method']],
        'to_cellsize': kwargs['cellsize'],
        'factor_bounds': kwargs['factor_bounds'],
        }

    cal.interpolate(**interpolate_kwargs)
    if cal.result is None:
        log.warning('no result, exiting')
        return

    # save result to HDF5 file
    resultfile = kwargs['resultfile']
    cal.save(resultfile)


def main():
    # parse command line arguments
    arguments = get_parser().parse_args()

    # run
    calibrate(**vars(arguments))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
