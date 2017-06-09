# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.calibration import Calibrator, ked, idw

import argparse
import logging


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
    parser.add_argument('-rain', '--rainstations', type=str,
        help=('rainstation coords and values in JSON file'))
    parser.add_argument('-cs', '--cellsize', type=tuple,
        help=('grid cellsize for interpolation'))
    parser.add_argument('-fbnd', '--factor_bounds', type=tuple,
        help=('min-max bounds for calibration correction factor'))
    parser.add_argument('resultfile', type=str,
        help=('output calibrate path to HDF5 file'))
    return parser


def calibrate(**kwargs):
    # calibrator instance
    calibrator_kwargs = {
        'aggregatefile': kwargs['aggregatefile'],
        'calibratefile': kwargs['calibratefile'],
        'rainstations': kwargs['rainstations'],
        }
    cal = Calibrator(**calibrator_kwargs)

    # read mask
    if kwargs.get('areamaskfile'):
        areamask = files.read_mask(kwargs['areamaskfile'])

    # perform calibration by interpolation
    interpolate_kwargs = {
        'method': ked,
        'cellsize': kwargs['cellsize'],
        'factor_bounds': kwargs['factor_bounds'],
        'areamask': areamask,
        }
    cal.interpolate(**interpolate_kwargs)
    if cal.result is None:
        interpolate_kwargs['method'] = idw
        cal.interpolate(**interpolate_kwargs)
        if cal.result is None:
            cal.use_aggregate()

    # save result to HDF5 file
    resultfile = kwargs['resultfile']
    cal.save_result(resultfile)


def main():
    # parse command line arguments
    arguments = get_parser().parse_args()

    # run
    calibrate(**vars(arguments))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
