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
    parser.add_argument('-c', '--calibratefile', type=str,
        help=('input pre-existing calibrate file path to HDF5 file'))
    parser.add_argument('-r', '--rainstations', type=str,
            help=('rainstation coords and values in JSON file'))
    parser.add_argument('rainstations', type=str,
                help=('rainstation coords and values in JSON file'))
    return parser

def calibrate(**kwargs):
    # calibrator instance
    calibrator_kwargs = {
        'aggregatefile': kwargs['aggregatefile'],
        'calibratefile': kwargs['calibratefile'],
        'rainstations': kwargs['rainstations'],
        }
    cal = Calibrator(**calibrator_kwargs)

    # perform calibration by interpolation
    interpolate_kwargs = {
        'method': ked,
        'res': kwargs['res'],
        'factor_bounds': kwargs['factor_bounds'],
        'countrymask': config.COUNTRYMASK,
        }
    cal.interpolate(**interpolate_kwargs)
    if cal.result is None:
        interpolate_kwargs['method'] = idw
        cal.interpolate(**interpolate_kwargs)
        if cal.result is None:
            cal.use_aggregate()

    resultfile = kwargs['resultfile']
    # save result to HDF5 file
    cal.save_result(resultfile)


def main():
    # parse command line arguments
    arguments = get_parser().parse_args()
    calibrate(**vars(arguments))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
