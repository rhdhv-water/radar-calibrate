# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate import files
from radar_calibrate import plot

import argparse
import logging
import os

def get_parser():
    '''get argumentparser and add arguments'''
    parser = argparse.ArgumentParser(
        'plot radar images',
        )

    # Command line arguments
    parser.add_argument('file', type=str,
        help=('input path to HDF5 file'))
    parser.add_argument('-f', '--imagefile', type=str,
        help=('output calibrate path to HDF5 file'))
    parser.add_argument('--figsize', type=float, nargs=2,
        default=[10., 10.],
        help=('figure size (width, height) [inch]'))
    parser.add_argument('--shapefile', type=str,
        help=('path to background shapefile'))
    parser.add_argument('--vrange', type=float, nargs=2,
        default=[1., 30.],
        help=('value range for image colormap'))
    return parser


def plot_image(**kwargs):
    # read HDF5 file
    h5file = kwargs['file']
    array, basegrid, timestamp = files.read_aggregate(h5file)

    # plot to image file
    imagefile = kwargs.get('imagefile')
    if imagefile is None:
        filepath, ext = os.path.splitext(h5file)
        imagefile = filepath + '.png'

    # get figure size [inch]
    figsize = kwargs.get('figsize')

    # get shapefile from config if not given
    shapefile = kwargs.get('shapefile')
    if shapefile is None:
        shapefile = testconfig.BACKGROUND_SHAPEFILE

    # read features from shapefile
    features = files.read_shape(shapefile, basegrid.get_geotransform())

    # unpack value range
    vrange = kwargs['vrange']
    vmin, vmax = vrange

    plot.image(array,
        imagefile=imagefile,
        figsize=figsize,
        features=features,
        vmin=vmin,
        vmax=vmax,
        title=timestamp.isoformat(),
        )


def main():
    # parse command line arguments
    arguments = get_parser().parse_args()

    # run
    plot_image(**vars(arguments))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
