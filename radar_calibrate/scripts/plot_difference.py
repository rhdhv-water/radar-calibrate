# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate import files
from radar_calibrate import plot

import numpy as np

import argparse
import logging
import os

def get_parser():
    '''get argumentparser and add arguments'''
    parser = argparse.ArgumentParser(
        'plot difference of two radar images',
        )

    # Command line arguments
    parser.add_argument('firstfile', type=str,
        help=('input path to first HDF5 file (first - second)'))
    parser.add_argument('secondfile', type=str,
            help=('input path to second HDF5 file (first - second)'))
    parser.add_argument('-f', '--imagefile', type=str,
        help=('output calibrate path to HDF5 file'))
    parser.add_argument('--figsize', type=float, nargs=2,
        default=[10., 10.],
        help=('figure size (width, height) [inch]'))
    parser.add_argument('--shapefile', type=str,
        help=('path to background shapefile'))
    parser.add_argument('--vrange', type=float, nargs=2,
        default=[-10., 10.],
        help=('value range for image colormap'))
    parser.add_argument('--mask_range', type=float, nargs=2,
            default=[-1., 1.],
            help=('value range for image colormap'))
    return parser


def plot_image(**kwargs):
    # read first HDF5 file
    firstfile = kwargs['firstfile']
    first, basegrid, timestamp = files.read_file(firstfile)

    # read second HDF5 file
    secondfile = kwargs['secondfile']
    second, basegrid, timestamp = files.read_file(secondfile)

    # difference
    difference = first - second
    if kwargs.get('mask_range'):
        mask_lower, mask_upper = kwargs['mask_range']
        mask = (difference > mask_lower) & (difference < mask_upper)
        difference = np.ma.masked_array(difference, mask=mask)

    # plot to image file
    imagefile = kwargs.get('imagefile')
    if imagefile is None:
        filedir, filename = os.path.split(firstfile)
        name, ext = os.path.splitext(filename)
        imagefile = os.path.join(filedir, 'difference_{}.png'.format(name))

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

    plot.image(difference,
        imagefile=imagefile,
        figsize=figsize,
        features=features,
        vmin=vmin,
        vmax=vmax,
        cmap_under=None,
        cmap='coolwarm_r',
        )


def main():
    # parse command line arguments
    arguments = get_parser().parse_args()

    # run
    plot_image(**vars(arguments))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()
