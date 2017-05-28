# -*- coding: utf-8 -*-
"""tests for files module.

Run using pytest:
> python -m pytest test_gridtools.py

"""
# Royal HaskoningDHV

from radar_calibrate import config
from radar_calibrate import files
from radar_calibrate import utils

import numpy

import logging
import glob
import os


def test_hdf2raster():
    datafolder = r'data'
    h5files = glob.glob(os.path.join(datafolder, '*.h5'))
    for h5file in h5files:
        logging.info('writing {} to raster'.format(os.path.basename(h5file)))
        filename = os.path.basename(h5file)
        name, ext = os.path.splitext(filename)
        rasterfile = os.path.join(config.RASTERDIR, name + '.tif')
        files.hdf2raster(h5file, rasterfile)


def test_rainstations2shape():
    h5files = [
        (r'data\24uur_20170223080000.h5', r'data\RAD_TF2400_U_20170223080000.h5'),
        (r'data\24uur_20170228080000.h5', r'data\RAD_TF2400_U_20170228080000.h5'),
        (r'data\24uur_20170305080000.h5', r'data\RAD_TF2400_U_20170305080000.h5'),
        ]
    for aggregatefile, calibratefile in h5files:
        logging.info('writing {} to shapefile'.format(
            os.path.basename(calibratefile)))
        filename = os.path.basename(calibratefile)
        name, ext = os.path.splitext(filename)
        shapefile = os.path.join(config.SHAPEDIR, name + '.shp')
        files.rainstations2shape(aggregatefile, calibratefile, shapefile,
            agg=utils.safe_first)
