# -*- coding: utf-8 -*-
"""tests for files module.

Run using pytest:
> python -m pytest test_gridtools.py

"""
# Royal HaskoningDHV

from radar_calibrate.tests import config
from radar_calibrate.tests import utils

import numpy as np

import logging
import glob
import os

def test_hdf2raster():
    h5files = glob.glob(os.path.join(config.DATADIR, '*.h5'))
    for h5file in h5files:
        logging.info('writing {} to raster'.format(os.path.basename(h5file)))
        filename = os.path.basename(h5file)
        name, ext = os.path.splitext(filename)
        rasterfile = os.path.join(config.RASTERDIR, name + '.tif')
        utils.hdf2raster(h5file, rasterfile)


def test_rainstations2shape():
    h5files = [
        (r'24uur_20170223080000.h5', r'RAD_TF2400_U_20170223080000.h5'),
        (r'24uur_20170228080000.h5', r'RAD_TF2400_U_20170228080000.h5'),
        (r'24uur_20170305080000.h5', r'RAD_TF2400_U_20170305080000.h5'),
        ]
    for aggregatefile, calibratefile in h5files:
        logging.info('writing {} to shapefile'.format(
            os.path.basename(calibratefile)))
        filename = os.path.basename(calibratefile)
        name, ext = os.path.splitext(filename)
        shapefile = os.path.join(config.SHAPEDIR, name + '.shp')
        aggregatefile = os.path.join(config.DATADIR, aggregatefile)
        calibratefile = os.path.join(config.DATADIR, calibratefile)
        utils.rainstations2shape(aggregatefile, calibratefile, shapefile,
            agg=utils.safe_first)
