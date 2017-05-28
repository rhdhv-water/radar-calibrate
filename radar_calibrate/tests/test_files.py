# -*- coding: utf-8 -*-
"""tests for files module.

Run using pytest:
> python -m pytest test_gridtools.py

"""
# Royal HaskoningDHV

from radar_calibrate import files

import numpy

import logging
import glob
import os

def test_hdf2raster():
    datafolder = r'data'
    h5files = glob.glob(os.path.join(datafolder, '*.h5'))
    for h5file in h5files:
        logging.info('writing {} to raster'.format(os.path.basename(h5file)))
        filename, ext = os.path.splitext(h5file)
        rasterfile = filename + '.tif'
        files.hdf2raster(h5file, rasterfile)
