# -*- coding: utf-8 -*-
"""Collection of functions operating on georeferenced grids.

To be merged with openradar.gridtools.

"""
# Royal HaskoningDHV
from radar_calibrate import config
from radar_calibrate import gridtools

from affine import Affine
from fiona.crs import from_epsg
import scipy.interpolate as inter
import scipy.ndimage.interpolation as interpolation
import rasterio
import numpy
import fiona
import h5py

import logging
import os


def get_imagedata(ds, fill_value=65535):
    """Summary

    Parameters
    ----------
    ds : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    imagedata = ds['image1/image_data'][:]
    imagedata = numpy.ma.masked_equal(imagedata, fill_value)
    return imagedata.astype(numpy.float64).filled(numpy.nan) * 1e-2
