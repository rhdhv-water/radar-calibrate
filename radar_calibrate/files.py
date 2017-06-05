# -*- coding: utf-8 -*-
"""Collection of functions operating on georeferenced grids.

To be merged with openradar.gridtools.

"""
# Royal HaskoningDHV
import numpy as np

import logging


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
    imagedata = np.ma.masked_equal(imagedata, fill_value)
    return imagedata.astype(np.float64).filled(np.nan) * 1e-2
