# -*- coding: utf-8 -*-
"""Collection of functions operating on georeferenced grids.

To be merged with openradar.gridtools.

"""
# Royal HaskoningDHV
from radar_calibrate import gridtools

import numpy as np
import h5py

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


def read_aggregate(aggregatefile):
    '''read aggregate and grid properties from aggregate file'''
    with h5py.File(aggregatefile, 'r') as ds:
        aggregate = get_imagedata(ds)
        grid_extent = ds.attrs['grid_extent']
        grid_size = [int(i) for i in ds.attrs['grid_size']]

    # construct basegrid
    basegrid = gridtools.BaseGrid(extent=grid_extent,
        size=grid_size)

    return aggregate, basegrid


def read_calibrate(calibratefile):
    '''read calibrate and station measurements from calibrate file'''
    with h5py.File(calibratefile, 'r') as ds:
        calibrate = get_imagedata(ds)
        rainstation_coords = ds.attrs['cal_station_coords']
        rainstation_values = ds.attrs['cal_station_measurements']
        rainstations = np.concat()

    return calibrate, rainstations


def save_calibrate(calibratefile, attrs):
    with h5py.File(calibratefile, 'w') as ds:
        pass
