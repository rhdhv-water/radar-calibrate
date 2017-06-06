# -*- coding: utf-8 -*-
"""Collection of functions operating on georeferenced grids.

To be merged with openradar.gridtools.

"""
# Royal HaskoningDHV
from radar_calibrate import gridtools

import numpy as np
import h5py

import logging


def get_imagedata(ds, fill_value=65535, mm_factor=1e-2):
    '''get precipitation as masked array from image1/image_data group'''
    imagedata = ds['image1/image_data'][:]
    imagedata = np.ma.masked_equal(imagedata, fill_value)
    return imagedata.astype(np.float64) * mm_factor


def add_imagedata(ds, data, int_factor=1e2):
    '''add data as integer array to HDF5 dataset'''
    dataset = ds.create_dataset('image1/image_data',
        data.shape,
        dtype='u2', compression='gzip', shuffle=True)
    dataset[...] = np.uint16(np.round(data * int_factor)).filled()


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
        station_coords = ds.attrs['cal_station_coords']
        station_values = ds.attrs['cal_station_measurements']

    return calibrate, (station_coords, station_values)


def read_mask(maskfile):
    with h5py.File(maskfile, 'r') as ds:
        mask = ds['mask'][...]
        return mask
        

def save_result(resultfile, result, attrs):
    '''save result to HDF5 format'''
    with h5py.File(resultfile, 'w') as ds:
        # save result to image dataset
        add_imagedata(ds, result)

        # save attributes
        for key, value in attrs.items():
            ds.attrs[key] = value
