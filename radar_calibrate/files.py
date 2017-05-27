# -*- coding: utf-8 -*-
"""Collection of functions operating on georeferenced grids.

To be merged with openradar.gridtools.

"""
# Royal HaskoningDHV
from radar_calibrate import gridtools

import rasterio
import fiona
import h5py

import logging


def get_testdata(aggregatefile, calibratefile):
    """Summary

    Parameters
    ----------
    aggregatefile : TYPE
        Description
    calibratefile : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    # read aggregate and grid properties from aggregate file
    with h5py.File(aggregatefile, 'r') as ds:
        aggregate = get_data(ds)
        grid_extent = ds.attrs['grid_extent']
        grid_size = [int(i) for i in ds.attrs['grid_size']]

    # construct basegrid
    basegrid = openradar_gridtools.BaseGrid(extent=grid_extent,
        size=grid_size)

    # read calibrate and station measurements from calibrate file
    with h5py.File(calibratefile, 'r') as ds:
        calibrate = get_data(ds)
        cal_station_coords = ds.attrs['cal_station_coords']
        cal_station_values = ds.attrs['cal_station_measurements']

    # sample aggregate at calibration station coordinates
    radar_values = gridtools.sample_grid(
        coords=cal_station_coords,
        grid=aggregate,
        geotransform=basegrid.get_geotransform(),
        )

    # transform aggregate grid to coordinate and value vectors
    xi, yi = basegrid.get_grid()
    zi = aggregate.flatten()

    # back to vague naming
    x = cal_station_coords[:, 0]
    y = cal_station_coords[:, 1]
    z = cal_station_values
    radar = radar_values

    # return tuple
    return x, y, z, radar, xi, yi, zi


def get_data(dataset):
    """Summary

    Parameters
    ----------
    dataset : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    import pdb; pdb.set_trace()
    return dataset['image1/image_data'][:].astype(numpy.float64) * 1e-2


