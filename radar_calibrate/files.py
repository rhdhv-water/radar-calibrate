# -*- coding: utf-8 -*-
"""Collection of functions operating on georeferenced grids.

To be merged with openradar.gridtools.

"""
# Royal HaskoningDHV
from radar_calibrate import gridtools

from affine import Affine
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
        aggregate = get_imagedata(ds)
        grid_extent = ds.attrs['grid_extent']
        grid_size = [int(i) for i in ds.attrs['grid_size']]

    # construct basegrid
    basegrid = gridtools.BaseGrid(extent=grid_extent,
        size=grid_size)

    # read calibrate and station measurements from calibrate file
    with h5py.File(calibratefile, 'r') as ds:
        calibrate = get_imagedata(ds)
        cal_station_coords = ds.attrs['cal_station_coords']
        cal_station_values = ds.attrs['cal_station_measurements']

    # select measurements which are equal or larger than zero
    rain_ge_zero = cal_station_values >= 0.
    cal_station_coords = cal_station_coords[rain_ge_zero, :]
    cal_station_values = cal_station_values[rain_ge_zero]


    # sample aggregate at calibration station coordinates
    radar_values = gridtools.sample_grid(
        coords=cal_station_coords,
        grid=aggregate,
        geotransform=basegrid.get_geotransform(),
        )
    radar = numpy.array([v for v in radar_values])

    # unselect measurement if radar is NaN
    radar_nan = numpy.isnan(radar)
    cal_station_coords = cal_station_coords[~radar_nan, :]
    cal_station_values = cal_station_values[~radar_nan]
    radar = radar[~radar_nan]

    # unpack and rename coordinates
    x = cal_station_coords[:, 0]
    y = cal_station_coords[:, 1]
    z = cal_station_values

    # transform aggregate grid to coordinate and value vectors
    xi, yi = [numpy.float32(a).flatten() for a in basegrid.get_grid()]
    zi = aggregate.flatten()

    # return tuple
    return x, y, z, radar, xi, yi, zi, aggregate, calibrate


def hdf2raster(h5file, rasterfile,
    dtype=rasterio.float64, driver='GTiff', epsg=28992, fill_value=-9999):
    """Summary

    Parameters
    ----------
    h5file : TYPE
        Description
    rasterfile : TYPE
        Description
    dtype : TYPE, optional
        Description
    driver : str, optional
        Description
    epsg : int, optional
        Description
    fill_value : TYPE, optional
        Description
    """
    # open h5 file and get values and attributes
    with h5py.File(h5file, 'r') as ds:
        values = get_imagedata(ds)
        grid_extent = ds.attrs['grid_extent']
        grid_size = [int(i) for i in ds.attrs['grid_size']]

    # construct basegrid
    basegrid = gridtools.BaseGrid(extent=grid_extent,
        size=grid_size)

    # construct Affine geotransform
    transform = Affine.from_gdal(*basegrid.get_geotransform())

    # write to raster file
    write_raster(values, rasterfile, transform, dtype, driver, epsg, fill_value)


def write_raster(array, rasterfile, transform,
    dtype=rasterio.float64, driver='GTiff', epsg=None, fill_value=-9999):
    """Write georeferenced array to single band raster file.

    Parameters
    ----------
    array : TYPE
        Description
    rasterfile : TYPE
        Description
    transform : TYPE
        Description
    dtype : TYPE, optional
        Description
    driver : str, optional
        Description
    epsg : int, optional
        coordinate reference EPSG code
    fill_value : TYPE, optional
        Description
    """
    if epsg is not None:
        try:
            crs = rasterio.crs.CRS.from_epsg(epsg)
        except ValueError:
            pass
    else:
        crs = None

    # construct raster file profile
    nrows, ncols = array.shape
    profile = {
        'width': ncols,
        'height': nrows,
        'count': 1,
        'dtype': dtype,
        'driver': driver,
        'crs': crs,
        'transform': transform,
        'nodata': fill_value,
        }

    # replace NaN with fill_value
    array[numpy.isnan(array)] = fill_value

    # write to file
    logging.debug('writing to {}'.format(os.path.basename(rasterfile)))
    with rasterio.open(rasterfile, 'w', **profile) as dst:
        dst.write(array.astype(dtype), 1)
