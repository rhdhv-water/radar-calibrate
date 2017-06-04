# -*- coding: utf-8 -*-
"""Summary
"""
# Royal HaskoningDHV
from radar_calibrate import config

import h5py

from collections import namedtuple
import time
import os

Timedresult = namedtuple('timedresult', 'dt result')


def timethis(func):
    """Function or method decorator measuring execution time

    See https://www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-funcs

    Parameters
    ----------
    func : Callable
        Function or method to be timed

    Returns
    -------
    Timedresult
        Named tuple containing execution time and result
    """
    def timed(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        dt = toc - tic
        return Timedresult(dt=dt, result=result)

    return timed


def safe_first(array):
    try:
        return array.flatten()[0]
    except IndexError:
        return numpy.nan


def get_testdata(aggregatefile, calibratefile, reshape = None):
    """Summary

    Parameters
    ----------
    aggregatefile : TYPE
        Description
    calibratefile : TYPE
        Description
    reshape : INT
        Resamples the aggregate grid to a higher resolution as dummy input
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

    # coordinate index vectors
    ncols, nrows = grid_size
    cellwidth, cellheight = basegrid.get_cellsize()
    left, right, top, bottom = grid_extent

    if reshape == None:
        # define xi and yi arrays and zi
        xi = numpy.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
        yi = numpy.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)
        zi = aggregate
    else:
        # recalculate the coordinate index vectors
        ncols = ncols * reshape
        nrows = nrows * reshape
        cellwidth = cellwidth / reshape
        cellheight = cellheight / reshape
        # define xi and yi arrays and zi based on interpolation of the aggregate
#        zi = numpy.empty(numpy.array(aggregate.shape) * 10)
        xi = numpy.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
        yi = numpy.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)
        zi = interpolation.zoom(aggregate, reshape, order=0)

#        xi = numpy.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
#        yi = numpy.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)
#        vals = numpy.reshape(aggregate, (len(aggregate[:][0]) * len(aggregate[:])))
#        pts = numpy.array([[i,j] for i in xi[:,0] for j in yi[0,:]] )
#        zi = inter.griddata(pts, vals, (xi, yi), method='linear')
#
    # calibrate kwargs to dict
    calibrate_kwargs = {
        'x': x,
        'y': y,
        'z': z,
        'radar': radar,
        'xi': xi,
        'yi': yi,
        'zi': zi,
        }


    # return tuple
    return calibrate_kwargs, aggregate, calibrate


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


def rainstations2shape(aggregatefile, calibratefile, shapefile,
    calibrate=None, epsg=28992, driver='ESRI Shapefile', agg=numpy.median):
    """Summary

    Parameters
    ----------
    aggregatefile : TYPE
        Description
    calibratefile : TYPE
        Description
    shapefile : TYPE
        Description
    calibrate : None, optional
        Description
    epsg : int, optional
        Description
    driver : str, optional
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
        if calibrate is None:
            calibrate = get_imagedata(ds)
        cal_station_coords = ds.attrs['cal_station_coords']
        cal_station_values = ds.attrs['cal_station_measurements']

    # sample aggregate at calibration station coordinates
    radar_values = gridtools.sample_grid(
        coords=cal_station_coords,
        grid=aggregate,
        geotransform=basegrid.get_geotransform(),
        agg=agg,
        )
    radar = numpy.array([v for v in radar_values])

    # sample calibrate at calibration station coordinates
    cal_radar_values = gridtools.sample_grid(
        coords=cal_station_coords,
        grid=calibrate,
        geotransform=basegrid.get_geotransform(),
        agg=agg,
        )
    cal_radar = numpy.array([v for v in cal_radar_values])

    # records
    zipped_rows = zip(cal_station_coords, cal_station_values, radar, cal_radar)
    records = ({'geometry': {
        'coordinates': (x , y),
        'type': 'Point',
        },
        'properties': {
            'x': x,
            'y': y,
            'z': z,
            'radar': radar,
            'cal_radar': cal_radar,
            'dz':  cal_radar - z,
            }
        } for (x, y), z, radar, cal_radar in zipped_rows)

    # shapefile coordinate reference
    crs = from_epsg(epsg)

    # shapefile schema
    schema = {
        'geometry': 'Point',
        'properties': {
            'x': 'float',
            'y': 'float',
            'z': 'float',
            'radar': 'float',
            'cal_radar': 'float',
            'dz': 'float',
            }
        }

    # write to file
    logging.debug('writing to {}'.format(os.path.basename(shapefile)))
    write_shape(records, shapefile, crs, schema, driver=driver)


def write_shape(records, shapefile, crs, schema, driver='ESRI Shapefile'):
    """Summary

    Parameters
    ----------
    records : TYPE
        Description
    shapefile : TYPE
        Description
    crs : TYPE
        Description
    schema : TYPE
        Description
    driver : str, optional
        Description
    """
    with fiona.open(shapefile, 'w', driver, schema, crs) as dst:
        dst.writerecords(records)
