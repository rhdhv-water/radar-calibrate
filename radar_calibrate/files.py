# -*- coding: utf-8 -*-
# Royal HaskoningDHV
from radar_calibrate import gridtools

import numpy as np
import fiona
import h5py

import datetime
import logging
import json
import os

log = logging.getLogger(os.path.basename(__file__))


def get_imagedata(ds, name,
    fill_value=65535, dtype=np.float64, multiplier=1e-2):
    '''get precipitation as masked array from image1/image_data group'''
    data = ds[name][:]
    data = np.ma.masked_equal(data, fill_value)
    return data.astype(dtype) * multiplier


def add_imagedata(ds, name, data, fill_value=65535, multiplier=1e2):
    '''add data as integer array to HDF5 dataset'''
    int_dataset = ds.create_dataset(name,
        data.shape,
        dtype='u2', compression='gzip', shuffle=True)
    int_dataset[...] = np.ma.filled(np.uint16(np.round(data * multiplier)),
        fill_value=fill_value)


def add_floatdata(ds, name, data, fill_value=-9999):
    '''add data as float array to HDF5 dataset'''
    float_dataset = ds.create_dataset(name, data.shape, dtype='f4',
                                compression='gzip', shuffle=True)
    float_dataset[...] = data.filled(fill_value).astype(np.float32)


def read_rainstations(rainstationsfile, timestamp):
    '''read rainstation values from JSON file at timestamp'''
    log.debug('reading rainstations from file {file:}'.format(
        	file=os.path.basename(rainstationsfile)))
    with open(rainstationsfile) as f:
        try:
            data = json.load(f)[timestamp.strftime('%Y-%m-%dT%H:%M:%S')]
        except KeyError:
            logging.warning('timestamp not in {file:}'.format(
                file=os.path.basename(rainstationsfile)))
            return np.array([]), np.array([])
        station_coords = np.array([s['coords'] for s in data])
        station_values = np.array([s['value'] for s in data])
        return station_coords, station_values


def read_file(datafile):
    '''read data and grid properties from HDF5 file'''
    log.debug('reading data from file {file:}'.format(
        file=os.path.basename(datafile)))
    with h5py.File(datafile, 'r') as ds:
        data = get_imagedata(ds, 'image1/image_data')
        grid_extent = ds.attrs['grid_extent']
        grid_size = [int(i) for i in ds.attrs['grid_size']]

        timestamp = ds.attrs['timestamp_last_composite']
        try:
            timestamp = timestamp.decode('utf-8')
        except AttributeError:
            pass
        timestamp = datetime.datetime.strptime(timestamp,
            '%Y%m%d%H%M%S')

    # construct basegrid
    basegrid = gridtools.BaseGrid(extent=grid_extent,
        size=grid_size)

    return data, basegrid, timestamp


def rainstations_from_calibrate(calibratefile):
    '''read calibrate and station measurements from calibrate file'''
    log.debug('reading calibrate from file {file:}'.format(
        file=os.path.basename(calibratefile)))
    with h5py.File(calibratefile, 'r') as ds:
        calibrate = get_imagedata(ds, 'image1/image_data')
        station_coords = ds.attrs['cal_station_coords']
        station_values = ds.attrs['cal_station_measurements']

    return station_coords, station_values

def read_mask(maskfile):
    '''read mask from HDF5 file'''
    log.debug('reading mask from file {file:}'.format(
        file=os.path.basename(maskfile)))
    with h5py.File(maskfile, 'r') as ds:
        mask = ds['mask'][...]
        return mask


def read_shape(shapefile, geotransform=None):
    '''read geometry from shapefile and transform for plotting'''
    if geotransform is not None:
        left, cellwidth, _, top, _, cellheight = geotransform
    with fiona.open(shapefile) as src:
        for row in src:
            x, y = zip(*row['geometry']['coordinates'])
            x = np.array(x)
            y = np.array(y)
            if geotransform is not None:
                x = (x - left) / cellwidth
                y = (y - top) / cellheight
            yield x, y


def save_result(resultfile, result, sigma, attrs):
    '''save result to HDF5 format'''
    log.debug('writing result to file {file:}'.format(
        file=os.path.basename(resultfile)))
    with h5py.File(resultfile, 'w') as ds:
        # save result to image dataset
        add_imagedata(ds, 'image1/image_data', result)

        # save sigma (kriging variance)
        if sigma is not None:
            add_floatdata(ds, 'sigma', sigma)

        # save attributes
        for key, value in attrs.items():
            ds.attrs[key] = value
