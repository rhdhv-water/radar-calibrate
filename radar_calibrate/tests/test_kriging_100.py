# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import gridtools
from radar_calibrate import files
from radar_calibrate import kriging
from radar_calibrate import utils

import numpy

import logging
import os

@utils.timethis
def krige_py(x, y, z, radar, xi, yi, zi):
    return kriging.ked_py(x, y, z, radar, xi, yi, zi)

def test_compare_grid(plot_comparison=False, timestamp='20170223080000'):
    # test data from files
    aggregatefile = r'data\24uur_{}.h5'.format(timestamp)
    calibratefile = r'data\RAD_TF2400_U_{}.h5'.format(timestamp)
    calibrate_kwargs, aggregate, calibrate = files.get_testdata(
        aggregatefile,
        calibratefile,
        )
    # NaN mask
    nan_mask = numpy.isnan(aggregate)

    # ked using Python 1000*1000
    timedresult_py = krige_py(**calibrate_kwargs)
    logging.info('ked in python took {dt:.2f} seconds'.format(
        dt=timedresult_py.dt))
    rain_est_py, sigma = timedresult_py.result
    calibrate_py = utils.apply_countrymask(
        rain_est_py.reshape(calibrate.shape), aggregate)
    calibrate_py[nan_mask] = numpy.nan

    # ked using Python 500*500
    calibrate_kwargs, aggregate, calibrate = files.get_testdata(
        aggregatefile,
        calibratefile,
        reshape = 2,
        )
    timedresult_py = krige_py(**calibrate_kwargs)
    logging.info('ked in python took {dt:.2f} seconds'.format(
        dt=timedresult_py.dt))
    rain_est_py, sigma = timedresult_py.result
#    calibrate_py = utils.apply_countrymask(
#        rain_est_py.reshape(calibrate.shape), aggregate)
#    calibrate_py[nan_mask] = numpy.nan

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_compare_grid(plot_comparison=True)







#
#import h5py
#import numpy
#import matplotlib.pyplot as plt
#import scipy.interpolate as inter
#from time import time
#
#from radar_calibrate import kriging
#
#file_aggregate = r'..\data\24uur_20170223080000.h5'
#file_smooth = r'..\data\24uur_20170223080000_smooth.h5'
#file_calibrate = r'..\data\RAD_TF2400_U_20170223080000.h5'
#
#with h5py.File(file_aggregate, 'r') as ds:
#    aggregate = numpy.float64(ds['precipitation'][:]).T
#    grid_extent = ds.attrs['grid_extent']
#    grid_size = ds.attrs['grid_size']
#    left, right, top, bottom = grid_extent
#    pixelwidth = (right - left) / grid_size[0]
#    pixelheight = (bottom - top) / grid_size[1]
#with h5py.File(file_calibrate, 'r') as ds:
#    calibrate = numpy.float64(ds['image1/image_data']).T # Index is [x][y]
#    coords = numpy.array(ds.attrs['cal_station_coords'])
#    x = coords[:, 0]
#    y = coords[:, 1]
#    z = numpy.array(ds.attrs['cal_station_measurements'])
#
##==============================================================================
## Prep data as numpy arrays, as the required format of the krige modules
##==============================================================================
#radar = numpy.array(kriging.get_radar_for_locations(x, y, grid_extent, aggregate, pixelwidth, pixelheight))
#xi, yi = kriging.get_grid(aggregate, grid_extent, pixelwidth, pixelheight)
## Initialize smoothing grid
#factor_x = 10
#factor_y = 10
#pixelwidth_smooth = pixelwidth/factor_x
#pixelheight_smooth = pixelheight/factor_y
#zi_smooth = numpy.empty(numpy.array(aggregate.shape) * 10)
#xi_smooth, yi_smooth = kriging.get_grid(zi_smooth, grid_extent, pixelwidth_smooth, pixelheight_smooth)
## Interpolate grid with inter from scipy.
#vals = numpy.reshape(aggregate, (len(aggregate[:][0]) * len(aggregate[:])))
#pts = numpy.array([[i,j] for i in xi[:,0] for j in yi[0,:]] )
#zi_smooth = inter.griddata(pts, vals, (xi_smooth, yi_smooth), method='linear')
#
#xi_smooth = numpy.float32(xi_smooth).flatten()
#yi_smooth = numpy.float32(yi_smooth).flatten()
#zi_smooth = numpy.float32(zi_smooth).flatten()
#
##==============================================================================
## RUN FUNCTIONS
##==============================================================================
##tic = time()
##rain_est_R = kriging.ked_R(x, y, z, radar, xi_smooth, yi_smooth, zi_smooth, False)
##calibrate_R = rain_est_R.reshape(zi_smooth.shape)
##print("R:" + str(time() - tic) + "seconds")
#
#tic = time()
#rain_est_py = kriging.ked_py_l(x, y, z, radar, xi_smooth, yi_smooth, zi_smooth, False)
#calibrate_py = rain_est_py.reshape(zi_smooth.shape)
#print("py:" + str(time() - tic) + "seconds")
#
## Plot radar_calibrate_R
#plt.figure()
#f221 = plt.subplot(2, 1, 1)
#plt.imshow(aggregate, cmap='rainbow', vmin=0, vmax=40)
#plt.ylabel('y-coordinate')
#plt.title('aggregate')
#f222 = plt.subplot(2, 1, 2)
#plt.imshow(zi_smooth, cmap='rainbow', vmin=0, vmax=40)
#plt.title('$calibrate_{original}$')
