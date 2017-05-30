# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate import gridtools
from radar_calibrate import files
from radar_calibrate import kriging
from radar_calibrate import utils
from radar_calibrate import plot
from radar_calibrate import config 

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

    # initialize variables    
    reshapes = [1, 1.5, 2, 4, 5]
    timedresults = []
    reshapes_size = []
    
    for i in range(len(reshapes)):
        # get test data
        calibrate_kwargs, aggregate, calibrate = files.get_testdata(
        aggregatefile,
        calibratefile,
        reshape = reshapes[i],
        )
        # NaN mask
        nan_mask = numpy.isnan(aggregate)
        
        # ked using Python with a reshaped grid
        timedresult = krige_py(**calibrate_kwargs)
        logging.info('ked in python took {dt:.2f} seconds'.format(
            dt=timedresult.dt))
        timedresults.append(timedresult.dt)
        rain_est, sigma = timedresult.result
        reshapes_size.append(rain_est.size)
#        calibrate = utils.apply_countrymask(
#            rain_est.reshape(calibrate.shape), aggregate)
#        calibrate[nan_mask] = numpy.nan

    # plot
    if plot_comparison:
        imagefile = os.path.join(config.PLOTDIR, 'time_ked_{}.png'.format(
            timestamp))
        plot.timedresults(reshapes, timedresults, imagefile=imagefile,)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_compare_grid(plot_comparison=True)



#import scipy.interpolate as inter
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

