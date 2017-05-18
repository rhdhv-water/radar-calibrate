#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

  
#==============================================================================
# FUNCTIONS
#==============================================================================
def get_radar_for_locations(x, y, grid_extent, aggregate, block=2):
    """
    Radar "pixel"values for location closest to weather station.
    Returns those pixels that are closest to the rain stations
    """
    # Changes name size to block as size is a function as well and might confuse
    left, right, top, bottom = grid_extent
    block = block # use number of surrounding pixels
    radar = []
    for i in range(len(x)):
        xoff = int((x[i] - left) / pixelwidth)
        yoff = int((y[i] - bottom) / pixelheight) # pixelheight is -1000
        data = aggregate[xoff:xoff + block, yoff:yoff + block]
        radar.append(numpy.median(data))
    return radar

def get_grid(aggregate, grid_extent):
    """
    Return x and y coordinates of cell centers.
    """
#    cellwidth, cellheight = aggregate.get_cellsize()
    left, right, top, bottom = grid_extent

    nx = numpy.size(aggregate, 0)
    ny = numpy.size(aggregate, 1)
    xmin = left + pixelwidth / 2
    xmax = right - pixelwidth / 2
    ymin = bottom - pixelheight / 2
    ymax = top + pixelheight / 2

    xi, yi = numpy.mgrid[xmin:xmax:nx * 1j, ymax:ymin:ny * 1j,]
    xi = numpy.float32(xi).flatten()
    yi = numpy.float32(yi).flatten()
    return xi, yi

def plot_vgm_R(vgm_py, residual_py):
    figure = plt.figure()
    plt.plot(vgm_py[1,:],vgm_py[0,:],'r*')
    plt.xlabel('distance')
    plt.ylabel('semivariance')
    plt.title('R variogram')
    return figure
    
    
def ked_R(x, y, z, radar, xi, yi, zi):
    """
    Run the kriging method using the R module "gstat".
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long.
    Inputs xi, yi and zi (radar size) should be equally long.
    """

    import rpy2.robjects as robj
    robj.r.library('gstat')

    # Modification to prevent singular matrix (Tom)
    radar += (1e-9 * numpy.random.rand(len(radar)))

    # Convert data readible to R
    radar = robj.FloatVector(radar)
    x, y, z = robj.FloatVector(x), robj.FloatVector(y), robj.FloatVector(z)
    xi, yi, zi = robj.FloatVector(xi), robj.FloatVector(yi), robj.FloatVector(zi)
    rain_radar_frame = robj.DataFrame({'x': x, 'y': y, 'z': z, 'radar': radar})
    rain_radar_frame_py = numpy.array(rain_radar_frame)
    radar_frame = robj.DataFrame({'x': xi, 'y': yi, 'radar': zi})
    radar_frame_py = numpy.array(radar_frame)

    # Create predictor
    vgm = robj.r.variogram(robj.r("z~radar"),
                           robj.r('~ x + y'),
                           data=rain_radar_frame,
                           cutoff=50000,
                           width=5000)
    vgm_py = numpy.array(vgm)
    residual_callable = robj.r('fit.variogram')
    residual = residual_callable(vgm, robj.r.vgm(1, 'Sph', 25000, 1))
    residual_py = numpy.array(residual)
    plot_vgm_R(vgm_py, residual_py).show()

    ked = robj.r('NULL')
    ked = robj.r.gstat(ked, 'raingauge', robj.r("z~radar"),
                       robj.r('~ x + y'),
                       data=rain_radar_frame,
                       model=residual, nmax=40)

    # Run predictor
    result = robj.r.predict(ked, radar_frame, nsim=0)
    rain_est = numpy.array(result[2])

    # Correction 0's and extreme kriged values
    zi = numpy.array(zi)
    zero_or_no_data = numpy.logical_or(zi == 0, zi == -9999)
    correction_factor = numpy.ones(zi.shape)
    correction_factor[~zero_or_no_data] = (rain_est[~zero_or_no_data] / zi[~zero_or_no_data])
    leave_uncalibrated = numpy.logical_or(correction_factor < 0, correction_factor > 10)
    logging.info('Leaving {} extreme pixels uncalibrated.'.format(leave_uncalibrated.sum(),))
#    rain_est[leave_uncalibrated] = zi[leave_uncalibrated]
    
    leave_uncalibrated = leave_uncalibrated.reshape([500, 490])
    return rain_est, correction_factor

def ked_py(x, y, z, radar, xi, yi, zi):
    """
    Run the kriging method using the Pykrige module "gstat".
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long. 
    Inputs xi, yi and zi (radar size) should be equally long.
    """
    import pykrige
    # Create predictor
    ked = pykrige.UniversalKriging(x, y, z, 
                                   drift_terms = ["specified"],
                                   specified_drift = [radar,],
                                   variogram_model = "spherical",
                                   variogram_parameters = {'sill': 100, 'range': 50000, 'nugget': 0},
#                                   variogram_model = 'custom',
#                                   variogram_function([100,50000,0], 5000)
                                   nlags = 10,
                                   verbose = True,
                                   
    )
    ked.display_variogram_model()
    
    # Run predictor
    y_pred = ked.execute('points', xi, yi, 
                         specified_drift_arrays = [zi,],
                         backend="vectorized",
    )
    rain_est = numpy.squeeze(y_pred)[0]
    return rain_est
