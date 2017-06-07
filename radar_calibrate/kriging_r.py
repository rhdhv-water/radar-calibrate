# -*- coding: utf-8 -*-
# Royal HaskoningDHV

import rpy2.robjects as robj
import numpy as np

import logging

log = logging.getLogger(os.path.basename(__file__))


def ked_r(x, y, z, radar, xi, yi, zi):

    """
    Run the kriging method using the R module "gstat".
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long.
    Inputs xi, yi and zi (radar size) should be equally long.
    Input vario will display the variogram.

    Returns calibrated grid
    """
    robj.r.library('gstat')
    # xi, yi, zi to vectors

    xi, yi = np.meshgrid(xi, yi, indexing='xy')
    xi = np.float32(xi).flatten()
    yi = np.float32(yi).flatten()
    zi = np.float32(zi).flatten()

    # Modification to prevent singular matrix (Tom)
    radar += (1e-9 * np.random.rand(len(radar)))

    # variables to R dataframes
    rain_radar_frame = robj.DataFrame({
        'x': robj.FloatVector(x),
        'y': robj.FloatVector(y),
        'z': robj.FloatVector(z),
        'radar': robj.FloatVector(radar),
        })

    radar_frame = robj.DataFrame({
        'x': robj.FloatVector(xi),
        'y': robj.FloatVector(yi),
        'radar': robj.FloatVector(zi),
        })

    # try kriging
    try:
        vgm = robj.r.variogram(robj.r("z~radar"), robj.r('~ x + y'),
                               data=rain_radar_frame,
                               cutoff=50000, width=5000)
        residual_callable = robj.r('fit.variogram')
        residual = residual_callable(vgm, robj.r.vgm(1, 'Sph', 25000, 1))
        ked = robj.r('NULL')
        ked = robj.r.gstat(ked, 'raingauge', robj.r("z~radar"),
                           robj.r('~ x + y'),
                           data=rain_radar_frame,
                           model=residual, nmax=40)
        result = robj.r.predict(ked, radar_frame, nsim=0)
        rain_est = np.array(result[2])
    except:
        log.exception('Exception during kriging:')
        rain_est = zi

    return rain_est, None, None
