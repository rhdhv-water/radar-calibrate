# -*- coding: utf-8 -*-
# Royal HaskoningDHV

import pykrige

import logging


def ked_py(x, y, z, radar, xi, yi, zi, plot_vario=False):
    """
    Run the kriging method using the Python module Pykrige using vectorized backend to save time (high memory).
    Kriging External Drift (or universal kriging).
    Input x, y, z and radar (rainstation size) shoud be equally long and as numpy array.
    Input xi, yi and zi (radar size) should be equally long and as numpy array.
    Input vario will display the variogram.

    Returns calibrated grid
    """
    # Create predictor
    ked = pykrige.UniversalKriging(x, y, z,
                                   drift_terms = ["specified"],
                                   specified_drift = [radar,],
                                   variogram_model = "spherical",
#                                   variogram_parameters = {'sill': 80, 'range': 25000, 'nugget': 0},
#                                   variogram_model = 'custom',
#                                   variogram_function([100,50000,0], 5000)
#                                   nlags = 10,
                                   verbose = False,

    )
    if plot_vario:
        ked.display_variogram_model()
    # Run predictor
    rain_est,sigma = ked.execute('grid', xi, yi,
                         specified_drift_arrays = [zi,],
                         backend="vectorized",
    )
    return rain_est, sigma


