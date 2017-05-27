#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

import numpy 

def idw(x, y, z, xi, yi, p=2):
    """
    Simple idw function. Slow, but memory efficient implementation.
    
    Input x, y, z (rainstation size) shoud be equally long.
    Inputs xi, yi (radar size) should be equally long.
    Input p is the power factor.
    
    Returns calibrated grid (zi)
    """
    sum_of_weights = numpy.zeros(xi.shape)
    sum_of_weighted_gauges = numpy.zeros(xi.shape)
    for i in range(x.size):
        distance = numpy.sqrt((x[i] - xi) ** 2 + (y[i] - yi) ** 2)
        weight = 1.0 / distance ** p
        weighted_gauge = z[i] * weight
        sum_of_weights += weight
        sum_of_weighted_gauges += weighted_gauge
    zi = sum_of_weighted_gauges / sum_of_weights

    return zi