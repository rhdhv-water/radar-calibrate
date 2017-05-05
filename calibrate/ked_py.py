#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

"""
# IN:
len(x) = 71
len(y) = 71
len(z) = 71
len(radar) = 71
len(xi) = 245000
len(yi) = 245000
len(zi) = 245000

# OUT:
shape(rain_est) = 490*500
"""

import numpy
import pykrige

def ked_Py(x, y, z, radar, xi, yi, zi):
    """
    Run the kriging method using the python module "Pykrige".
    """
    import pykrige
    
    ked = pykrige.UniversalKriging(x, y, z)
    y_pred = ked.execute('grid', xi, yi)

    rain_est = numpy.squeeze(y_pred)
    return rain_est

