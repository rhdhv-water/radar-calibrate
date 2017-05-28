# -*- coding: utf-8 -*-
# Royal HaskoningDHV
import os

# variable folder
VAR = os.path.join('.', 'var')
if not os.path.exists(VAR):
    os.mkdir(VAR)

# raster folder
RASTERDIR = os.path.join(VAR, 'raster')
if not os.path.exists(RASTERDIR):
    os.mkdir(RASTERDIR)

# plot folder
PLOTDIR = os.path.join(VAR, 'plot')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)
