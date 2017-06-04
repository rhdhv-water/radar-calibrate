# -*- coding: utf-8 -*-
# Royal HaskoningDHV
import os

# variable folder
VAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'var')
if not os.path.exists(VAR):
    os.mkdir(VAR)

# raster folder
RASTERDIR = os.path.join(VAR, 'raster')
if not os.path.exists(RASTERDIR):
    os.mkdir(RASTERDIR)

# shape folder
SHAPEDIR = os.path.join(VAR, 'shape')
if not os.path.exists(SHAPEDIR):
    os.mkdir(SHAPEDIR)

# plot folder
PLOTDIR = os.path.join(VAR, 'plot')
if not os.path.exists(PLOTDIR):
    os.mkdir(PLOTDIR)

# misc folder
MISCDIR = os.path.join('.', 'misc')
