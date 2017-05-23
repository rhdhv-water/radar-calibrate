#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

import h5py
import numpy
import os

root = r'C:\Project_OG\BA8186_NRR\2_technical'
file_aggregate = r'\radar-calibrate\data\24uur_20170223080000.h5'

with h5py.File(os.path.join(root + file_aggregate), 'r') as ds:
    aggregate = numpy.float64(ds['precipitation'][:]).T # Index is [x][y]
    grid_extent = ds.attrs['grid_extent']
    grid_size = ds.attrs['grid_size']
    left, right, top, bottom = grid_extent
    pixelwidth = (right - left) / grid_size[0]
    pixelheight = (bottom - top) / grid_size[1]
    
#with h5py.File(os.path.join(root + "_smooth_" + file_aggregate), 'w') as ds:
    

    