#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

"""
# IN:
len(x) = 71 NPArray
len(y) = 71 NPArray
len(z) = 71 NPArray
len(radar) = 71 NPArray
len(xi) = 245000 NPArray
len(yi) = 245000 NPArray
len(zi) = 245000 NPArray

# OUT:
shape(rain_est) = 490 (r) * 500 (c)
     product_corners = -110000,210000,-110000,700000,390000,700000,390000,210000
"""


import os
import numpy
import pandas as pd

folder = r'C:\Project_OG\BA8186_NRR\2_technical\radar-calibrate\data'
csvfile = r'\rainstations.csv'
rainstation = pd.read_csv(folder + csvfile)
x = rainstation['x']
y = rainstation['y']
z = rainstation['z']


