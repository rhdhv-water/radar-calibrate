#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV
"""
create dummy data
"""

#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

import numpy as np
import h5py

import datetime
import glob
import os
import pandas as pd

folder = r'C:\Project_OG\BA8186_NRR\2_technical\radar-calibrate\data'
csvfile = r'\rainstations.csv'
h5files = glob.glob(os.path.join(folder, '*.h5'))
filenames = []

for h5file in h5files:
    with h5py.File(h5file) as ds:
        values = np.ma.masked_equal(ds['image1/image_data'][:],65535)
        date = datetime.datetime.strptime(os.path.basename(h5file),'radar_5min_%Y%m%d%H%M%S.h5')
        bar = np.linspace(0, 50, 10, endpoint=True)
        filename = folder + "\\" + date.strftime("%Y%m%d%H%M%S")

rainstation = pd.read_csv(folder + csvfile)
x = rainstation['x'].as_matrix()
y = rainstation['y'].as_matrix()
z = rainstation['z'].as_matrix()

radar = np.array(None)
for i in range(len(z)):
    radar = (values[int(round((x[i] + 110000)/1000))][int(round((y[i] -210000)/1000))])
    