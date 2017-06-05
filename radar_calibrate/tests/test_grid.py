#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

import numpy as np

def grid_arjan():
    ncols, nrows = 20, 10  # Similar to ncols, nrows
    left, right, top, bottom = 0,20,30,20
    cellwidth, cellheight = 1,1

    xmin = left + cellwidth / 2
    xmax = right - cellwidth / 2
    ymin = bottom + cellheight / 2
    ymax = top - cellheight / 2

    yi, xi = np.mgrid[
        ymax:ymin:nrows * 1j, xmin:xmax:ncols * 1j]
    return xi, yi

def grid():
    ncols, nrows = 20, 10
    left,right,top,bottom = 0,20,30,20
    cellwidth,cellheight = 1,1

    xi = np.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
    yi = np.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)


    xi, yi = np.meshgrid(xi, yi, indexing='xy')
    return xi, yi
