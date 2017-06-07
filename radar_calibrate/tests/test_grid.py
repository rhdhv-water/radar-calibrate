#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

import numpy as np


def test_grid_equal():
    size = 20, 10
    extent = 0, 20, 30, 20
    cellsize = 1, 1
    xi, yi = grid_mgrid(size, extent, cellsize)
    xi2, yi2 = grid_meshgrid(size, extent, cellsize)
    np.testing.assert_allclose(xi, xi2)
    np.testing.assert_allclose(yi, yi2)


def grid_mgrid(size, extent, cellsize):
    ncols, nrows = size
    left, right, top, bottom = extent
    cellwidth, cellheight = cellsize

    xmin = left + cellwidth / 2
    xmax = right - cellwidth / 2
    ymin = bottom + cellheight / 2
    ymax = top - cellheight / 2

    yi, xi = np.mgrid[
        ymax:ymin:nrows * 1j, xmin:xmax:ncols * 1j]
    return xi, yi

def grid_meshgrid(size, extent, cellsize):
    ncols, nrows = size
    left, right, top, bottom = extent
    cellwidth, cellheight = cellsize

    xi = np.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
    yi = np.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)

    xi, yi = np.meshgrid(xi, yi, indexing='xy')
    return xi, yi
