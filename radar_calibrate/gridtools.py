# -*- coding: utf-8 -*-
"""Collection of functions operating on georeferenced grids.

To be merged with openradar.gridtools.

"""
# Royal HaskoningDHV

# add all from openradar.gridtools
from openradar.gridtools import BaseGrid as OpenRadarBaseGrid
from scipy.interpolate import griddata
import numpy as np

import os


def sample_array(coords, array, geotransform, blocksize=2, agg=np.median):
    '''sample georeferenced array at given coordinates'''
    # unpack geotransform
    left, cellwidth, _, top, _, cellheight = geotransform

    values = []
    for x, y in coords:
        # x, y to row and column indices
        col_left = int(np.floor((x - left) / cellwidth))
        row_top = int(np.floor((y - top) / cellheight)) # cellheight is < 0

        if (col_left < 0) or (row_top < 0):  # prevent negative indexing
            yield np.nan
            continue

        # add blocksize note: block is not centered around x,y
        col_idx = slice(col_left, col_left + blocksize)
        row_idx = slice(row_top, row_top + blocksize)

        # sample array and aggregate block values
        try:
            block = array[row_idx, col_idx]
        except IndexError:
            block = np.nan

        yield agg(block)


def resample(xi, yi, zi, xi_new, yi_new, method='linear'):
    '''resample array given old and new grid coordinates'''
    xi, yi = np.meshgrid(xi, yi, indexing='xy')
    xi_new, yi_new = np.meshgrid(xi_new, yi_new, indexing='xy')
    zi = np.ma.filled(zi, fill_value=np.nan)
    zi_new = griddata(
        (xi.flatten(), yi.flatten()),
        zi.flatten(),
        (xi_new.flatten(), yi_new.flatten()),
        method=method,
        )
    zi_new = zi_new.reshape(xi_new.shape)
    zi_new = np.ma.masked_invalid(zi_new)
    return zi_new


class BaseGrid(OpenRadarBaseGrid):
    def get_grid(self, cellsize=None):
        '''get x, y grid coordinates as 1-D vectors'''
        left, right, top, bottom = self.extent

        if cellsize is not None:
            cellwidth, cellheight = cellsize
            ncols = int((right - left) / cellwidth)
            nrows = int((top- bottom) / cellheight)
        else:
            cellwidth, cellheight = self.get_cellsize()
            ncols, nrows = self.size

        xi = np.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
        yi = np.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)
        return xi, yi
