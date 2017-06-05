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
    """Sample georeferenced array at given coordinates
    Parameters
    ----------
    coords : Sequence
        Sequence or iterator yielding X,Y coordinate pairs
    array : Array
        Description
    geotransform : TYPE
        Description
    blocksize : int, optional
        Description
    agg : TYPE, optional
        Description

    Yields
    ------
    TYPE
        Description
    """
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

        # add blocksize
        col_idx = slice(col_left, col_left + blocksize)
        row_idx = slice(row_top, row_top + blocksize)

        # sample array and aggregate block values
        try:
            block = array[row_idx, col_idx]
        except IndexError:
            block = np.nan

        yield agg(block)


def resample(xi, yi, zi, xi_new, yi_new, method='linear'):
    grid = np.meshgrid(xi, yi, indexing='xy')
    grid_new = np.meshgrid(xi_new, yi_new, indexing='xy')
    resampled = griddata(grid, zi, grid_new, method=method)
    return resampled


def add_vector_layer(shapefile, geotransform):
    left, cellwidth, _, top, _, cellheight = geotransform
    with fiona.open(shapefile) as src:
        for row in src:
            x, y = zip(*row['geometry']['coordinates'])
            x = np.array(x)
            y = np.array(y)
            x = (x - left) / cellwidth
            y = (y - top) / cellheight
            yield x, y


class BaseGrid(OpenRadarBaseGrid):
    def get_grid(self, res=None):
        left, right, top, bottom = self.extent
        ncols, nrows = self.size

        if res is not None:
            cellwidth, cellheight = res
        else:
            cellwidth, cellheight = self.get_cellsize()
            
        xi = np.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
        yi = np.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)
        return xi, yi
