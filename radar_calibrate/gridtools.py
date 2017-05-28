# -*- coding: utf-8 -*-
"""Collection of functions operating on georeferenced grids.

To be merged with openradar.gridtools.

"""
# Royal HaskoningDHV

# add all from openradar.gridtools
from openradar.gridtools import BaseGrid

import numpy


def sample_grid(coords, grid, geotransform,
    fill_value=None, blocksize=2, agg=numpy.median):
    """Sample georeferenced grid at given coordinates
    Parameters
    ----------
    coords : Sequence
        Sequence or iterator yielding X,Y coordinate pairs
    grid : Array
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
        col_left = int(numpy.floor((x - left) / cellwidth))
        row_top = int(numpy.floor((y - top) / cellheight)) # cellheight is < 0

        if (col_left < 0) or (row_top < 0):  # prevent negative indexing
            yield numpy.nan
            continue

        # add blocksize
        col_idx = slice(col_left, col_left + blocksize)
        row_idx = slice(row_top, row_top + blocksize)

        # sample grid and aggregate block values
        try:
            block = grid[row_idx, col_idx]
        except IndexError:
            block = numpy.nan

        if (fill_value is not None) and numpy.any(block == fill_value):
            yield numpy.nan

        yield agg(block)
