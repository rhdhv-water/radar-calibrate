# -*- coding: utf-8 -*-
# Royal HaskoningDHV

import numpy as np

def sample_grid(coords, grid, extent, res, blocksize=2, agg=numpy.median):
    """
    Parameters
    ----------
    coords : TYPE
        Description
    grid : TYPE
        Description
    extent : TYPE
        Description
    res : TYPE
        Description
    blocksize : int, optional
        Description
    agg : TYPE, optional
        Description

    Returns
    -------
    TYPE
        Description


    """
    # unpack extent
    left, right, top, bottom = grid_extent

     # unpack resolution
    xres, yres = res

    values = []
    for x, y in coords:
        # x, y to row and column indices
        col_left = int((x - left) / xres)
        row_top = int((y - top) / yres) # yres is < 0

        # add blocksize
        col_idx = slice(col_left, col_left + blocksize)
        row_idx = slice(row_top, row_top + blocksize)

        # sample grid and aggregate block values
        try:
            block = grid[row_idx, col_idx]
        except IndexError:
            block = np.nan
        values.append(agg(block))
    return values

