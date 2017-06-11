# -*- coding: utf-8 -*-
# Royal HaskoningDHV


from openradar.gridtools import BaseGrid as OpenRadarBaseGrid
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject
from affine import Affine
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

        # sampe array and aggregate block values
        try:
            block = array[row_idx, col_idx]
        except IndexError:
            block = np.nan

        yield agg(block)


def resample(array, basegrid, to_cellsize,
    resampling=Resampling.nearest, epsg=28992):
    '''resample array to new cellsize using rasterio warp'''
    # unpack extent
    left, right, top, bottom = basegrid.extent

    # cellsize to scaling
    cellwidth, cellheight = to_cellsize

    # construct Affine geotransforms
    src_transform = Affine.from_gdal(*basegrid.get_geotransform())
    dst_transform = (
        Affine.translation(left, top) * Affine.scale(cellwidth, -cellheight)
        )

    # crs
    src_crs = CRS.from_epsg(epsg)
    dst_crs = src_crs

    # initialize destination array
    new_grid = basegrid.rescale(to_cellsize)
    new_ncols, new_nrows = new_grid.size
    resampled = np.empty((new_nrows, new_ncols))

    # fill masked values with NaN
    array_filled = np.ma.filled(array, fill_value=np.nan)

    # reproject
    reproject(array_filled, resampled,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
        )

    # mask NaN values
    resampled_masked = np.ma.masked_invalid(resampled)

    return resampled_masked


class BaseGrid(OpenRadarBaseGrid):
    def get_grid(self):
        '''get x, y grid coordinates as 1-D vectors'''
        left, right, top, bottom = self.extent

        cellwidth, cellheight = self.get_cellsize()
        ncols, nrows = self.size

        xi = np.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
        yi = np.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)
        return xi, yi

    def rescale(self, to_cellsize):
        '''return rescaled grid with given cellsize (cellheight is negative)'''
        left, right, top, bottom = self.extent
        new_cellwidth, new_cellheight = to_cellsize

        new_ncols = int((right - left) / new_cellwidth)
        new_nrows = int((top - bottom) / new_cellheight)
        new_size = new_ncols, new_nrows

        return BaseGrid(extent=self.extent, size=new_size)
