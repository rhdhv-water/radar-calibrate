# -*- coding: utf-8 -*-
"""tests for gridtools module.

Run using pytest:
> python -m pytest test_gridtools.py

"""
# Royal HaskoningDHV

from radar_calibrate import gridtools
from radar_calibrate.tests import testutils

from rasterio.enums import Resampling
import numpy as np


def test_sample_first():
    # test coordinates
    test_coords = [(25., 5.), ]  # second column, second row

    # test array
    test_array = np.array([[0, 1], [2, 3]])

    # geotransform (left, cellwidth, 0., top, 0., -cellheight)
    test_geotransform = 10., 10., 0., 20., 0., -10.

    # sample array
    samples = gridtools.sample_array(test_coords, test_array, test_geotransform,
        blocksize=1,
        agg=testutils.safe_first,
        )

    # compare result
    result = next(samples)
    assert np.isclose(result, 3.)


def test_sample_multiple():
    # test coordinates
    test_coords = [
            (5., 5.),  # 1. outside
            (25., 5.),  # 2. second column, second row
            (15., 7.),  # 3. first column, second row
            (29., 20.),  # 4. second column, first row
            (30., 20.),  # 5. outside
            ]

    # test array
    test_array = np.array([[0, 1], [2, 3]])

    # geotransform (left, cellwidth, 0., top, 0., -cellheight)
    test_geotransform = 10., 10., 0., 20., 0., -10.

    # sample array
    samples = gridtools.sample_array(test_coords, test_array, test_geotransform,
        blocksize=1,
        agg=testutils.safe_first,
        )

    # compare result
    result = np.array([s for s in samples])
    desired = [np.nan, 3., 2., 1., np.nan]
    np.testing.assert_allclose(result, desired)


def test_sample_median():
    # test coordinates
    test_coords = [
            (10., 20.),  # first row, first column
            ]

    # test array
    test_array = np.array([[0, 1], [2, 3]])

    # geotransform (left, cellwidth, 0., top, 0., -cellheight)
    test_geotransform = 10., 10., 0., 20., 0., -10.

    # sample array
    samples = gridtools.sample_array(test_coords, test_array, test_geotransform,
        blocksize=2,
        agg=np.median,
        )

    # compare result
    result = next(samples)
    assert np.isclose(result, 1.5)


def test_resample_nearest():
    nrows, ncols = 2, 50
    extent = 0., 50., 2., 0.
    to_cellsize = 10., 1.
    test_array = np.linspace(0., 50., num=100).reshape(nrows, ncols)
    test_grid = gridtools.BaseGrid(extent=extent, size=(ncols, nrows))
    resampled = gridtools.resample(test_array, test_grid,
        to_cellsize=to_cellsize,
        resampling=Resampling.nearest,
        )
    assert np.isclose(resampled[0, 0], test_array[0, 5])


def test_resample_average():
    nrows, ncols = 2, 50
    extent = 0., 50., 2., 0.
    to_cellsize = 10., 1.
    test_array = np.linspace(0., 50., num=100).reshape(nrows, ncols)
    test_grid = gridtools.BaseGrid(extent=extent, size=(ncols, nrows))
    resampled = gridtools.resample(test_array, test_grid,
        to_cellsize=to_cellsize,
        resampling=Resampling.average,
        )
    assert np.isclose(resampled[0, 0], test_array[0, 5].mean())
