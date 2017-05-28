# -*- coding: utf-8 -*-
"""tests for grid tools module.

Run using pytest:
> python -m pytest test_gridtools.py

"""
# Royal HaskoningDHV

from radar_calibrate import gridtools
from radar_calibrate import utils

import numpy


def test_sample_first():
    # test coordinates
    test_coords = [(25., 5.), ]  # second column, second row

    # test array
    test_grid = numpy.array([[0, 1], [2, 3]])

    # geotransform (left, cellwidth, 0., top, 0., -cellheight)
    test_geotransform = 10., 10., 0., 20., 0., -10.

    # sample grid
    samples = gridtools.sample_grid(test_coords, test_grid, test_geotransform,
        blocksize=1,
        agg=utils.safe_first,
        )

    # compare result
    result = next(samples)
    assert numpy.isclose(result, 3.)


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
    test_grid = numpy.array([[0, 1], [2, 3]])

    # geotransform (left, cellwidth, 0., top, 0., -cellheight)
    test_geotransform = 10., 10., 0., 20., 0., -10.

    # sample grid
    samples = gridtools.sample_grid(test_coords, test_grid, test_geotransform,
        blocksize=1,
        agg=utils.safe_first,
        )

    # compare result
    result = numpy.array([s for s in samples])
    desired = [numpy.nan, 3., 2., 1., numpy.nan]
    numpy.testing.assert_allclose(result, desired)


def test_sample_median():
    # test coordinates
    test_coords = [
            (10., 20.),  # first row, first column
            ]

    # test array
    test_grid = numpy.array([[0, 1], [2, 3]])

    # geotransform (left, cellwidth, 0., top, 0., -cellheight)
    test_geotransform = 10., 10., 0., 20., 0., -10.

    # sample grid
    samples = gridtools.sample_grid(test_coords, test_grid, test_geotransform,
        blocksize=2,
        agg=numpy.median,
        )

    # compare result
    result = next(samples)
    assert numpy.isclose(result, 1.5)
