# -*- coding: utf-8 -*-
"""Summary
"""
# Royal HaskoningDHV
from radar_calibrate import config

import h5py

from collections import namedtuple
import time
import os

Timedresult = namedtuple('timedresult', 'dt result')


def timethis(func):
    """Function or method decorator measuring execution time

    See https://www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-funcs

    Parameters
    ----------
    func : Callable
        Function or method to be timed

    Returns
    -------
    Timedresult
        Named tuple containing execution time and result
    """
    def timed(*args, **kwargs):
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        dt = toc - tic
        return Timedresult(dt=dt, result=result)

    return timed


def safe_first(array):
    try:
        return array.flatten()[0]
    except IndexError:
        return numpy.nan

countrymask_path = os.path.join(config.MISCDIR, 'countrymask.h5')
with h5py.File(countrymask_path, 'r') as ds:
    mask = ds['mask'][:]

def apply_countrymask(calibrate, aggregate):
    """
    Get a prefabricated country mask, 1 everywhere in the country and 0
    50 km outside the country. If extent and cellsize are not as in
    config.py, this breaks.
    """
    return mask * calibrate + (1 - mask) * aggregate
