# -*- coding: utf-8 -*-
"""Summary
"""
# Royal HaskoningDHV

from collections import namedtuple
import time

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
