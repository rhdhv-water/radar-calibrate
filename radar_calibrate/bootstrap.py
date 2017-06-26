# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.calibration import Calibrator
from radar_calibrate import gridtools

import numpy as np

from multiprocessing import Pool
from collections import defaultdict
from functools import partial
import logging
import os

import pdb

log = logging.getLogger(os.path.basename(__file__))


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def bootstrap_worker(i, cal, method):
    wcal = cal.copy()
    station_coords, station_values = wcal.rainstations
    number_of_stations = wcal.number_of_stations

    selected_stations = np.arange(number_of_stations)!=i
    coords = station_coords[selected_stations],
    values = station_values[selected_stations]
    wcal.rainstations = coords, values

    wcal.interpolate(method=method)

    array = wcal.result["calibrate"].filled(np.nan)
    calibrate_values = gridtools.sample_array(
        coords=station_coords,
        array=array,
        geotransform=wcal.basegrid.get_geotransform(),
        )
    calibrate_values = [v for v in calibrate_values]

    # get values for the station left out
    calibrate_value = calibrate_values[i]
    station_value = station_values[i]
    return {
        'diff': calibrate_value - station_value,
        'rmse': rmse(calibrate_value, station_value)
        }

class BootStrappedCalibrator(Calibrator):
    def bootstrap_fraction(self, method, nsim=10, sample_fraction=0.8):
        station_coords, station_values = self.rainstations
        number_of_stations = self.number_of_stations

        result = []
        for i in range(nsim):
            log.info('simulation {i:d}/{nsim:d}'.format(i=i, nsim=nsim))
            random_floats = np.random.random_sample(size=number_of_stations)
            idx = random_floats < sample_fraction
            self.rainstations = station_coords[idx], station_values[idx]
            self.interpolate(method=method)
            if self.result is None:
                logging.warning('interpolation failed')
                continue
            calibrate_values = gridtools.sample_array(
                coords=station_coords[~idx],
                array=self.calibrate.filled(np.nan),
                geotransform=self.basegrid.get_geotransform(),
                )
            calibrate_values = [v for v in calibrate_values]
            result.append({
                'rmse': rmse(calibrate_values, station_values[~idx])
                })
        return result

    def bootstrap_single(self, method):
        station_coords, station_values = self.rainstations
        nsim = self.number_of_stations

        # initialize the result
        result = {"x":station_coords[:,0], "y":station_coords[:,1],
                  "diff":np.full(len(station_values),np.nan),
                  "rmse":np.full(len(station_values),np.nan)
                  }
        for i in range(nsim):
            log.info('simulation station {i:d}/{nsim:d}'.format(i=i+1, nsim=nsim))
            coords = station_coords[np.arange(len(station_coords))!=i]
            values = station_values[np.arange(len(station_values))!=i]

            self.rainstations = coords, values
            self.interpolate(method=method)
            array = self.result["calibrate"].filled(np.nan)
            calibrate_values = gridtools.sample_array(
                coords=station_coords,
                array=array,
                geotransform=self.basegrid.get_geotransform(),
                )
            calibrate_values = [v for v in calibrate_values]

            # get values for the station left out
            calibrate_value = calibrate_values[i]
            station_value = station_values[i]
            result['diff'][i] = calibrate_value - station_value
            result['rmse'][i] = rmse(calibrate_value, station_value)

        return result

    def bootstrap_single_mp(self, method, nproc=6):
        # create pool of processes
        pool = multiprocessing.Pool(nproc)

        # map to worker function
        index = np.arange(self.number_of_stations)
        worker = partial(bootstrap_worker, cal=self, method=method)
        pool.map(worker, index)

        # retrieve result
        pool.close()
        result = pool.join()

        # transpose
        transposedresult = defaultdict(list)
        for record in result:
            for key, value in record.items():
                transposedresult[key].append(value)

        return transposedresult
