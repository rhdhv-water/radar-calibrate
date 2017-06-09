# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.calibration import Calibrator
from radar_calibrate import gridtools

import numpy as np

import logging
import os

log = logging.getLogger(os.path.basename(__file__))


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


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
        pass
