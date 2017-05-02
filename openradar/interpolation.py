# (c) Nelen & Schuurmans.  GPL licensed, see LICENSE.rst.
# no utf-8 encoding stuff yet. CSV library becomes a nuisance

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import csv
import logging
import os
import zipfile

import numpy

from openradar import gridtools
from openradar import grounddata


class RainStation(object):
    '''
    Combine information of a weather station (lat,lon and sationid), with
    a bunch of measurements.
    '''
    def __init__(self, station_id, lat, lon, measurement, klasse):
        self.station_id = station_id
        self.lat = float(lat)
        self.lon = float(lon)
        self.klasse = int(klasse)
        self.measurement = measurement


class DataLoader(object):
    '''
    Dataloader accepts and processes the ground station data as well as the
    radar datasets.
    The input format of the ground station data is in csv. The Dataloader
    serves as a prerequisite for the Interpolator.
    '''
    def __init__(self, metafile, aggregate, timeframe):
        # Attributes
        self.delta = aggregate.timedelta
        self.date = aggregate.datetime
        self.timeframe = timeframe
        self.dataset = aggregate.get()
        # Basegrid
        self.basegrid = gridtools.BaseGrid(
            extent=self.dataset.attrs['grid_extent'],
            projection=self.dataset.attrs['grid_projection'],
            size=self.dataset.attrs['grid_size']
        )
        # Read csv file
        self.stationsdata = self.read_csv(metafile.encode('utf-8'))

    @classmethod
    def read_csv(cls, filename):
        '''
        This is quite a generic method. But because csv library does not read
        unicode it is specifically in this interpolation.py

        Tries to read file from zipfile with same name if possible,
        otherwise reads the plaintext file.
        '''
        basename = os.path.basename(filename)

        # Try to return from zipfile
        zippath = filename[:-3] + 'zip'
        if os.path.exists(zippath):
            with zipfile.ZipFile(zippath) as zip_file:
                with zip_file.open(basename) as csvfile:
                    reader = csv.reader(csvfile)
                    data = [i for i in reader]
                    logging.debug('Returned {} from zip'.format(basename))
                    return data

        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            data = [i for i in reader]
            logging.debug('Returned {} from plaintext'.format(basename))
            return data

    def processdata(self, skip=None, klasse=1):
        '''
        Takes apart the csv and creates instances of RainStation that can be
        used for the data analysis.
        '''
        dbdata = grounddata.get_rainstations(
            datetime=self.date,
            timeframe=self.timeframe)
        self.rainstations = []
        for d in dbdata:
            self.rainstations.append(RainStation(
                station_id=d['id'],
                lat=d['coords'][1],
                lon=d['coords'][0],
                klasse=0,
                measurement=-999. if d['value'] is None else d['value'],
            ))

    def stations_dummies(self):
        data = self.stationsdata
        xcol = data[0].index('X')
        ycol = data[0].index('Y')
        klassecol = data[0].index('KWALITEIT')
        idcol = data[0].index('ID')
        self.stations = []
        for line in data[1:]:
            self.stations.append(
                RainStation(line[idcol], line[xcol], line[ycol],
                            1.0, line[klassecol]))


class Interpolator(object):
    '''
    This where a few Interpolation techniques can be compared.
    Takes the data from the data handler and processes the different csv's
    to an interpolated grid.

    Interpolation techniques:
    * Inverse Distance Weighting
    * Kriging (colocated cokriging, kriging external drift, and ordinary)
    * Linear Rbf
    '''
    def __init__(self, dataloader):
        self.dataloader = dataloader
        precipitation = self.dataloader.dataset['precipitation'][:]
        self.precipitation = numpy.array(precipitation)

    def get_rain_and_location(self):
        '''
        Seperates the measurements location and data. Also requests the classes
        of each weather stations. These are now by default set to 1.
        '''
        xy = numpy.array([[i.lon, i.lat]
                          for i in self.dataloader.rainstations])
        z = numpy.array([i.measurement for i in self.dataloader.rainstations])
        klasse = numpy.array([i.klasse for i in self.dataloader.rainstations])
        mask = numpy.equal(z, -999.0)
        self.mask = mask
        z = numpy.ma.array(z, mask=mask)
        x, y = xy.T[0], xy.T[1]
        return x[~mask], y[~mask], z.data[~mask], klasse[~mask]

    def get_dummies(self):
        '''
        For correction field calculation dummies are needed to fill the gaps.
        '''
        self.dataloader.stations_dummies()
        xyz = numpy.ma.array([[i.lon, i.lat, i.measurement]
                              for i in self.dataloader.stations])
        return xyz.T

    def get_id(self):
        '''
        Get only the ids
        '''
        station_id = numpy.array(
            [i.station_id for i in self.dataloader.rainstations])
        return station_id[~self.mask]

    def get_correction_factor(self):
        '''
        Make a correction factor grid based on the input defined in
        the radar pixels and the weatherstations.
        Returns a numpy array with Correction Factors.
        '''
        x, y, z, klasse = self.get_rain_and_location()
        radar = numpy.array(self.get_radar_for_locations())

        # Calculate correction factor, but limit extreme cases
        zero_radar = radar == 0
        correction_factor = numpy.empty(z.shape)
        correction_factor[zero_radar] = 1.0
        correction_factor[~zero_radar] = z[~zero_radar] / radar[~zero_radar]
        correction_factor[correction_factor > 10] = 1.0
        correction_factor[correction_factor < 0.0] = 1.0

        stationsx, stationsy, stationsz = self.get_dummies()
        for i in range(len(z)):
            stationsz[stationsx == x[i]] = correction_factor[i]
        self.create_interpolation_grid()
        correction_factor_grid = self.get_idwgrid(
            stationsx, stationsy, stationsz)
        return correction_factor_grid

    def get_calibrated(self):
        '''
        Make a correction factor grid based on the input defined in
        the radar pixels and the weatherstations.
        Returns a numpy array with Correction Factors.
        '''
        x, y, z, klasse = self.get_rain_and_location()
        self.create_interpolation_grid()
        krige_grid = self.ked(x, y, z)
        return krige_grid

    def get_radar_for_locations(self, rasterdata=None, size=2):
        '''
        Radar "pixel"values for location closest to weather station.
        Returns those pixels that are closest to the rain stations
        '''
        basegrid = self.dataloader.basegrid
        if rasterdata is not None:
            rasterband = rasterdata.T
        else:
            rasterband = self.precipitation.T
        xy = numpy.array(self.get_rain_and_location()[0:2]).T
        geotransform = basegrid.get_geotransform()
        origx = geotransform[0]
        origy = geotransform[3]
        pixelwidth = geotransform[1]
        pixelheight = geotransform[5]
        radar_pixels = []
        for i in range(len(xy)):
            xoff = int((xy[i][0] - origx) / pixelwidth)
            yoff = int((xy[i][1] - origy) / pixelheight)
            size = size  # how many surrounding pixels you want to include
            data = rasterband[xoff:xoff + size, yoff:yoff + size]
            radar_pixels.append(numpy.median(data))
        return radar_pixels

    def create_interpolation_grid(self):
        '''
        Run this to get some basic stuff, like an empty grid to interpolated
        the data to, The extent and size of the tifs.
        '''
        basegrid = self.dataloader.basegrid
        self.nx, self.ny = basegrid.size
        self.xi, self.yi = [numpy.float32(array).flatten()
                            for array in basegrid.get_grid()]

    def get_idwgrid(self, x, y, z, p=2):
        '''
        This function returns a idwgrid with inputs x,y location and z
        as the to be interpolated value
        '''
        nx, ny = self.nx, self.ny
        xi, yi = self.xi, self.yi
        grid = self.simple_idw(x, y, z, xi, yi, p)
        grid = grid.reshape((ny, nx))
        return grid

    def get_linear_rbf(self, x, y, z):
        '''
        This function returns a idwgrid with inputs x,y location and z
        as the to be interpolated value
        '''
        nx, ny = self.nx, self.ny
        xi, yi = self.xi, self.yi
        grid = self.linear_rbf(x, y, z, xi, yi)
        grid = grid.reshape((ny, nx))
        return grid

    def simple_idw(self, x, y, z, xi, yi, p):
        '''
        Simple idw function. Slow, but memory efficient implementation.
        '''
        sum_of_weights = numpy.zeros(xi.shape)
        sum_of_weighted_gauges = numpy.zeros(xi.shape)
        for i in range(x.size):
            distance = numpy.sqrt((x[i] - xi) ** 2 + (y[i] - yi) ** 2)
            weight = 1.0 / distance ** p
            weighted_gauge = z[i] * weight
            sum_of_weights += weight
            sum_of_weighted_gauges += weighted_gauge
        zi = sum_of_weighted_gauges / sum_of_weights

        return zi

    def linear_rbf(self, x, y, z, xi, yi):
        '''
        Linear Rbf interpolation.
        http://en.wikipedia.org/wiki/Radial_basis_function
        '''
        dist = self.distance_matrix(x, y, xi, yi)
        # Mutual pariwise distances between observations
        internal_dist = self.distance_matrix(x, y, x, y)
        # Now solve for the weights such that mistfit at
        # the observations is minimized
        weights = numpy.linalg.solve(internal_dist, z)
        # Multiply the weights for each interpolated point by the distances
        zi = numpy.ma.dot(dist.T, weights)
        return zi

    def correction(self, ab):
        inhours = self.dataloader.delta.total_seconds()/60/60
        a, b = ab
        return a * inhours ** b

    def kriging_in_r(self, x, y, z):
        '''
        Cokriging (and ordinary kriging) is quite fast in R. This would
        anyway be more pragmatic than rewriting/porting it to Python.
        For the moment this will be the 'best' way as R makes it very easy to
        use kriging without fitting a variogram model, but using a standard
        variogram.
        '''
        self.create_interpolation_grid()
        import rpy2.robjects as robj
        robj.r.library('gstat')
        self.create_interpolation_grid()
        xi, yi = (robj.FloatVector(self.xi.tolist()),
                  robj.FloatVector(self.yi.tolist()))
        radar = self.get_radar_for_locations()
        radar = robj.FloatVector(radar)
        x, y, z = robj.FloatVector(x), robj.FloatVector(y), robj.FloatVector(z)
        rain_frame = robj.DataFrame({'x': x, 'y': y, 'z': z})
        target_frame = robj.DataFrame({'x': xi, 'y': yi})
        doy = self.dataloader.date.timetuple().tm_yday
        x_sill = [0.84, -0.25]
        a_sill = [0.20, -0.37]
        t0sill = [162, -0.03]
        x_range = [15.51, 0.09]
        a_range = [2.06, -0.12]
        t0range = [7.37, 0.22]
        sill_ = (self.correction(x_sill)+self.correction(a_sill) *
                 numpy.cos(2 * numpy.pi * 1/365*(doy-self.correction(t0sill)))
                 )**4
        range_ = (self.correction(x_range) +
                  self.correction(a_range) *
                  numpy.cos(2 * numpy.pi * 1/365*(doy-self.correction(t0range))
                            )
                  )**4
        vgm_args = {
            'nugget': 0,
            'model_type': 'Exp',
            'sill': sill_,
            'range': range_,
        }
        v = robj.r.vgm(vgm_args['sill'], vgm_args['model_type'],
                       vgm_args['range'], vgm_args['nugget'])
        krige = robj.r('NULL')
        krige = robj.r.gstat(krige, "rain", robj.r('z ~ 1'), robj.r('~ x + y'),
                             data=rain_frame, model=v, nmax=40)
        result = robj.r.predict(krige, target_frame)
        kriged_est = numpy.array(result[2])
        kriged_est = kriged_est.reshape((self.ny, self.nx))
        return kriged_est

    def cokriging_in_r(self, x, y, z):
        '''
        Cokriging (and ordinary kriging) is quite fast in R. This would anyway
        be more pragmatic than rewriting/porting it to Python.
        For the moment this will be the 'best' way as R makes it very easy to
        use kriging without fitting a variogram model, but using a standard
        variogram.
        '''
        import rpy2.robjects as robj
        robj.r.library('gstat')
        self.create_interpolation_grid()
        xi, yi = (robj.FloatVector(self.xi.tolist()),
                  robj.FloatVector(self.yi.tolist()))
        dataset = self.precipitation.flatten()
        rxi = robj.FloatVector(dataset.tolist())
        radar = self.get_radar_for_locations()
        radar = robj.FloatVector(radar)
        x, y, z = robj.FloatVector(x), robj.FloatVector(y), robj.FloatVector(z)
        rain_frame = robj.DataFrame({'x': x, 'y': y, 'z': z})
        radar_frame = robj.DataFrame({'x': xi, 'y': yi, 'radar': rxi})
        target_frame = robj.DataFrame({'x': xi, 'y': yi})
        doy = self.dataloader.date.timetuple().tm_yday
        x_sill = [0.84, -0.25]
        a_sill = [0.20, -0.37]
        t0sill = [162, -0.03]
        x_range = [15.51, 0.09]
        a_range = [2.06, -0.12]
        t0range = [7.37, 0.22]
        sill_ = (self.correction(x_sill)+self.correction(a_sill) *
                 numpy.cos(2 * numpy.pi * 1/365*(doy-self.correction(t0sill)))
                 )**4
        range_ = (self.correction(x_range)+self.correction(a_range) *
                  numpy.cos(2 * numpy.pi * 1/365*(doy-self.correction(t0range))
                            )
                  )**4
        vgm_args = {
            'nugget': 0,
            'model_type': 'Exp',
            'sill': sill_,
            'range': range_,
        }
        v = robj.r.vgm(vgm_args['sill'], vgm_args['model_type'],
                       vgm_args['range'], vgm_args['nugget'])
        correlation_radar_rain = numpy.abs(robj.r.cor(z, radar))
        if str(correlation_radar_rain[0]) != 'nan':
            # The variogram is combined. This is a bit awkward in Rpy.
            # So one way is change the args manually (see below)
            # or load variables in R before hand.
            variogram_radar = v
            variogram_rain = v
            cck = robj.r('NULL')
            cck = robj.r.gstat(cck, "rain", robj.r('z ~ 1'), robj.r('~ x + y'),
                               data=rain_frame, model=variogram_rain, nmax=40)
            cck = robj.r.gstat(cck, "radar", robj.r('radar~ 1'),
                               robj.r('~ x + y'),
                               data=radar_frame, model=variogram_radar,
                               merge=robj.r("c('rain','radar')"), nmax=1)
            cck = robj.r.gstat(cck, robj.r('c("rain", "radar")'),
                               model=v, nmax=40)
            result = robj.r.predict(cck, target_frame)
            self.crossval_cck = robj.r('gstat.cv')(cck)
            rain_est = numpy.array(result[2])
            rain_est = rain_est.reshape((self.ny, self.nx))
        else:
            rain_est = dataset.reshape((self.ny, self.nx))
        return rain_est

    def ked(self, x, y, z):
        '''
        Kriging External Drift (or universal kriging).
        Inputs should be equally long.

        TODO: take out general R stuff that CCK and OK also use.
        '''
        import rpy2.robjects as robj
        robj.r.library('gstat')
        self.create_interpolation_grid()
        xi, yi = (robj.FloatVector(self.xi.tolist()),
                  robj.FloatVector(self.yi.tolist()))
        dataset = self.precipitation.flatten()
        rxi = robj.FloatVector(dataset.tolist())
        radar = self.get_radar_for_locations()

        # Modification to prevent singular matrix (Tom)
        radar += (1e-9 * numpy.random.rand(len(radar)))

        radar = robj.FloatVector(radar)
        x, y, z = robj.FloatVector(x), robj.FloatVector(y), robj.FloatVector(z)
        rain_radar_frame = robj.DataFrame({'x': x, 'y': y, 'z': z,
                                           'radar': radar})
        radar_frame = robj.DataFrame({'x': xi, 'y': yi, 'radar': rxi})
        try:
            vgm = robj.r.variogram(robj.r("z~radar"), robj.r('~ x + y'),
                                   data=rain_radar_frame,
                                   cutoff=50000, width=5000)
            residual_callable = robj.r('fit.variogram')
            residual = residual_callable(vgm, robj.r.vgm(1, 'Sph', 25000, 1))
            ked = robj.r('NULL')
            ked = robj.r.gstat(ked, 'raingauge', robj.r("z~radar"),
                               robj.r('~ x + y'),
                               data=rain_radar_frame,
                               model=residual, nmax=40)
            result = robj.r.predict(ked, radar_frame, nsim=0)
            rain_est = numpy.array(result[2])
            # self.crossval_ked = robj.r('gstat.cv')(ked)
        except:
            logging.exception('Exception during kriging:')
            rain_est = dataset
        rain_est = rain_est.reshape((self.ny, self.nx))

        # handle extreme outcomes of kriging
        zero_or_no_data = numpy.logical_or(
            self.precipitation == 0, self.precipitation == -9999
        )
        correction_factor = numpy.ones(self.precipitation.shape)
        correction_factor[~zero_or_no_data] = (
            rain_est[~zero_or_no_data] / self.precipitation[~zero_or_no_data]
        )
        leave_uncalibrated = numpy.logical_or(
            correction_factor < 0, correction_factor > 10
        )
        logging.info('Leaving {} extreme pixels uncalibrated.'.format(
            leave_uncalibrated.sum(),
        ))
        rain_est[leave_uncalibrated] = self.precipitation[leave_uncalibrated]

        return rain_est

    def distance_matrix(self, x0, y0, x1, y1):
        '''
        Calculate the distance matrix
        '''
        obs = numpy.float32(numpy.vstack((x0, y0))).T
        interp = numpy.float32(numpy.vstack((x1, y1))).T

        # Make a distance matrix between pairwise observations
        # Note: from <http://stackoverflow.com/questions/1871536>
        # (Yay for ufuncs!)
        d0 = numpy.subtract.outer(obs[:, 0], interp[:, 0])
        d1 = numpy.subtract.outer(obs[:, 1], interp[:, 1])

        return numpy.hypot(d0, d1)
