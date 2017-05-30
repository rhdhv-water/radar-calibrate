#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Jonne Kleijer, Royal HaskoningDHV

import numpy

def grid_arjan():
#    cellwidth, cellheight = self.get_cellsize()
#    left, right, top, bottom = self.extent
    xcount, ycount = 20, 10
    left,right,top,bottom = 0,20,30,20
    cellwidth,cellheight = 1,1
        
    xmin = left + cellwidth / 2
    xmax = right - cellwidth / 2
    ymin = bottom + cellheight / 2
    ymax = top - cellheight / 2
    
    yi, xi = numpy.mgrid[
        ymax:ymin:ycount * 1j, xmin:xmax:xcount * 1j]
    return xi,yi

def grid():
#    cellwidth, cellheight = self.get_cellsize()
    ncols, nrows = 20, 10
    left,right,top,bottom = 0,20,30,20
    cellwidth,cellheight = 1,1
    
    xi = numpy.linspace(left + cellwidth/2, right - cellwidth/2, num=ncols)
    yi = numpy.linspace(top - cellheight/2, bottom + cellheight/2, num=nrows)
    
    
    yi, xi = numpy.meshgrid(yi, xi, indexing='xy')
    xi = xi.T
    yi = yi.T
    return xi,yi

