# -*- coding: utf-8 -*-
# Royal HaskoningDHV

from radar_calibrate.tests import testconfig
from radar_calibrate import files

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np

from itertools import cycle
import logging
import os

log = logging.getLogger(os.path.basename(__file__))

def image(array,
    imagefile=None, figsize=None,
    features=None, color='black', linewidth=0.5,
    cmap='viridis', cmap_under='white', cmap_over=None, cmap_bad='white',
    vmin=1., vmax=30., alpha=0.8,
    title=None, no_ticklabels=True):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    bxa = []

    cmap = plt.get_cmap(cmap)
    if cmap_under is not None:
        cmap.set_under(cmap_under)
    if cmap_over is not None:
        cmap.set_over(cmap_over)
    if cmap_bad is not None:
        cmap.set_bad(cmap_bad)
    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

    if features is not None:
        for x, y in features:
            ax.plot(x, y, color=color, linewidth=linewidth)

    ax.grid(linestyle= '--', linewidth=linewidth)
    if no_ticklabels:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

    if title is not None:
        ttl = ax.set_title(title)
        bxa.append(ttl)

    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.3, pack_start=False)
    fig.add_axes(cax)
    fig.colorbar(im, cax=cax, orientation='vertical')

    if imagefile is not None:
        logging.debug('writing image to {file:}'.format(
            file=os.path.basename(imagefile)))
        plt.savefig(imagefile, bbox_inches='tight', bbox_extra_artists=bxa)
    else:
        plt.show()


def vgm_r(vgm_py, residual_py):
    figure = plt.figure()
    plt.plot(vgm_py[1,:], vgm_py[0,:], 'r*')
    plt.xlabel('distance')
    plt.ylabel('semivariance')
    plt.title('R variogram')
    return figure

def error(z, radar):
    #Plot error of rainstations vs radar in (mm)
    plt.figure()
    plt.scatter(z, radar)
    plt.plot([0, 40], [0, 40], 'k')
    plt.xlabel('$P_{station}\/(mm)$')
    plt.ylabel('$P_{radar}\/(mm)$')
    plt.axis([0, 40, 0, 40])
    plt.show()


def histogram(calibrate, calibrate_r, calibrate_py):
    # Plot histgram error
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(calibrate_r.flatten() - calibrate.flatten(), bins=range(-10, 10, 1))
    plt.xlim([-10, 10])
    plt.title('$histogram\/\/of\/\/error\/\/ked_R$')
    plt.subplot(2, 1, 2)
    plt.hist(calibrate_py.flatten() - calibrate.flatten(), bins=range(-10, 10, 1))
    plt.xlim([-10, 10])
    plt.title('$histogram\/\/of\/\/error\/\/ked_{py}$')
    plt.tight_layout()
    plt.show()

def timedresults(reshapes, results, nstations, imagefile=None, xlim=None, ylim=None):
    """
    Show a graph to understand relation of time to run the KED and the reshape size

    """

    fig, ax = plt.subplots()
    bxa = []
    markers = cycle(['o', 's', 'v', '+', 'd'])
    for result in results:
        marker = next(markers)
        for timestamp, values in sorted(result.items()):
            ax.plot(reshapes, values,
            linestyle='', marker=marker, markersize=8, alpha=0.5, zorder=3,
            label='{} ({:d}) '.format(timestamp, nstations[timestamp]))

    ax.plot([0., 10.], np.exp(np.array([0., 10.])), color='lightcoral', label='$e^{x}$', zorder=1)
    ax.plot([0., 10.], np.array([1, 11.])**2, color='skyblue', label='$x^{2} + 1$', zorder=1)

    ax.grid(linestyle=':', linewidth=0.5, color='black')
    ax.set_yscale('log')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel('$downscale\/\/factor$')
    ax.set_ylabel('$time\/(s)$')
    ttl = ax.set_title('$downscale\/\/factor\/\/vs\/\/time\/\/to\/\/execute\/\/Kriging_{KED}\/\/in\/\/R$')
    bxa.append(ttl)

    lgd = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    bxa.append(lgd)

    if imagefile is not None:
        plt.savefig(imagefile, bbox_inches='tight', bbox_extra_artists=bxa)
    else:
        plt.show()

def compare_ked(aggregate,
    calibrate, calibrate_r, calibrate_py,
    imagefile=None,
    ):
    fig, axes  = plt.subplots(nrows=3, ncols=2,
        figsize=(8.27, 11.7), sharex=True, sharey=True)

    axes[0, 0].imshow(aggregate, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[0, 0].set_title('$aggregate$')
    axes[0, 0].grid(linestyle= '--', linewidth=0.5)
    axes[0, 0].xaxis.set_ticklabels([])
    axes[0, 0].yaxis.set_ticklabels([])
    for x, y in files.read_shape(testconfig.BG_SHAPEFILE):
        axes[0, 0].plot(x, y, color='skyblue', linewidth=0.5)

    axes[0, 1].imshow(calibrate, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[0, 1].set_title('$calibrate_{original}$')
    axes[0, 1].grid(linestyle= '--', linewidth=0.5)
    axes[0, 1].xaxis.set_ticklabels([])
    axes[0, 1].yaxis.set_ticklabels([])
    for x, y in files.read_shape(testconfig.BG_SHAPEFILE):
        axes[0, 1].plot(x, y, color='skyblue', linewidth=0.5)

    axes[1, 0].imshow(calibrate_r, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[1, 0].set_title('$calibrate_R$')
    axes[1, 0].grid(linestyle= '--', linewidth=0.5)
    axes[1, 0].xaxis.set_ticklabels([])
    axes[1, 0].yaxis.set_ticklabels([])
    for x, y in files.read_shape(testconfig.BG_SHAPEFILE):
        axes[1, 0].plot(x, y, color='skyblue', linewidth=0.5)

    axes[1, 1].imshow(abs((calibrate_r - calibrate) / calibrate)*100, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[1, 1].set_title('$calibrate_R - calibrate_{original}$')
    axes[1, 1].grid(linestyle= '--', linewidth=0.5)
    axes[1, 1].xaxis.set_ticklabels([])
    axes[1, 1].yaxis.set_ticklabels([])
    for x, y in files.read_shape(testconfig.BG_SHAPEFILE):
        axes[1, 1].plot(x, y, color='skyblue', linewidth=0.5)

    axes[2, 0].imshow(calibrate_py, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[2, 0].set_title('$calibrate_{py}$')
    axes[2, 0].grid(linestyle= '--', linewidth=0.5)
    axes[2, 0].xaxis.set_ticklabels([])
    axes[2, 0].yaxis.set_ticklabels([])
    for x, y in files.read_shape(testconfig.BG_SHAPEFILE):
        axes[2, 0].plot(x, y, color='skyblue', linewidth=0.5)

    im = axes[2, 1].imshow(abs((calibrate_py - calibrate) / calibrate)*100, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[2, 1].set_title('$calibrate_{py} - calibrate_{original}$')
    axes[2, 1].grid(linestyle= '--', linewidth=0.5)
    axes[2, 1].xaxis.set_ticklabels([])
    axes[2, 1].yaxis.set_ticklabels([])
    for x, y in files.read_shape(testconfig.BG_SHAPEFILE):
        axes[2, 1].plot(x, y, color='skyblue', linewidth=0.5)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.67, 0.02, 0.2])
    fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout()
    if imagefile is not None:
        plt.savefig(imagefile, bbox_inches='tight')
    else:
        plt.show()

def bootstrap(result, imagefile=None, zrange=(-10,10)):
    """
    Show a map to visualize the result of the bootstrap method for each rainstation

    INPUT
    results [dict] comprises of 3 arrays: x, y, difference
    imagefile [str] path to image file
    zrange [tup] (zmin, zmax) which is by default -10 to 10
    """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Result bootstrap {ts}'.format(ts=imagefile[-18:-6]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(linestyle= '--', linewidth=0.5)

    # plot calibrate - measured values
    cmap = plt.cm.get_cmap('coolwarm',zrange[1]-zrange[0])
    norm = colors.BoundaryNorm(range(zrange[0],zrange[1]), cmap.N)
    im = ax.scatter(result['x'],result['y'],c=result['diff'], cmap=cmap, norm=norm)

    # plot background shapefile
    for x, y in files.read_shape(testconfig.BACKGROUND_SHAPEFILE):
        ax.plot(x, y, color='skyblue', linewidth=0.5)

    fig.gca().set_aspect('equal', adjustable='box')
    cbar = fig.colorbar(im)
    cbar.ax.set_ylabel('Difference calibration - measured [mm/day]', rotation=270, labelpad=15)

    if imagefile is not None:
        logging.debug('writing image to {file:}'.format(
                file=os.path.basename(imagefile)))
        plt.savefig(imagefile, bbox_inches='tight')
    else:
        plt.show()
