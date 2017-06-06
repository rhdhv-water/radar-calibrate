# -*- coding: utf-8 -*-
"""

"""
# Royal HaskoningDHV
from radar_calibrate.tests import testconfig

import matplotlib.pyplot as plt
import numpy as np

from itertools import cycle
import os

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
            linestyle='', marker=marker, markersize=8, alpha=0.5,
            label='{} ({:d}) '.format(timestamp, nstations[timestamp]))

    ax.plot([0., 10.], np.exp(np.array([0., 10.])), color='indianred', label='$e^{x}$')

    ax.grid(linestyle=':', linewidth=0.5, color='black')
    ax.set_yscale('log')

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel('$downscale\/\/factor$')
    ax.set_ylabel('$time\/(s)$')
    ttl = ax.set_title('$downscale\/\/factor\/\/vs\/\/time\/\/to\/\/execute\/\/Kriging_{KED}$')
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
    bg_shape = testconfig.BG_SHAPE

    fig, axes  = plt.subplots(nrows=3, ncols=2,
        figsize=(8.27, 11.7), sharex=True, sharey=True)

    axes[0, 0].imshow(aggregate, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[0, 0].set_title('$aggregate$')
    axes[0, 0].grid(linestyle= '--', linewidth=0.5)
    axes[0, 0].xaxis.set_ticklabels([])
    axes[0, 0].yaxis.set_ticklabels([])
    for x, y in gridtools.add_vector_layer(bg_shape):
        axes[0, 0].plot(x, y, color='skyblue', linewidth=0.5)

    axes[0, 1].imshow(calibrate, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[0, 1].set_title('$calibrate_{original}$')
    axes[0, 1].grid(linestyle= '--', linewidth=0.5)
    axes[0, 1].xaxis.set_ticklabels([])
    axes[0, 1].yaxis.set_ticklabels([])
    for x, y in gridtools.add_vector_layer(bg_shape):
        axes[0, 1].plot(x, y, color='skyblue', linewidth=0.5)

    axes[1, 0].imshow(calibrate_r, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[1, 0].set_title('$calibrate_R$')
    axes[1, 0].grid(linestyle= '--', linewidth=0.5)
    axes[1, 0].xaxis.set_ticklabels([])
    axes[1, 0].yaxis.set_ticklabels([])
    for x, y in gridtools.add_vector_layer(bg_shape):
        axes[1, 0].plot(x, y, color='skyblue', linewidth=0.5)

    axes[1, 1].imshow(abs((calibrate_r - calibrate) / calibrate)*100, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[1, 1].set_title('$calibrate_R - calibrate_{original}$')
    axes[1, 1].grid(linestyle= '--', linewidth=0.5)
    axes[1, 1].xaxis.set_ticklabels([])
    axes[1, 1].yaxis.set_ticklabels([])
    for x, y in gridtools.add_vector_layer(bg_shape):
        axes[1, 1].plot(x, y, color='skyblue', linewidth=0.5)

    axes[2, 0].imshow(calibrate_py, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[2, 0].set_title('$calibrate_{py}$')
    axes[2, 0].grid(linestyle= '--', linewidth=0.5)
    axes[2, 0].xaxis.set_ticklabels([])
    axes[2, 0].yaxis.set_ticklabels([])
    for x, y in gridtools.add_vector_layer(bg_shape):
        axes[2, 0].plot(x, y, color='skyblue', linewidth=0.5)

    im = axes[2, 1].imshow(abs((calibrate_py - calibrate) / calibrate)*100, cmap='inferno', vmin=0, vmax=40, alpha=0.8)
    axes[2, 1].set_title('$calibrate_{py} - calibrate_{original}$')
    axes[2, 1].grid(linestyle= '--', linewidth=0.5)
    axes[2, 1].xaxis.set_ticklabels([])
    axes[2, 1].yaxis.set_ticklabels([])
    for x, y in gridtools.add_vector_layer(bg_shape):
        axes[2, 1].plot(x, y, color='skyblue', linewidth=0.5)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.67, 0.02, 0.2])
    fig.colorbar(im, cax=cbar_ax)

    # plt.tight_layout()
    if imagefile is not None:
        plt.savefig(imagefile, bbox_inches='tight')
    else:
        plt.show()
