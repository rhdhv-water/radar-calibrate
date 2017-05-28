# -*- coding: utf-8 -*-
"""

"""
# Royal HaskoningDHV

import matplotlib.pyplot as plt

def vgm_r(vgm_py, residual_py):
    figure = plt.figure()
    plt.plot(vgm_py[1,:], vgm_py[0,:], 'r*')
    plt.xlabel('distance')
    plt.ylabel('semivariance')
    plt.title('R variogram')
    return figure


def compare_ked(z, radar, aggregate, calibrate, calibrate_r, calibrate_py):
    #Plot error of rainstations vs radar in (mm)
    plt.figure()
    plt.scatter(z, radar)
    plt.plot([0, 40], [0, 40], 'k')
    plt.xlabel('$P_{station}\/(mm)$')
    plt.ylabel('$P_{radar}\/(mm)$')
    plt.axis([0, 40, 0, 40])
    plt.show()

    # Plot radar_calibrate_R
    plt.figure()
    f221 = plt.subplot(2, 2, 1)
    plt.imshow(aggregate, cmap='rainbow', vmin=0, vmax=40)
    plt.ylabel('y-coordinate')
    plt.title('aggregate')
    f222 = plt.subplot(2, 2, 2, sharex=f221, sharey=f221)
    plt.imshow(calibrate, cmap='rainbow', vmin=0, vmax=40)
    plt.title('$calibrate_{original}$')
    f223 = plt.subplot(2, 2, 3, sharex=f221, sharey=f221)
    plt.imshow(calibrate_r, cmap='rainbow', vmin=0, vmax=40)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.title('$calibrate_R$')
    plt.subplot(2, 2, 4, sharex=f221, sharey=f221)
    plt.imshow(calibrate_py, cmap='rainbow', vmin=0, vmax=40)
    plt.xlabel('x-coordinate')
    plt.title('$calibrate_{py}$')
    plt.tight_layout()
    plt.show()

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
