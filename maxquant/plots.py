__author__ = 'mfitzp'

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np

def correlation(cdf, cm=cm.Reds, vmin=None, vmax=None):
    data = cdf.values

    # Plot the distributions
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(1,1,1)

    if vmin is None:
        vmin = np.min(data)

    if vmax is None:
        vmax = np.max(data)

    i = ax.imshow(data, cmap=cm, vmin=vmin, vmax=vmax, interpolation='none')
    fig.colorbar(i)

    return fig
