import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def draw_pixels(fig, ax, pixel_sequences, inner, outer):
    m = np.zeros((inner[1] * outer[1], inner[0] * outer[0]))
    for k, pixel_seq in enumerate(pixel_sequences):
        oy = k / outer[0]
        ox = k % outer[0]
        for i,c in enumerate(pixel_seq):
            y = i / inner[0] + oy * inner[0]
            x = i % inner[0] + ox * inner[0]
            row = m[y]
            row[x] = c

    xmajorLocator = MultipleLocator(inner[0])
    ymajorLocator = MultipleLocator(inner[1])
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.grid(True)
    res = ax.imshow(m, cmap=plt.cm.Greys, interpolation='nearest')
    cb = fig.colorbar(res, cax=None, ax=ax)

if __name__ == "__main__":
    plt.ion()
    plt.show()
    fig = plt.figure()
    for i in xrange(10):
        pixel_sequence = [np.random.rand(784,1) for x in xrange(64)]
        plt.clf()
        ax = fig.add_subplot(111)
        draw_pixels(fig, ax, pixel_sequence, [28,28], [8,8])
        plt.draw()

