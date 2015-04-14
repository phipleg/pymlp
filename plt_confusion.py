import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.ticker import MultipleLocator

import time

majorLocator   = MultipleLocator(1)
def draw_confusion_matrix(fig, ax,  conf_matrix):
    ax.xaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_locator(majorLocator)
    conf_norm = [ row * 1.0 / np.sum(row) for row in conf_matrix]
    ax.set_aspect(1)
    res = ax.imshow(np.array(conf_norm), cmap=plt.cm.jet, interpolation='nearest')
    cb = fig.colorbar(res)

if __name__ == "__main__":
    plt.ion()
    plt.show()
    fig = plt.figure()
    for i in xrange(10):
        conf = np.random.rand(4,4)
        plt.clf()
        ax = fig.add_subplot(111)
        draw_confusion_matrix(fig, ax, conf)
        plt.draw()
        time.sleep(0.01)

