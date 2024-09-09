import matplotlib.pyplot as plt
import numpy as np

from util.math import sigmoid

if __name__ == '__main__':
    x = np.linspace(20, 40, 10000)

    for sl in [0.5, 1.0, 1.5, 2.0, 3.0]:
        y = sigmoid(x, sl, 30.0)
        plt.plot(x, y, label=f'slope={sl}')

    plt.grid()
    plt.legend()
    plt.show()