import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    distribution = tfp.distributions.Mixture(
        cat=tfp.distributions.Categorical(probs=[0.1, 0.2, 0.6, 0.1]),
        components=[
            tfp.distributions.Normal(loc=0.1, scale=0.01),
            tfp.distributions.Normal(loc=0.2, scale=0.01),
            tfp.distributions.Normal(loc=0.3, scale=0.01),
            tfp.distributions.Normal(loc=0.4, scale=0.01),
        ]
    )
    r = np.linspace(0, 0.5, 100)
    y = distribution.prob(r)
    print(y)
    plt.plot(r, y)
    # plt.xscale('log')
    plt.show()