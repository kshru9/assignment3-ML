from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from numpy.lib.npyio import load
from NN.nn_reg import init_random_params, build_toy_dataset, log_gaussian, logprob, nn_predict

from sklearn.datasets import load_boston

def run():

    init_scale = 0.1
    weight_prior_variance = 10.0
    init_params = init_random_params(init_scale, layer_sizes=[13, 4, 4, 1])

    inputs, targets = load_dataset()

    def objective(weights, t):
        return -logprob(weights, inputs, targets)\
               -log_gaussian(weights, weight_prior_variance)

    print(grad(objective)(init_params, 0))

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(params, t, g):
        print("Iteration {} log likelihood {}".format(t, -objective(params, t)))

        # Plot data and functions.
        plt.cla()
        ax.plot(inputs.ravel(), targets.ravel(), 'bx', ms=12)
        plot_inputs = np.reshape(np.linspace(-7, 7, num=300), (300,1))
        outputs = nn_predict(params, plot_inputs)
        ax.plot(plot_inputs, outputs, 'r', lw=3)
        ax.set_ylim([-1, 1])
        plt.draw()
        plt.pause(1.0/60.0)

    print("Optimizing network parameters...")
    optimized_params = adam(grad(objective), init_params,
                            step_size=0.01, num_iters=1000, callback=callback)


def load_dataset():
    data = load_boston()
    X = data.data
    y = data.target

    return X,y

if __name__ == '__main__':
    run()