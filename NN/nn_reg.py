from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd.misc import flatten


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def nn_predict(params, inputs, nonlinearity=np.tanh):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = nonlinearity(outputs)
    return outputs

def log_gaussian(params, scale):
    flat_params, _ = flatten(params)
    return np.sum(norm.logpdf(flat_params, 0, scale))

def logprob(weights, inputs, targets, noise_scale=0.1):
    predictions = nn_predict(weights, inputs)
    return np.sum(norm.logpdf(predictions, targets, noise_scale))

def build_toy_dataset(n_data=80, noise_std=0.1):
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data//2),
                              np.linspace(6, 8, num=n_data//2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs[:, np.newaxis]
    targets = targets[:, np.newaxis] / 2.0
    return inputs, targets