import numpy as np
import autograd.numpy as anp
from autograd import grad, elementwise_grad

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class RMSE(Loss):
    def __init__(self) -> None:
        pass

    def loss(self, y, p):
        # temp = np.diff(y-p)
        return 1/len(y) * anp.sum((y-p)**2)

    def gradient(self, y, y_pred):
        temp = grad(self.loss)
        return temp(y,y_pred)

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + anp.exp(-x))

    def gradient(self, x):
        # return self.__call__(x) * (1 - self.__call__(x))
        return elementwise_grad(self.__call__)(x)
class ReLU():
    def __call__(self, x):
        return anp.where(x >= 0, x, 0)

    def gradient(self,x):
        return elementwise_grad(self.__call__)(x)

class Softmax():
    def __call__(self, x):
        e_x = anp.exp(x - anp.max(x, axis=-1, keepdims=True))
        return e_x / anp.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        # p = self.__call__(x)
        # return p * (1 - p)
        return elementwise_grad(self.__call__)(x)

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)
        # return elementwise_grad(self.loss)(y,p)

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

def train_test_split(X, y, test_size=0.25, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test