import numpy as np
import math


class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

class ReLU():
    def __call__(self, x):
        return max(0,x)

class Softmax():
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)

class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        # print(p)
        return - ((y / p) + ((1 - y) / (1 - p)))

def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)

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

class MultilayerPerceptron():
    """Multilayer Perceptron classifier. A fully-connected neural network with one hidden layer.
    Unrolled to display the whole forward and backward pass.
    Parameters:
    -----------
    n_hidden: int:
        The number of processing nodes (neurons) in the hidden layer. 
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_hidden, n_iterations=100, learning_rate=0.01, classifier=False):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = Sigmoid()
        self.output_activation = Softmax()
        self.loss = CrossEntropy()
        self.classifier =classifier

    def _initialize_weights(self, X, y):
        n_samples, n_features = X.shape
        # _, n_outputs = y.shape
        
        # Hidden layer 1
        limit   = 1 / math.sqrt(n_features)
        self.W  = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))

        # Hidden layer 2
        limit   = 1 / math.sqrt(n_features)
        hl_2 = int(math.sqrt(self.n_hidden))
        self.W1  = np.random.uniform(-limit, limit, (self.n_hidden, hl_2))
        self.w1 = np.zeros((1, hl_2))

        # Output layer
        limit   = 1 / math.sqrt(self.n_hidden)
        self.V  = np.random.uniform(-limit, limit, (hl_2, 1))
        self.v0 = np.zeros((1, 1))
        

    def fit(self, X, y):

        self._initialize_weights(X, y)

        for i in range(self.n_iterations):

            # ..............
            #  Forward Pass
            # ..............

            # HIDDEN LAYER 1
            hidden_input = X.dot(self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)

            # HIDDEN LAYER 2
            hidden_input_2 = hidden_output.dot(self.W1) + self.w1
            hidden_output_2 = self.hidden_activation(hidden_input_2)

            # OUTPUT LAYER
            output_layer_input = hidden_output_2.dot(self.V) + self.v0
            y_pred = self.output_activation(output_layer_input)
            print(len(output_layer_input))
            # ...............
            #  Backward Pass
            # ...............

            # OUTPUT LAYER
            # Grad. w.r.t input of output layer
            grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            print(len(self.loss.gradient(y, y_pred)))
            grad_v = hidden_output_2.T.dot(grad_wrt_out_l_input)
            grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
            # HIDDEN LAYER
            # Grad. w.r.t input of hidden layer
            # print("grad_wrt_out_l_input",grad_wrt_out_l_input)
            # print(self.V)
            print(grad_wrt_out_l_input)
            grad_wrt_hidden_l_input_2 = grad_wrt_out_l_input.dot(self.V.T) * self.hidden_activation.gradient(hidden_input_2)
            grad_w1 = hidden_output.T.dot(grad_wrt_hidden_l_input_2)
            grad_w1_0 = np.sum(grad_wrt_hidden_l_input_2, axis=0, keepdims=True)

            grad_wrt_hidden_l_input = grad_wrt_hidden_l_input_2.dot(self.W1.T) * self.hidden_activation.gradient(hidden_input)
            grad_w = X.T.dot(grad_wrt_hidden_l_input)
            grad_w0 = np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            # Update weights (by gradient descent)
            # Move against the gradient to minimize loss
            # print(self.V, grad_v)
            self.V  -= self.learning_rate * grad_v
            self.v0 -= self.learning_rate * grad_v0
            self.W1 -= self.learning_rate * grad_w1
            self.w1 -= self.learning_rate * grad_w1_0
            self.W  -= self.learning_rate * grad_w
            self.w0 -= self.learning_rate * grad_w0

    # Use the trained model to predict labels of X
    def predict(self, X):
        # Forward pass:
        hidden_input = X.dot(self.W) + self.w0
        hidden_output = self.hidden_activation(hidden_input)

        hidden_input_2 = hidden_output.dot(self.W1) + self.w1
        hidden_output_2 = self.hidden_activation(hidden_input_2)

        output_layer_input = hidden_output_2.dot(self.V) + self.v0
        y_pred = self.output_activation(output_layer_input)
        return y_pred