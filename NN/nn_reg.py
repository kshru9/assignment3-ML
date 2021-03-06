import numpy as np
import math
from NN.utils import RMSE, ReLU, Softmax

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
    def __init__(self, n_hidden, n_iterations=3000, learning_rate=0.01):
        self.n_hidden = n_hidden
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.hidden_activation = ReLU()
        self.output_activation = Softmax()
        self.loss = RMSE()

    def _initialize_weights(self, X, y):
        n_samples, n_features = X.shape
        n_outputs = 1
        # Hidden layer
        limit   = 1 / math.sqrt(n_features)
        self.W  = np.random.uniform(-limit, limit, (n_features, self.n_hidden))
        self.w0 = np.zeros((1, self.n_hidden))
        # Output layer
        limit   = 1 / math.sqrt(self.n_hidden)
        self.V  = np.random.uniform(-limit, limit, (self.n_hidden, n_outputs))
        self.v0 = np.zeros((1, n_outputs))

    def fit(self, X, y):

        self._initialize_weights(X, y)

        for i in range(self.n_iterations):

            # ..............
            #  Forward Pass
            # ..............

            # HIDDEN LAYER
            hidden_input = np.dot(X,self.W) + self.w0
            hidden_output = self.hidden_activation(hidden_input)
            # OUTPUT LAYER
            output_layer_input = np.dot(hidden_output, self.V) + self.v0
            y_pred = self.output_activation(output_layer_input)

            # ...............
            #  Backward Pass
            # ...............

            gradsv = np.dot(hidden_output.T, y_pred)
            gradsv0 = np.sum(y_pred, axis=0)

            dhidden = np.dot(y_pred, self.V.T)
            dhidden[hidden_output <= 0] = 0

            gradsw = np.dot(X.T, dhidden)
            gradswo = np.sum(dhidden, axis=0)


            # # OUTPUT LAYER
            # # Grad. w.r.t input of output layer
            # grad_wrt_out_l_input = self.loss.gradient(y, y_pred) * self.output_activation.gradient(output_layer_input)
            # grad_v = np.dot(hidden_output.T, grad_wrt_out_l_input)
            # grad_v0 = np.sum(grad_wrt_out_l_input, axis=0, keepdims=True)
            # # HIDDEN LAYER
            # # Grad. w.r.t input of hidden layer
            # grad_wrt_hidden_l_input = self.hidden_activation.gradient(hidden_output).dot(grad_v)
            # temp = np.dot(X.T, grad_wrt_hidden_l_input)
            # grad_w = 1./X.shape[1] * np.dot(grad_wrt_hidden_l_input, X)
            # # grad_w = X.T.dot(grad_wrt_hidden_l_input)
            # grad_w0 = 1./X.shape[1] * np.sum(grad_wrt_hidden_l_input, axis=0, keepdims=True)

            # dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        
            # dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
            # dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
            # dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
            # dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                                
            # dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
            # dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
            # dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
            # dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))

            # Update weights (by gradient descent)
            # Move against the gradient to minimize loss
            self.V  -= self.learning_rate * gradsv
            self.v0 -= self.learning_rate * gradsv0
            self.W  -= self.learning_rate * gradsw
            self.w0 -= self.learning_rate * gradswo

    # Use the trained model to predict labels of X
    def predict(self, X):

        z1 = X.dot(self.W) + self.w0
        a1 = np.maximum(0, z1) # pass through ReLU activation function
        scores = a1.dot(self.V) + self.v0
        y_pred = np.argmax(scores, axis=1)
        # # Forward pass:
        # hidden_input = X.dot(self.W) + self.w0
        # hidden_output = self.hidden_activation(hidden_input)

        # output_layer_input = hidden_output.dot(self.V) + self.v0
        # y_pred = self.output_activation(output_layer_input)
        # print("V:", self.V)
        return y_pred