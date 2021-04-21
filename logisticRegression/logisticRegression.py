import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self) -> None:
        self.num_of_iterations = None
        self.thetas = None
        self.tolerance = None

        self.thetas_history = []
        self.cost_func_history = []

    def fit(self, X, y, tol=0.0001, n_iter=100, lr=0.01, fit_intercept=True, reg=None):
        '''
        Function to train model using vectorised gradient descent.
        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number
        :return None
        '''

        # handling fit intercept param
        if (fit_intercept == True):
            self.num_of_thetas = len(list(X.columns))+1
            thetas = pd.Series(np.random.randn(self.num_of_thetas))
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        else:
            self.num_of_thetas = len(list(X.columns))
            thetas = pd.Series(np.random.randn(self.num_of_thetas))
        
        self.num_of_samples = len(X)
        self.num_of_iterations = n_iter
        self.learning_rate = lr
        self.tolerance = tol
        self.fit_intercept = fit_intercept

        for it in range(self.num_of_iterations):
            h = self.hypothesis(X,thetas)
            for attr in range(self.num_of_thetas):
                thetas[attr] -= (lr/self.num_of_samples) * np.sum((h-y)*X.iloc[:, attr])
            self.thetas_history.append(thetas)
            self.cost_func_history.append(self.cost_function(X,y,thetas))

        return thetas

    def fit_autograd(self, X, y, tol, n_iter=100, lr=0.01, fit_intercept=True):

        from autograd import grad
        from autograd import elementwise_grad as egrad
        import autograd.numpy as npa
        from math import e, log

        # def sigmoid(x):
        #     return 0.5 * (np.tanh(x / 2.) + 1)

        # def logistic_predictions(weights, inputs):
        #     # Outputs probability of a label being true according to logistic model.
        #     return sigmoid(np.dot(inputs, weights))

        def training_loss(weights):
            # Training loss is the negative log-likelihood of the training labels.
            
            preds = (1/(1 + e**( np.dot(X_il, weights) )) - 0.00001)
            label_probabilities = preds * y_il + (1 - preds) * (1 - y_il)
            return -np.sum(log(label_probabilities))

        # # Define a function that returns gradients of training loss using Autograd.
        # training_gradient_fun = grad(training_loss)

        # # Optimize weights using gradient descent.
        # weights = np.array([0.0, 0.0, 0.0])
        # print("Initial loss:", training_loss(weights))
        # for i in range(100):
        #     weights -= training_gradient_fun(weights) * 0.01

        # print("Trained loss:", training_loss(weights))

        def cost_function(X, y, thetas,num_of_samples,tol):
            """Calculating cost function to update the thetas values in Gradient descent"""

            z = npa.dot(X,thetas)
            sigmoid = 1 / (1 + npa.exp( -z ) ) - tol
            return - (npa.sum(npa.log(y*sigmoid + (1-y)*(1-sigmoid))))
            # return - ((1/num_of_samples) * npa.sum(y * npa.log(sigmoid) + (1-y)* npa.log(1-sigmoid)) )
        
        # handling fit intercept param
        if (fit_intercept == True):
            self.num_of_thetas = len(list(X.columns))+1
            thetas = pd.Series(np.random.randn(self.num_of_thetas))
            bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))
            X = pd.concat([bias,X],axis=1)
        else:
            self.num_of_thetas = len(list(X.columns))
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))
        
        self.num_of_samples = len(X)
        self.num_of_iterations = n_iter
        self.learning_rate = lr
        self.tolerance = tol
        self.fit_intercept = fit_intercept

        mygrad = grad(training_loss)

        X_il,y_il = X.to_numpy(),y.to_numpy()
        theta_il = thetas.to_numpy()

        for it in range(self.num_of_iterations):

            temp_grad = mygrad(theta_il)
            print(temp_grad)
            print("thetas:",thetas)
            thetas -= (lr/self.num_of_samples) * temp_grad
            self.thetas_history.append(thetas)
            self.cost_func_history.append(temp_grad)

        return thetas

    def fit_l1_regularised(self):
        pass

    def fit_l2_regularised(self):
        pass
        
    def hypothesis(self, X, theta):
        """Calculating step function for logistic regression"""

        z = np.dot(theta, X.T)
        sigmoid = 1 / (1 + np.exp( -z ) ) - self.tolerance
        return sigmoid

    def cost_function(self, X, y, thetas):
        """Calculating cost function to update the thetas values in Gradient descent"""

        hyp = self.hypothesis(X, thetas)
        cost = - ((1/self.num_of_samples) * np.sum(y * np.log(hyp + 1e-5) + (1-y)* np.log(1-hyp+1e-5)) )
        # print("printing cost:", cost)
        return cost

    def predict(self, X):
        """Prediction function"""

        bias = pd.DataFrame(pd.Series([1.0 for i in range(len(X))]))

        if self.fit_intercept:
            X = pd.concat([bias,X],axis=1)

        theta = pd.Series(self.thetas_history[-1])
        hyp = self.hypothesis(X, theta)

        for i in range(len(hyp)):
            hyp[i] = 1 if hyp[i] >=0.5 else 0
        
        return hyp

    def plot():
        pass
