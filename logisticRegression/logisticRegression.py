import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self) -> None:
        self.num_of_iterations = None
        self.thetas = None
        self.tolerance = None

        self.thetas_history = []
        self.cost_func_history = []

    def fit_unregularised(self, X, y, tol, n_iter=100, lr=0.01, fit_intercept=True):
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
            self.thetas = pd.Series(np.random.randn(self.num_of_thetas))
        
        self.num_of_samples = len(X)
        self.num_of_iterations = n_iter
        self.learning_rate = lr
        self.tolerance = tol
        self.fit_intercept = fit_intercept

        # init hypothesis and cost functions
        self.thetas_history.append(thetas)
        self.cost_func_history.append(self.cost_function(X,y,thetas))

        for it in self.num_of_iterations:
            h = self.hypothesis(X,thetas)
            for attr in range(self.num_of_thetas):
                thetas[attr] -= (lr/self.num_of_samples) * np.sum((h-y)*X.iloc[:, attr])
            self.cost_func_history.append(self.cost_function(X,y,thetas))

        return self.thetas

    def fit_l1_regularised(self):
        pass

    def fit_l2_regularised(self):
        pass

    def hypothesis(self, X, theta):
        """
        Calculating step function for logistic regression
        """
        z = np.dot(X.T, theta)
        sigmoid = 1 / (1 + np.exp( -z ) ) - self.tolerance
        return sigmoid

    def cost_function(self, X, y, thetas):
        """Calculating cost function to update the thetas values in Gradient descent"""
        hyp = self.hypothesis(X, thetas)
        cost = - ((1/self.num_of_samples) * np.sum(y * np.log(hyp) + (1-y)* np.log(1-hyp)) )
        return cost

    def predict(self, X):
        """Prediction function"""
        # h = hypothesis(X, theta)
        # for i in range(len(h)):
        #     h[i]=1 if h[i]>=0.5 else 0
        # y = list(y)
        # acc = np.sum([y[i] == h[i] for i in range(len(y))])/len(y)
        # return J, acc

        theta = pd.Series(self.thetas_history[-1])
        hyp = self.hypothesis(X, theta)

        for i in range(len(hyp)):
            hyp[i] = 1 if hyp[i] >=0.5 else 0
        
        return hyp

    def plot():
        pass
