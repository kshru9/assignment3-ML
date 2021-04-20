import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression

class LRMulitclass():

    def __init__(self) -> None:
        self.models = []


    def fit(self, X, y, lr=0.01, n_iter=100):
        self.num_of_thetas = len(list(X.columns))
        thetas = pd.Series(np.random.randn(self.num_of_thetas))
        
        self.num_of_samples = len(X)
        self.num_of_iterations = n_iter
        self.learning_rate = lr

        # init hypothesis and cost functions
        self.thetas_history.append(thetas)
        self.cost_func_history.append(self.cost_function(X,y,thetas))

        y_onehot = self.onehotencoder(y)

        for model in range(self.num_of_out_classes):
            y_temp = y_onehot.iloc[:,model]
            LR = LogisticRegression()
            thetas = LR.fit_unregularised(X,y,n_iter=n_iter, lr=lr, fit_intercept=False)
            self.models.append(LR)


    def onehotencoder(self,y):
        onehot = []
        self.num_of_out_classes = len(y.unique())
        for i in range(self.num_of_samples):
            pos = y[i]
            temp = [-1]*self.num_of_out_classes
            temp[pos] = 1
            onehot.append(temp)
        onehot = pd.DataFrame(onehot)
        return onehot

    def predict(self,):
        pass