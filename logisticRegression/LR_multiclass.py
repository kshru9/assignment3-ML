import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression

class LRMulitclass():

    def __init__(self) -> None:
        self.models = []
        self.thetas_history = []

    def fit(self, X, y, lr=0.01, n_iter=100):
        self.num_of_thetas = len(list(X.columns))
        # thetas = pd.Series(np.random.randn(self.num_of_thetas))
        
        self.num_of_samples = len(X)
        self.num_of_iterations = n_iter
        self.learning_rate = lr
        # self.num_of_out_classes = len(y.columns())

        # init hypothesis and cost functions
        # self.thetas_history.append(thetas)
        # self.cost_func_history.append(self.cost_function(X,y,thetas))

        y_onehot = self.onehotencoder(y)
        print(self.num_of_out_classes)
        # print(self.num_of_thetas)
        for model in range(self.num_of_out_classes):
            y_temp = y_onehot.iloc[:,model]
            # print("y_temp:", y_temp)
            LR = LogisticRegression()
            thetas = LR.fit(X,y_temp,n_iter=n_iter, lr=lr, fit_intercept=False)
            # print("thetas:",thetas)
            self.models.append(LR)
            self.thetas_history.append(thetas)
        # print("printingn thetas hist")
        # print(self.thetas_history)

    def onehotencoder(self,y):
        onehot = []
        self.num_of_out_classes = len(y.unique())
        for i in range(self.num_of_samples):
            # print("i",i)
            pos = y[i]
            temp = [0]*self.num_of_out_classes
            temp[pos] = 1
            onehot.append(temp)
        onehot = pd.DataFrame(onehot)
        return onehot

    def hypothesis(self, X, theta):
        """Calculating step function for logistic regression"""
        z = np.dot(theta, X.T)
        sigmoid = 1 / (1 + np.exp( -z ) ) - 0.0001
        return sigmoid
    
    def softmax(self,y_linear):
        exp = np.exp(y_linear).reshape(-1,1)
        # print("exp:" , exp)
        norms = np.sum(exp).reshape(-1,1)
        return exp / norms

    def hypothesis_multiclass(self,X,theta):
        z = np.dot(theta, X.T) + 0.0001
        # print("z:" , z)
        sm = self.softmax(z)
        my_list = map(lambda x: x[0], sm)
        return pd.Series(my_list)

    def cost_function_multiclass(self,X,y,thetas):
        hyp = self.hypothesis_multiclass(X,thetas)
        return - np.sum(y * np.log(hyp))
    
    def predict(self,X):
        model_predict = []
        for i in range(len(self.models)):
            model_predict.append(self.hypothesis_multiclass(X, theta=self.thetas_history[i]))
        
        # print("model_pred:",model_predict)
        final_pred = []
        for i in range(len(X)):
            mx = 0
            mx_index = 0
            for j in range(len(self.models)):
                # print("model_predict[j][i]:", model_predict[j][i])
                if (mx < model_predict[j][i]):
                    mx = model_predict[j][i]
                    mx_index = j
                # mx = max(mx, model_predict[j][i])
                
            final_pred.append(mx_index)

        # print(final_pred)
        return pd.Series(final_pred)