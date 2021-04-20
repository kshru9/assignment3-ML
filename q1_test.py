import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
import sys
from metrics import accuracy, recall, precision

np.random.seed(42)

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(0,2,size=N))


for fit_intercept in [True, False]:
    LR = LogisticRegression()

    LR.fit_unregularised(X,y,tol=0.0001,lr=0.001,n_iter=10000,fit_intercept=fit_intercept)

    y_hat = LR.predict(X)

    print('Accuracy: ', accuracy(y_hat=y_hat, y=y))

    # print(y_hat,y)


# for fit_intercept in [True, False]:
#     LR = LogisticRegression()

#     LR.fit_unregularised_autograd(X,y,tol=0.000001, n_iter=100, lr=0.01, fit_intercept=True)

#     y_hat = LR.predict(X)

#     print('Accuracy: ', accuracy(y_hat=y_hat, y=y))