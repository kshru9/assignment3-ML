import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
from metrics import accuracy, recall, precision

np.random.seed(42)

N = 30
P = 3
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(0,P,size=N))


for fit_intercept in [True, False]:
    LR = LogisticRegression()

    LR.fit_multiclass(X,y,tol=0.0001,lr=0.01,n_iter=100,fit_intercept=fit_intercept)

    y_hat = LR.predict(X)

    print('Accuracy: ', accuracy(y_hat=y_hat, y=y))