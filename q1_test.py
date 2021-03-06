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

LR = LogisticRegression()

LR.fit(X,y,tol=0.0001,lr=0.001,n_iter=500,fit_intercept=True)

y_hat = LR.predict(X)

print('Accuracy: ', accuracy(y_hat=y_hat, y=y))