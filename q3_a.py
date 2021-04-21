import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.LR_multiclass import LRMulitclass
from metrics import accuracy, recall, precision

np.random.seed(42)

N = 30
P = 3
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(0,P,size=N))


LR = LRMulitclass()

LR.fit(X,y,lr=0.0001,n_iter=100000)

y_hat = LR.predict(X)

print('Accuracy: ', accuracy(y_hat=y_hat, y=y))