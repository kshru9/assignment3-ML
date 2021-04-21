import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.LR_multiclass import LRMulitclass
from metrics import accuracy, recall, precision

np.random.seed(42)

N = 30
P = 3
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(0,3,size=N))


LR = LRMulitclass()

LR.fit(X,y,lr=0.001,n_iter=1000)

y_hat = LR.predict(X)
print("y:",y)
print("y_hat:",y_hat)

print('Accuracy: ', accuracy(y_hat=y_hat, y=y))