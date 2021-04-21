import numpy as np
import pandas as pd
from logisticRegression.logisticRegression import LogisticRegression
from metrics import accuracy


np.random.seed(42)

N = 30
P = 2
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(0,2,size=N))

scores = []
lambdas = []

for i in range(1,9):
    lambdas.append(i*0.25)
    # print(X_train)
    LR = LogisticRegression()

    LR.fit(X,y,tol=0.00001, n_iter=100, lr=0.01, fit_intercept=True, reg='L2', lbda=(i*0.25))
    
    y_hat = LR.predict(X)

    temp_acc = accuracy(y_hat=y_hat, y=y)
    
    scores.append(temp_acc)
    
    print("______")
    print("lambda:",i*0.25)
    print('Accuracy: ', temp_acc)
    print("______")

print("Average accuracy:", sum(scores)/len(scores))