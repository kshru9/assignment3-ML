import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.logisticRegression import LogisticRegression
import sys
from metrics import accuracy, recall, precision

from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import load_breast_cancer

np.random.seed(42)
data = load_breast_cancer()

cancer_df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
features = data['feature_names']
cancer_df['target'] = data['target']
cancer_df.sample(5)

K = 3

kf3 = KFold(n_splits=K, shuffle=True)

scores = []

i = 1
for train_index, test_index in kf3.split(cancer_df):
    
    X_train = pd.DataFrame(cancer_df.iloc[train_index].loc[:, features])
    X_test = pd.DataFrame(cancer_df.iloc[test_index][features])
    y_train = pd.Series(cancer_df.iloc[train_index].loc[:,'target'])
    y_test = pd.Series(cancer_df.loc[test_index]['target'])

    LR = LogisticRegression()

    LR.fit_unregularised(X_train,y_train,tol=0.000001, n_iter=400, lr=0.01, fit_intercept=False)
    
    y_hat = LR.predict(X_test)

    temp_acc = accuracy(y_hat=y_hat, y=y_test)
    
    scores.append(temp_acc)
    
    print('Accuracy: ', temp_acc)

    i += 1

print("Average accuracy:", sum(scores)/len(scores))