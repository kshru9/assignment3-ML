import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression.LR_multiclass import LRMulitclass
from metrics import accuracy

from sklearn.model_selection import KFold, train_test_split
from sklearn.datasets import load_digits

np.random.seed(42)
data = load_digits()

# print(data)

cancer_df = pd.DataFrame(data=data['data'], columns=data['feature_names'])
features = data['feature_names']
cancer_df['target'] = data['target']
cancer_df.sample(5)

K = 4

kf3 = KFold(n_splits=K, shuffle=True)

scores = []

i = 1
for train_index, test_index in kf3.split(cancer_df):
    
    X_train = pd.DataFrame(cancer_df.iloc[train_index].loc[:, features]).reset_index(drop=True)
    X_test = pd.DataFrame(cancer_df.iloc[test_index][features]).reset_index(drop=True)
    y_train = pd.Series(cancer_df.iloc[train_index]['target']).reset_index(drop=True)
    y_test = pd.Series(cancer_df.loc[test_index]['target']).reset_index(drop=True)

    # print(X_train.columns)
    # print(X_test)
    # print("y_test:", y_test)
    LR = LRMulitclass()

    LR.fit(X_train,y_train, n_iter=300, lr=0.01)
    
    y_hat = LR.predict(X_test)
    # print("y_hat:",y_hat)
    temp_acc = accuracy(y_hat=y_hat, y=y_test)
    
    scores.append(temp_acc)
    
    print('Accuracy: ', temp_acc)

    i += 1

print("Average accuracy:", sum(scores)/len(scores))