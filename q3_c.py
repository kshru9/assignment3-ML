import numpy as np
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
from logisticRegression.LR_multiclass import LRMulitclass
from metrics import accuracy
import math
from sklearn.model_selection import KFold
from sklearn.datasets import load_digits


def visualise_confusion(y_hat,y,digit):
    mat = [[0,0],[0,0]]
    samples = len(y)
    for i in range(samples):
        if (digit == y[i]):
            if (y[i] == y_hat[i]):
                mat[0][0] += 1
            else:
                mat[1][0] += 1
        else:
            if (y[i] == y_hat[i]):
                mat[1][1] += 1
            else:
                mat[0][1] += 1

    for i in range(0,2):
        for j in range(0,2):
            mat[i][j] = mat[i][j] / samples

    return mat

def matthew_coeff(mat):
    num = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]
    deno = (mat[0][0] + mat[0][1]) * (mat[0][0] + mat[1][0]) * (mat[1][1] + mat[0][1]) * (mat[1][0] + mat[1][1])
    deno = math.sqrt(deno)
    return num/deno

def for_each_digit(y_hat,y):
    all_coeff = dict()
    for i in range(0,10):
        all_coeff[i] = 0

    min_c = 0
    least_conf = 0
    for i in range(0,10):
        temp = visualise_confusion(y_hat,y,i)
        coeff = matthew_coeff(temp)
        all_coeff[i] = coeff
        if (min_c > coeff):
            min_c = coeff
            least_conf = i
    
    max_c_1 = 0
    max_conf_1 = 0
    for i in range(0,10):
        if (max_conf_1 < all_coeff[i]):
            max_conf_1 = all_coeff[i]
            max_c_1 = i

    all_coeff[max_c_1] = -1
    max_c_2 = 0
    max_conf_2 = 0
    for i in range(0,10):
        if (max_conf_2 < all_coeff[i]):
            max_conf_2 = all_coeff[i]
            max_c_2 = i
    
    return (min_c, max_c_1, max_c_2)

np.random.seed(42)
data = load_digits()

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

    LR = LRMulitclass()

    LR.fit(X_train,y_train, n_iter=400, lr=0.01)
    
    y_hat = LR.predict(X_test)

    temp_acc = accuracy(y_hat=y_hat, y=y_test)
    
    scores.append(temp_acc)
    
    print('Accuracy: ', temp_acc)

    i += 1

print("Average accuracy:", sum(scores)/len(scores))

print(y_test)
print(y_hat)

samples = len(y_test)

y = []
y_hat_t = []
for i in range(samples):
    y.append(y_test.iloc[i])
    y_hat_t.append(y_hat[i])

print(for_each_digit(y_hat,y_test))