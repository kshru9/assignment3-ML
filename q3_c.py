import numpy as np
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
from logisticRegression.LR_multiclass import LRMulitclass
from metrics import accuracy

from sklearn.model_selection import KFold
from sklearn.datasets import load_digits

from q3_conf import confusion_matrix, find_digits

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

samples = len(y_test)

y = []
y_hat_t = []
for i in range(samples):
    y.append(y_test.iloc[i])
    y_hat_t.append(y_hat[i])

mat = confusion_matrix(y_hat_t, y)
print(find_digits(mat))