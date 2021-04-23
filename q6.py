from NN.nn_reg import MultilayerPerceptron
from sklearn.datasets import load_boston
from metrics import rmse
from NN.utils import normalize, train_test_split

def main():
    data = load_boston()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, seed=1)

    reg = MultilayerPerceptron(n_hidden=32, n_iterations=10, learning_rate=0.001)

    reg.fit(X_train,y_train)

    y_hat = reg.predict(X_test)

    print("rmse:", rmse(y_hat, y_test))

if __name__ == '__main__':
    main()