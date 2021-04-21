import numpy as np
from sklearn import datasets

from NN.nn import MultilayerPerceptron, train_test_split, normalize

def main():
    data = datasets.load_boston()
    X = normalize(data.data)
    y = data.target

    # Convert the nominal y values to binary
    # y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, seed=1)

    # MLP
    clf = MultilayerPerceptron(n_hidden=16,
        n_iterations=1000,
        learning_rate=0.01,
        classifier=False)

    clf.fit(X_train, y_train)
    # y_pred = np.argmax(clf.predict(X_test), axis=1)
    # y_test = np.argmax(y_test, axis=1)

    # accuracy = accuracy_score(y_test, y_pred)
    # print ("Accuracy:", accuracy)

    # # Reduce dimension to two using PCA and plot the results
    # Plot().plot_in_2d(X_test, y_pred, title="Multilayer Perceptron", accuracy=accuracy, legend_labels=np.unique(y))

if __name__ == "__main__":
    main()