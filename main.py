import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

test_size = 0.25
batch_size = 1000


def get_data(csv_file):
    return pd.read_csv(csv_file)


def decision_tree(x, y):
    classifier = tree.DecisionTreeClassifier(max_depth=15)
    return classifier.fit(x, y)


def svm(x, y):
    classifier = SVC(gamma='auto')
    return classifier.fit(x, y)


def multilayer_perceptron(x, y):
    classifier = MLPClassifier()
    return classifier.fit(x, y)


def main():
    file_name = "adult.csv"
    data = get_data(file_name)
    data = data.replace(' ?', np.nan)
    data = data.dropna(axis=0)
    data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    y_data = data.iloc[:, -1]
    x_data = data.drop(data.columns[-1], axis=1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=test_size)
    plt.figure()
    # decision tree
    # dt_classifier = decision_tree(x_train, y_train)
    # plot_tree(dt_classifier, filled=True)
    # plt.show()
    # support vector machine
    # svm_classifier = svm(x_data, y_data)
    # multilayer perceptron
    mlp_classifier = multilayer_perceptron(x_data, y_data)


if __name__ == "__main__":
    main()