import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

test_size = 0.25
batch_size = 1000


def get_data(csv_file):
    return pd.read_csv(csv_file)


def decision_tree(x, y, params):
    classifier = tree.DecisionTreeClassifier()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def svm(x, y):
    classifier = SVC(gamma='auto')
    return classifier.fit(x, y)


def multilayer_perceptron(x, y):
    classifier = MLPClassifier()
    return classifier.fit(x, y)


def main():
    file_name = "adult.csv"
    # get and clean data
    data = get_data(file_name)
    data = data.replace(' ?', np.nan)
    data = data.dropna(axis=0)
    data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    y_data = data.iloc[:, -1]
    x_data = data.drop(data.columns[-1], axis=1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=test_size)
    # decision tree
    # plt.figure()
    dt_params = {'criterion': ['gini', 'entropy'],
                 'splitter': ['best', 'random'],
                 'max_depth': [i for i in range(5, 105, 10)]}
    dt_classifier = decision_tree(x_train, y_train, dt_params)
    # print(dt_classifier.best_params_)
    y_pred = dt_classifier.predict(x_valid)
    print(classification_report(y_valid, y_pred))
    print(confusion_matrix(y_valid, y_pred))
    # plot_tree(dt_classifier, filled=True)
    # plt.show()
    # support vector machine
    # svm_classifier = svm(x_data, y_data)
    # multilayer perceptron
    # mlp_classifier = multilayer_perceptron(x_data, y_data)


if __name__ == "__main__":
    main()