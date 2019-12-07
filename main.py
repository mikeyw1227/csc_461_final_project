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
    data = pd.read_csv(csv_file)
    data = data.replace(' ?', np.nan)
    data = data.dropna(axis=0)
    data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    return data


def results(model, x_test, y_test):
    print(model.best_params_)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def decision_tree(x, y, params):
    classifier = tree.DecisionTreeClassifier()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def svm(x, y, params):
    classifier = SVC()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def multilayer_perceptron(x, y):
    classifier = MLPClassifier()
    return classifier.fit(x, y)


def main():
    # get and clean data
    data = get_data("adult.csv")
    y_data = data.iloc[:, -1]
    x_data = data.drop(data.columns[-1], axis=1)
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=test_size)
    # decision tree
    # dt_params = {'criterion': ['gini', 'entropy'],
    #              'splitter': ['best', 'random'],
    #              'max_depth': [i for i in range(5, 105, 10)]}
    # dt_classifier = decision_tree(x_train, y_train, dt_params)
    # results(dt_classifier, x_valid, y_valid)
    # support vector machine
    svm_params = {'C': [i for i in range(1, 5, 1)],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                  'degree': [i for i in range(1, 5, 1)],
                  'gamma': ['auto', 'scale']}
    svm_classifier = svm(x_data, y_data, svm_params)
    results(svm_classifier, x_valid, y_valid)
    # multilayer perceptron
    # mlp_classifier = multilayer_perceptron(x_data, y_data)


if __name__ == "__main__":
    main()