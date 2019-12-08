import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from enum import Enum


class Model(Enum):
    ALL = 0
    DT = 1
    SVM = 2
    LG = 3
    MLP = 4
    SGD_SVM = 5
    SGD_LG = 6
    PERCEP = 7
    KNN = 8
    FOREST = 9


def get_data(csv_file):
    data = pd.read_csv(csv_file)
    data = data.replace(' ?', np.nan)
    data = data.dropna(axis=0)
    data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    scaler = preprocessing.MinMaxScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    return pd.DataFrame(data)


def results(model, x_test, y_test):
    print(model.best_params_)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return


def decision_tree(x, y, params):
    classifier = DecisionTreeClassifier()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def train_dt(x_train, x_valid, y_train, y_valid):
    # decision tree
    dt_params = {'criterion': ['gini', 'entropy'],
                  'splitter': ['best', 'random'],
                  'max_depth': [i for i in range(5, 105, 10)]}
    dt_classifier = decision_tree(x_train, y_train, dt_params)
    results(dt_classifier, x_valid, y_valid)
    return


def svm(x, y, params):
    classifier = SVC()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def train_svm(x_train, x_valid, y_train, y_valid):
    # support vector machine
    svm_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'gamma': ['auto', 'scale'],
                  'degree': [i for i in range(1, 5, 1)],
                  'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}
    svm_classifier = svm(x_train, y_train, svm_params)
    results(svm_classifier, x_valid, y_valid)
    return


def sgd(x, y, params):
    classifier = SGDClassifier();
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def train_sgd_svm(x_train, x_valid, y_train, y_valid):
    # SGD SVM
    sgdsvm_params = {
            'loss': ('hinge', 'squared_hinge'),
            'penalty': ('none', 'l1', 'l2', 'elasticnet'), 
            'max_iter': [i for i in range(1000, 10001, 1000)],
    }
    sgdsvm_classifier = sgd(x_train, y_train, sgdsvm_params)
    results(sgdsvm_classifier, x_valid, y_valid)
    return


def train_sgd_lg(x_train, x_valid, y_train, y_valid):
    # SGD Logistic Regression
    sgdlg_params = {
            'loss': ('log',),
            'penalty': ('none', 'l1', 'l2', 'elasticnet'), 
            'max_iter': [i for i in range(1000, 10001, 1000)],
            'learning_rate': ('constant', 'optimal', 'invscaling', 'adaptive'),
            'eta0': (0.001,)
    }
    sgdlg_classifier = sgd(x_train, y_train, sgdlg_params)
    results(sgdlg_classifier, x_valid, y_valid)
    return

def logistic_regression(x, y, params):
    classifier = LogisticRegression()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def train_lg(x_train, x_valid, y_train, y_valid):
    # Logistic Regression
    lg_params = {
        'penalty': ('l1', 'l2', 'none', 'elasticnet'),
        'solver': ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'),
        'max_iter': [i for i in range(100, 600, 100)]
    }
    lg_classifier = logistic_regression(x_train, y_train, lg_params)
    results(lg_classifier, x_valid, y_valid)
    return


def multilayer_perceptron(x, y, params):
    classifier = MLPClassifier()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def train_mlp(x_train, x_valid, y_train, y_valid):
    # MLP
    n = int(2 / 3 * x_train.shape[1])  # suitable because source
    mlp_params = {
        'hidden_layer_sizes': [],
        'activation': ('logistic', 'identity', 'tanh', 'relu'),
        'solver': ('lbfgs', 'sgd', 'adam'),
        'learning_rate': ('constant', 'invscaling')
    }
    for i in range(2, 11):
        mlp_params['hidden_layer_sizes'].append([n for _ in range(i)])
    mlp_classifier = multilayer_perceptron(x_train, y_train, mlp_params)
    results(mlp_classifier, x_valid, y_valid)
    return

def linear_perceptron(x, y, params):
    classifier = Perceptron()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def train_lp(x_train, x_valid, y_train, y_valid):
    lp_params = {'penalty': ['l2', 'l1', 'elasticnet'],
                 'max_iter': [1000, 5000, 10000]}
    lp_classifier = linear_perceptron(x_train, y_train, lp_params)
    results(lp_classifier, x_valid, y_valid)
    return


def random_forest(x, y, params):
    classifier = RandomForestClassifier()
    optimized_classifier = GridSearchCV(classifier, params)
    return optimized_classifier.fit(x, y)


def train_rf(x_train, x_valid, y_train, y_valid):
    # random forest
    rf_params = {'criterion': ['gini', 'entropy'],
                 'n_estimators': [i for i in range(10, 101, 10)],
                 'max_depth': [i for i in range(5, 105, 10)]}
    rf_classifier = random_forest(x_train, y_train, rf_params)
    results(rf_classifier, x_valid, y_valid)
    return


def main():
    # get and clean data
    data = get_data("adult-train.csv")
    y_data = data.iloc[:, -1]
    x_data = data.drop(data.columns[-1], axis=1)
    test_size = 0.25
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=test_size)
    e = Model.PERCEP
    if e == Model.DT:
        train_dt(x_train, x_valid, y_train, y_valid)
    elif e == Model.SVM:
        train_svm(x_train, x_valid, y_train, y_valid)
    elif e == Model.LG:
        train_lg(x_train, x_valid, y_train, y_valid)
    elif e == Model.MLP:
        train_mlp(x_train, x_valid, y_train, y_valid)
    elif e == Model.SGD_SVM:
        train_sgd_svm(x_train, x_valid, y_train, y_valid)
    elif e == Model.SGD_LG:
        train_sgd_lg(x_train, x_valid, y_train, y_valid)
    elif e == Model.PERCEP:
        train_lp(x_train, x_valid, y_train, y_valid)
    elif e == Model.KNN:
        pass
    elif e == Model.FOREST:
        train_rf(x_train, x_valid, y_train, y_valid)
    else:
        train_dt(x_train, x_valid, y_train, y_valid)
        print()
        train_svm(x_train, x_valid, y_train, y_valid)
        print()
        train_lg(x_train, x_valid, y_train, y_valid)
        print()
        train_mlp(x_train, x_valid, y_train, y_valid)
        print()
        train_sgd_svm(x_train, x_valid, y_train, y_valid)
        print()
        train_sgd_lg(x_train, x_valid, y_train, y_valid)
        print()
        train_lp(x_train, x_valid, y_train, y_valid)
        print()
        train_rf(x_train, x_valid, y_train, y_valid)
        print()
        # TODO copy all functions from above
    return


if __name__ == "__main__":
    main()
