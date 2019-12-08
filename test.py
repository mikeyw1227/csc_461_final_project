import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def get_data(csv_file):
    data = pd.read_csv(csv_file)
    data = data.replace(' ?', np.nan)
    data = data.dropna(axis=0)
    data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(
                                                   lambda x: x.astype('category'))
    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    scaler = preprocessing.MinMaxScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    return pd.DataFrame(data)


def results(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return

def best_dt(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Decision Tree')
    dt = DecisionTreeClassifier(criterion='gini', max_depth=5, splitter='best')
    dt = dt.fit(x_train, y_train)
    print('Validation Results')
    results(dt, x_valid, y_valid)
    print()
    print('Test Results')
    results(dt, x_test, y_test)
    print()
    print()
    return


def best_svm(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Kernel Support Vector Machine')
    svm = SVC(C=0.7, degree=4, gamma='scale', kernel='poly')
    svm = svm.fit(x_train, y_train)
    print('Validation Results')
    results(svm, x_valid, y_valid)
    print()
    print('Test Results')
    results(svm, x_test, y_test)
    print()
    print()
    return


def best_lg(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Logistic Regression')
    lg = LogisticRegression(max_iter=100, penalty='none', solver='lbfgs')
    lg = lg.fit(x_train, y_train)
    print('Validation Results')
    results(lg, x_valid, y_valid)
    print()
    print('Test Results')
    results(lg, x_test, y_test)
    print()
    print()
    return


def best_mlp(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Multi-Layer Perceptron')
    layers = [9, 9, 9, 9, 9, 9]
    mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=layers, 
                        learning_rate='invscaling', solver='adam')
    mlp = mlp.fit(x_train, y_train)
    print('Validation Results')
    results(mlp, x_valid, y_valid)
    print()
    print('Test Results')
    results(mlp, x_test, y_test)
    print()
    print()
    return


def best_sgd_svm(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Stochastic Gradient Descent Support Vector Machine')
    sgd_svm = SGDClassifier(loss='hinge', max_iter=100000, penalty='l2')
    sgd_svm = sgd_svm.fit(x_train, y_train)
    print('Validation Results')
    results(sgd_svm, x_valid, y_valid)
    print()
    print('Test Results')
    results(sgd_svm, x_test, y_test)
    print()
    print()
    return


def best_sgd_lg(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Stochastic Gradient Descent Logistic Regression')
    sgd_lg = SGDClassifier(loss='log', max_iter=8000, penalty='l1', 
                           learning_rate='optimal', eta0=0.001)
    sgd_lg = sgd_lg.fit(x_train, y_train)
    print('Validation Results')
    results(sgd_lg, x_valid, y_valid)
    print()
    print('Test Results')
    results(sgd_lg, x_test, y_test)
    print()
    print()
    return


def best_percep(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Perceptron')
    p = Perceptron(max_iter=1000, penalty='l2') 
    p = p.fit(x_train, y_train)
    print('Validation Results')
    results(p, x_valid, y_valid)
    print()
    print('Test Results')
    results(p, x_test, y_test)
    print()
    print()
    return


def best_knn(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('K Nearest Neighbors')
    knn = KNeighborsClassifier(algorithm='auto', n_neighbors=13, p=1, 
                               weights='uniform')
    knn = knn.fit(x_train, y_train)
    print('Validation Results')
    results(knn, x_valid, y_valid)
    print()
    print('Test Results')
    results(knn, x_test, y_test)
    print()
    print()
    return


def best_rf(x_train, y_train, x_valid, y_valid, x_test, y_test):
    print('Random Forest')
    rf = RandomForestClassifier(criterion='entropy', max_depth=15, 
                                n_estimators=100)
    rf = rf.fit(x_train, y_train)
    print('Validation Results')
    results(rf, x_valid, y_valid)
    print()
    print('Test Results')
    results(rf, x_test, y_test)
    print()
    print()
    return


def main():
    # get and clean data
    data = get_data("adult-train.csv")
    y_data = data.iloc[:, -1]
    x_data = data.drop(data.columns[-1], axis=1)
    val_size = 0.25
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, 
                                                          test_size=val_size)
    
    # get the test data
    test = get_data("adult-test.csv")
    y_test = test.iloc[:, -1]
    x_test = test.drop(test.columns[-1], axis=1)
   
    # best decision tree
    #best_dt(x_train, y_train, x_valid, y_valid, x_test, y_test)
    
    # best kernel svm
    #best_svm(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # best logistic regression
    #best_lg(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # best mlp
    #best_mlp(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # best sgd svm
    #best_sgd_svm(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # best sgd lg
    #best_sgd_lg(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # best perceptron
    #best_percep(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # best knn
    #best_knn(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # best random forest
    best_rf(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # best gradient boosted decision tree




if __name__ == "__main__":
    main()
