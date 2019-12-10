#!/usr/bin/env python3
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

def plot_accuracy(title, valid, test):
    runs = [i for i in range(1, 11)]
    plt.title(title)
    plt.plot(runs, valid, label='valid')
    plt.plot(runs, test, label='test')
    plt.legend()
    plt.show()
    return

def all_accuracy(title, names, test):
    runs = [i for i in range(1, 11)]
    plt.title(f'All {title} Accuracy')
    for n, t in zip(names, test):
        plt.plot(runs, t, label=n)
    plt.legend()
    plt.show()
    return


def plot_confusion_matrix(array, name):
    rows = (1, 0)
    cols = (1, 0)
    df_cm = pd.DataFrame(array, rows, cols)
    plt.title(f'{name} Confusion Matrix')
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 16}, fmt='d', cmap=plt.cm.Blues)
    plt.show()
    return


def get_data(csv_file):
    data = pd.read_csv(csv_file)
    cols = [c for c in data.columns]
    data = data.replace(' ?', np.nan)
    data = data.dropna(axis=0)
    
    data[data.select_dtypes(['object']).columns] = data.select_dtypes(['object']).apply(
                                                   lambda x: x.astype('category'))
    cat_cols = data.select_dtypes(['category']).columns
    data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
    scaler = preprocessing.MinMaxScaler()
    scaler = scaler.fit(data)
    data = scaler.transform(data)
    return pd.DataFrame(data, columns=cols)


def main():
    names = ('Decision Tree', 
              'Kernel SVM',
              'Logistic Regression',
              'Multi-Layer Perceptron',
              'Stochastic Gradient Descent Support Vector Machine',
              'Stochastic Gradient Descent Logistic Regression',
              'Perceptron',
              'K Nearest Neighbor',
              'Random Forest',
              'Gradient Boosting Decision Tree')

    # from 0-8.txt and test_results.txt
    valid = ((0.84, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.84, 0.84, 0.85),
             (0.84, 0.85, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84),
             (0.82, 0.82, 0.83, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.81),
             (0.84, 0.85, 0.84, 0.85, 0.84, 0.84, 0.85, 0.84, 0.84, 0.84),
             (0.81, 0.82, 0.82, 0.81, 0.81, 0.82, 0.82, 0.83, 0.82, 0.82),
             (0.82, 0.82, 0.83, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82),
             (0.76, 0.57, 0.52, 0.78, 0.81, 0.81, 0.73, 0.49, 0.78, 0.80),
             (0.83, 0.83, 0.83, 0.84, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83),
             (0.86, 0.87, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86),
             (0.86, 0.87, 0.85, 0.86, 0.87, 0.86, 0.86, 0.86, 0.86, 0.86))

    test = ((0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84),
            (0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84),
            (0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82, 0.82),
            (0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84, 0.84),
            (0.81, 0.82, 0.81, 0.81, 0.81, 0.82, 0.82, 0.82, 0.82, 0.82),
            (0.82, 0.82, 0.81, 0.82, 0.82, 0.82, 0.81, 0.82, 0.82, 0.82),
            (0.77, 0.57, 0.52, 0.78, 0.80, 0.81, 0.73, 0.50, 0.78, 0.81),
            (0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83, 0.83),
            (0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85),
            (0.85, 0.85, 0.85, 0.85, 0.86, 0.85, 0.85, 0.85, 0.85, 0.85))

    for n, v, t in zip(names, valid, test):
        plot_accuracy(n, v, t)

    all_accuracy('Valid', names, valid)
    all_accuracy('Test', names, test)

    # test_results.txt, test matrices
    cm = (np.array([[10780, 580], [1858, 1842]]),  
          np.array([[10634, 726], [1618, 2082]]),
          np.array([[10630, 730], [1981, 1719]]),
          np.array([[10515, 845], [1595, 2105]]),
          np.array([[10546, 814], [1911, 1789]]),
          np.array([[10363, 997], [1760, 1940]]),
          np.array([[10287, 1073], [1848, 1852]]),
          np.array([[10359, 1001], [1598, 2102]]),
          np.array([[10635, 725], [1536, 2164]]),
          np.array([[10689, 671], [1524, 2176]]))
    
    for c, n in zip(cm, names):
        plot_confusion_matrix(c, n)

    data = get_data('adult-train.csv')
    print(data.head(20))
    for c in data:
        sn.distplot(data[c], kde=False)
        plt.show()

if __name__ == "__main__":
    main()
