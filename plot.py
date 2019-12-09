#!/usr/bin/env python3
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy(title, valid, test):
    runs = [i for i in range(1, 11)]
    plt.title(title)
    plt.plot(runs, valid, label='valid')
    plt.plot(runs, test, label='test')
    plt.legend(loc=3, bbox_to_anchor=(1,0))
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


'''
Code for this block and the next adapted from sklearn documentation:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
'''
def plot_confusion_matrix(cm, name):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           title=f'{name} Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    print(fig.show())
    return


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

    #for n, v, t in zip(names, valid, test):
    #    plot_accuracy(n, v, t)

    #all_accuracy('Valid', names, valid)
    #all_accuracy('Test', names, test)

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
    plot_confusion_matrix(cm[0], names[0])


if __name__ == "__main__":
    main()
