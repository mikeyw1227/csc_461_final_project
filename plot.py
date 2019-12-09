#!/usr/bin/env python3
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

    all_accuracy('Valid', names, valid)
    all_accuracy('Test', names, test)

if __name__ == "__main__":
    main()
