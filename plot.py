#!/usr/bin/env python3
import matplotlib.pyplot as plt
plt.style.use('seaborn=whitegrid')

def plot_accuracy(title, valid, test):
    runs = [i for i in range(1, 23)]
    plt.title(title)
    plt.plot(runs, valid, label='valid')
    plt.plot(runs, title, label='title')
    plt.legend(loc=3, bbox_to_anchor=(1,0))
    plt.show()
    return


def main():
    titles = ('Decision Tree', 
              'Kernel SVM',
              'Logistic Regression',
              'Multi-Layer Perceptron',
              'Stochastic Gradient Descent Support Vector Machine',
              'Stochastic Gradient Descent Logistic Regression',
              'Perceptron',
              'K Nearest Neighbor',
              'Random Forest'
              'Gradient Boosting Decision Tree')

    valid = ((0.84, 0.85, 0.85, 0.85, ),
             (0.84, 0.85, 0.84, 0.84, ),
             (0.82, 0.82, 0.83, 0.82, ),
             (0.84, 0.85, 0.84, 0.85, ),
             (0.81, 0.82, 0.82, 0.81, ),
             (0.82, 0.82, 0.83, 0.82, ),
             (0.76, 0.57, 0.52, 0.78, ),
             (0.83, 0.83, 0.83, 0.84, ),
             (0.86, 0.87, 0.86, 0.86, ),
             (0.86, 0.87, 0.85, 0.86, ))

    test = ((0.84, 0.84, 0.84, 0.84, ),
            (0.84, 0.84, 0.84, 0.84, ),
            (0.82, 0.82, 0.82, 0.82, ),
            (0.84, 0.84, 0.84, 0.84, ),
            (0.81, 0.82, 0.81, 0.81, ),
            (0.82, 0.82, 0.81, 0.82, ),
            (0.77, 0.57, 0.52, 0.78, ),
            (0.83, 0.83, 0.83, 0.83, ),
            (0.85, 0.85, 0.85, 0.85, ),
            (0.85, 0.85, 0.85, 0.85, ))

if __name__ == "__main__":
    main()
