import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree

test_size = 0.25
batch_size = 1000


def get_data(csv_file):
    data_set = pd.read_csv(csv_file)
    return data_set.to_numpy()


def decision_tree(x, y):
    classifier = tree.DecisionTreeClassifier(max_depth=10)
    return classifier.fit(x, y)


def main():
    file_name = "adult.data"
    data = get_data(file_name)
    training, validate = train_test_split(data, test_size=test_size)
    print(validate)


if __name__ == "__main__":
    main()