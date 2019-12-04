import numpy as np
import pandas as pd


def get_data(csv_file):
    data_set = pd.read_csv(csv_file)
    return data_set.to_numpy()


def main():
    file_name = "adult.data"
    data = get_data(file_name)
    print(data)


if __name__ == "__main__":
    main()