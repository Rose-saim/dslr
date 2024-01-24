import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
import seaborn as sn

def parse_arguments():
    """
    Parse the command line argument.
    Positional argument:
    - The program takes one positional argument, the path of the dataset.
    Optional arguments:
    - Bonus : [--bonus | -b] display more metrics.
    - Compare : [--compare | -c] compare with real describe().
    - Help : [--help | -h] display an help message.
    Usage:
      python describe.py [-b | --bonus] [-c | --compare] [-h | --help] data.csv
    """
    try:
        parser = ArgumentParser(
            prog="describe",
            description="This program takes a dataset path as argument. " +
            "It displays informations for all numerical features."
        )
        parser.add_argument(
            dest="dataset_path",
            type=str,
            help="Path to the dataset."
        )
        args = parser.parse_args()
        return (
            args.dataset_path
        )
    except Exception as e:
        print("Error parsing arguments: ", e)
        exit()
    
def mean(li):
    if (len(li) == 0):
        return None
    ret = 0
    for x in li:
        ret = ret + x
    return (float(ret / len(li)))


def std(li):
    return float(np.std(li, ddof=1))

def main():
    # 1. Take data and clean
    data = pd.read_csv('../datasets/dataset_train.csv')
    data = data.drop('Index', axis=1)
    numeric_columns = data.select_dtypes(include='number')
    scatter_info = pd.DataFrame(
        index=["std"],
        columns=numeric_columns.columns,
        dtype=float
    )

    corr_ = numeric_columns.corr()

    max_corr = 0
    max_pair_feature = (None, None)

    for i in range(corr_.shape[0]):
        for j in range(i):
            if abs(corr_.iloc[i, j]) > max_corr:
                max_corr = abs(corr_.iloc[i, j])
                max_pair_feature = (corr_.index[i], corr_.columns[j])
    
    print(f"Correlation matrix:\n{corr_}")
    print(f"Max correlation coefficient: {max_corr}")
    print(f"Max pair of features: {max_pair_feature}")

    # Plot the scatter plot
    plt.scatter(numeric_columns[max_pair_feature[0]], numeric_columns[max_pair_feature[1]])
    plt.xlabel(max_pair_feature[0])
    plt.ylabel(max_pair_feature[1])
    plt.title(f"Scatter plot of {max_pair_feature[0]} and {max_pair_feature[1]}")
    plt.show()


if __name__ == "__main__":
    main()