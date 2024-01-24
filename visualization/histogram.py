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
    lng = len(li)
    if (lng == 0):
        return None
    mn = mean(li)
    ret = 0
    for x in li:
        ret = ret + (x - mn)**2
    return (float(math.sqrt(ret / (lng - 1))))


def main():
    # 1. Take data and clean
    data = pd.read_csv('../datasets/dataset_train.csv')
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    numeric_columns = data.select_dtypes(include='number')
    histoinf = pd.DataFrame(
        index=["std"],
        columns=numeric_columns.columns,
        dtype=float
    )
    num_features = len(numeric_columns.columns)
    num_rows = (num_features + 3) // 4  # Calculer le nombre de lignes nécessaire


    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(10, 2 * num_rows))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Initialize an empty DataFrame for the histogram
    histogram_data = pd.DataFrame(columns=["Feature", "Std"])

    # Define colors for each house
    colors = {'Ravenclaw': 'blue', 'Slytherin': 'green', 'Gryffindor': 'red', 'Hufflepuff': 'yellow'}

    for i, column_name in enumerate(numeric_columns.columns):
        std_v = std(numeric_columns[column_name].dropna().to_numpy())
        histoinf.loc["std", column_name] = std_v
        
        # Append a row to the histogram DataFrame without including the index
        histogram_data = pd.concat([histogram_data, pd.DataFrame({"Feature": [column_name], "Std": [math.log(std_v)]})], ignore_index=True)

        # Check if the 'Hogwarts House' column exists before using it
        if 'Hogwarts House' in data.columns:
            # Plot the histogram on the i-th subplot with different colors for each house
            for house in houses:
                sn.histplot(data=data[data['Hogwarts House'] == house][column_name].dropna(),
                            ax=axes[i//4, i%4], color=colors[house], label=house)

            axes[i//4, i%4].set_title(f'Histogram for {column_name}')
            
            # Remove y-axis labels for a cleaner look
            axes[i//4, i%4].set_ylabel('')
            
            # Add legend to the subplot
            axes[i//4, i%4].legend()

    # Display the overall title
    plt.suptitle("Histograms for Numerical Features")
    plt.show()


if __name__ == "__main__":
    main()