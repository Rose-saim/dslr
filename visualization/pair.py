import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def main():
    # 1. Take data and clean
    data = pd.read_csv('../datasets/dataset_train.csv')
    data = data.drop('Index', axis=1)
    colors = {'Ravenclaw': 'blue', 'Slytherin': 'green', 'Gryffindor': 'red', 'Hufflepuff': 'yellow'}

    sn.set(font_scale=0.5)
    pair_plot = sn.pairplot(
        data=data,
        hue="Hogwarts House",
        dropna=True,
        kind="scatter",
        diag_kind="hist",
        markers=".",
        height=0.5,  # inch
        aspect=1,  # width = height * aspect
        palette=colors,
        plot_kws={
            "alpha": 0.5,
        },
        diag_kws={
            "alpha": 0.5,
        }
    )
    pair_plot.fig.suptitle(
        f"Pair plot representing feature correlations." +
        " (Scatter plot matrix with histograms)",
        y=0.9975
    )
    plt.show()


if __name__ == "__main__":
    main()