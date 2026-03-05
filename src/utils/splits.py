import matplotlib.pyplot as plt
from pandas import DataFrame


def check_split(df: DataFrame, col: str):
    df[col].value_counts().plot(kind="bar")
    for i in range(2):
            plt.text(
                i,
                df[col].value_counts().iloc[i],
                f"{df[col].value_counts().iloc[i]} ({df[col].value_counts().iloc[i] / len(df) * 100:.2f}%)",
                ha="center",
                va="bottom",
            )
    plt.show()
