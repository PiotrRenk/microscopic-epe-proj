import matplotlib.pyplot as plt
from pandas import DataFrame


def check_split(df_train: DataFrame, df_test: DataFrame, col: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for idx, (df, ax) in enumerate([(df_train, axes[0]), (df_test, axes[1])]):
        df[col].value_counts().plot(kind="bar", ax=ax)
        for i in range(len(df[col].value_counts())):
            ax.text(
                i,
                df[col].value_counts().iloc[i],
                f"{df[col].value_counts().iloc[i]} ({df[col].value_counts().iloc[i] / len(df) * 100:.2f}%)",
                ha="center",
                va="bottom",
            )
    
    axes[0].set_title(f"Training Set ({len(df_train)})")
    axes[1].set_title(f"Test Set ({len(df_test)})")
    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Count")
    axes[0].set_xlabel(col)
    axes[1].set_xlabel(col)

    plt.tight_layout()
    plt.show()
