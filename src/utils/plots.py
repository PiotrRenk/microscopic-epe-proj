"""
Plotting functions:
- `plot_distribution(df, target_col)`
- `plot_roc(false_positive_rate, true_positive_rate, total_roc_auc)`
- `plot_confusion_matrix(y_true, y_pred=None, y_pred_probs=None, threshold=0.5)`
- `plot_confusion_matrix_multiclass(y_true, y_pred=None, y_pred_probs=None, threshold=0.5, label_encoder=None)`
- `plot_feature_importance(model, X, y, title=None, max_vars=6)`
- `plot_discrimination_threshold(model, X, y)`
- `plot_threshold_tradeoff(y_true, y_pred_probs)`

"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dalex import Explainer
from yellowbrick.classifier import DiscriminationThreshold

from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.preprocessing import LabelEncoder


def plot_distribution(
    df: DataFrame,
    target_col: str,
    title: str | None = None,
    xlabel: str | None = None,
    fig_width: int = 1000,
    fig_height: int = 600,
    save_path: str | None = None,
):
    plt.figure(figsize=(fig_width / 100, fig_height / 100), dpi=100)
    ax = sns.barplot(
        x=df[target_col].value_counts().index,
        y=df[target_col].value_counts().values,
        color="#284577",
    )
    ax.bar_label(ax.containers[0], fontsize=16, padding=-1)
    plt.xticks(ticks=[0, 1], labels=["No", "Yes"], fontsize=14)
    plt.yticks(fontsize=14)
    if xlabel is None:
        xlabel = target_col
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Number of Patients", fontsize=16)
    if title is None:
        title = f"Distribution of {target_col}"
    plt.title(title, fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()


def plot_roc(
    false_positive_rate: NDArray,
    true_positive_rate: NDArray,
    total_roc_auc: float,
    fig_width: int = 1000,
    fig_height: int = 600,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(fig_width / 100, fig_height / 100), dpi=100)
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="#284577",
        lw=3,
        label=f"ROC (AUC = {total_roc_auc:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("ROC curve", fontsize=18)
    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()


def plot_confusion_matrix(
    y_true: NDArray,
    y_pred: NDArray | None = None,
    y_pred_probs: NDArray | None = None,
    threshold: float = 0.5,
    fig_width: int = 800,
    fig_height: int = 600,
    save_path: str | None = None,
) -> None:
    if y_pred_probs is not None:
        y_pred = (np.array(y_pred_probs) >= threshold).astype(int)
    if y_pred is None:
        raise ValueError("Provide either y_pred or y_pred_probs")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(fig_width / 100, fig_height / 100), dpi=100)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix (threshold={threshold})")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print("=" * 30)
    print(f"Sensitivity: {tp / (tp + fn):.4f}")
    print(f"Specificity: {tn / (tn + fp):.4f}")
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
    print("=" * 30)


def plot_confusion_matrix_multiclass(
    y_true: NDArray,
    y_pred: NDArray | None = None,
    y_pred_probs: NDArray | None = None,
    threshold: float = 0.5,
    label_encoder: LabelEncoder | None = None,
) -> None:
    if y_pred_probs is not None:
        y_pred = (np.array(y_pred_probs) >= threshold).astype(int)
    if y_pred is None:
        raise ValueError("Provide either y_pred or y_pred_probs")

    if label_encoder is not None:
        class_names = label_encoder.classes_
        y_true_decoded = label_encoder.inverse_transform(y_true.astype(int))
        y_pred_decoded = label_encoder.inverse_transform(y_pred.astype(int))
    else:
        class_names = None
        y_true_decoded = y_true
        y_pred_decoded = y_pred

    cm = confusion_matrix(y_true_decoded, y_pred_decoded, labels=class_names)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix (threshold={threshold})")
    plt.show()


def plot_feature_importance(
    model,
    X: DataFrame,
    y: Series,
    title: str | None = None,
    max_vars: int = 6,
    fig_width: int = 1000,
    fig_height: int = 600,
) -> None:
    explainer = Explainer(model, X, y)
    importance = explainer.model_parts()
    fig = importance.plot(max_vars=max_vars, show=False)
    assert fig is not None
    
    if title is None:
        fig.update_layout(
            title_text=title, title_x=0.5, title_font_size=25, title_font_color="black"
        )
    fig.update_layout(font_color="black", font_size=14)
    fig.update_layout(width=fig_width, height=fig_height)
    fig.update_annotations(text="", selector={"text": "XGBClassifier"})
    fig.update_annotations(font_size=18)
    fig.update_traces(marker_color="#284577")  # '#46bac2'
    fig.update_layout(yaxis_tickfont_size=18)
    fig.show()


def plot_discrimination_threshold(
    model,
    X: DataFrame,
    y: Series,
    is_fitted: str | bool = "auto",
    fig_width: int = 1000,
    fig_height: int = 600,
) -> None:
    visualizer = DiscriminationThreshold(
        model,
        random_state=2,
        is_fitted=is_fitted,
        size=(fig_width, fig_height),
    )
    visualizer.fit(X, y)
    visualizer.show()
    visualizer.show()


# TODO: add fig size, save path, color
def plot_threshold_tradeoff(y_true: NDArray, y_pred_probs: NDArray) -> None:
    thresholds = np.arange(0.0, 1.01, 0.01)
    sensitivities = []
    specificities = []
    accuracies = []

    for threshold in thresholds:
        y_pred = (y_pred_probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        accuracies.append(accuracy)

    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    intersection_idx = np.argwhere(
        np.isclose(sensitivities, specificities, atol=0.05)
    ).flatten()
    min_diff = float("inf")
    best_thresh = -1
    for t, v, v2 in zip(
        thresholds[intersection_idx],
        sensitivities[intersection_idx],
        specificities[intersection_idx],
    ):
        diff = abs(v - v2)
        if diff < min_diff:
            min_diff = diff
            best_thresh = t

    print(f"Intersection at threshold: {best_thresh:.2f}, diff: {min_diff}")

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, sensitivities, label="Sensitivity", color="blue")
    plt.plot(thresholds, specificities, label="Specificity", color="orange")
    plt.plot(thresholds, accuracies, label="Accuracy", color="green")
    plt.axvline(
        x=best_thresh,
        color="red",
        linestyle="--",
        label=f"Intersection: {best_thresh:.2f}",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Metric Value")
    plt.legend(frameon=True)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.grid(True)
    plt.show()
