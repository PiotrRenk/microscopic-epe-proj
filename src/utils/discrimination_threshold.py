import numpy as np
from sklearn.metrics import precision_recall_curve

from numpy.typing import NDArray


def get_discrimination_threshold(
    y_true: NDArray,
    y_pred_prob: NDArray,
    method: str = "f1",
):
    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        scores = 2 * (precision * recall) / (precision + recall)
    elif method == "accuracy":
        thresholds = np.unique(y_pred_prob)
        scores = np.array([])
        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)
            accuracy = np.mean(y_pred == y_true)
            scores = np.append(scores, accuracy)
    else:
        raise ValueError("Method must be either 'f1' or 'accuracy'.")

    best_threshold_idx = np.argmax(scores)
    best_threshold = thresholds[best_threshold_idx]
    return best_threshold
