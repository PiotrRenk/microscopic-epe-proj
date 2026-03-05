from dataclasses import dataclass

from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline


@dataclass
class EvaluationResult:
    model_name: str
    y_true: NDArray
    y_pred: NDArray
    y_pred_probs: NDArray
    fp_rate: NDArray
    tp_rate: NDArray
    roc_auc: float
