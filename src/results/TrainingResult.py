from dataclasses import dataclass
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline

@dataclass
class TrainingResult:
    model_name: str
    y_true: NDArray
    y_pred: NDArray
    y_pred_probs: NDArray
    fp_rate: NDArray
    tp_rate: NDArray
    roc_auc: float
    pipeline: Pipeline
    best_params: dict[str, float]
    data_splits: dict[str, DataFrame | Series] # TODO: should be a list in case we use k-fold CV

    def get_preprocessor(self) -> Pipeline:
        return self.pipeline.named_steps['preprocessor']