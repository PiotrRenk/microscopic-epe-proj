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
    data_splits: list[dict[str, DataFrame | Series]]
    """
    List of all data splits used in training.
    When using cross-validation with `n` folds, list will contain `n` dictionaries. When using hold out there will be only one dictionary in the list.
    Each dictionary has following keys:
    - 'X_train': DataFrame with training features for the fold
    - 'y_train': Series with training labels for the fold
    - 'X_test': DataFrame with testing features for the fold
    - 'y_test': Series with testing labels for the fold
    """

    def get_preprocessor(self) -> Pipeline:
        return self.pipeline.named_steps["preprocessor"]
