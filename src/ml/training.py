import numpy as np
import optuna
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from results.EvaluationResult import EvaluationResult
from results.TrainingResult import TrainingResult


def get_model_pipeline(
    model, numerical_cols: list[str], categorical_cols: list[str]
) -> Pipeline:
    num_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("normalization", MinMaxScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_cols),
            ("cat", cat_pipeline, categorical_cols),
        ]
    )
    model_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", model)]
    )
    return model_pipeline


# TODO: check returning of data splits
def train_and_evaluate_model(
    model,
    X: DataFrame,
    y: Series,
    numerical_cols: list[str],
    categorical_cols: list[str],
    n_folds: int = 5,
    tune_params: bool = False,
    param_function: callable | None = None,
    tuning_scoring: str = "roc_auc",
    tuning_direction: str = "maximize",
    tuning_test_size: float = 0.2,
    n_jobs: int = -1,
) -> TrainingResult:
    y_pred_probs = np.array([])
    y_pred = np.array([])
    y_true = np.array([])

    data_splits = []
    model_pipeline: Pipeline = get_model_pipeline(
        model, numerical_cols, categorical_cols)

    if not tune_params:
        print("Training model with default hyperparameters...\n")

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)

        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model_pipeline = get_model_pipeline(
                model, numerical_cols, categorical_cols)

            X_train_split, X_test_split = X.iloc[train_idx], X.iloc[test_idx]
            y_train_split, y_test_split = y.iloc[train_idx], y.iloc[test_idx]
            data_splits.append(
                {
                    "X_train": X_train_split,
                    "y_train": y_train_split,
                    "X_test": X_test_split,
                    "y_test": y_test_split,
                }
            )

            model_pipeline.fit(X_train_split, y_train_split)
            y_pred_prob_fold = model_pipeline.predict_proba(X_test_split)[:, 1]
            y_pred_fold = model_pipeline.predict(X_test_split)
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test_split, y_pred_prob_fold
            )
            y_pred_probs = np.concatenate([y_pred_probs, y_pred_prob_fold])
            y_pred = np.concatenate([y_pred, y_pred_fold])
            y_true = np.concatenate([y_true, y_test_split])
            print(f"Fold {i}:")
            print(
                f"ROC AUC score: {roc_auc_score(y_test_split, y_pred_prob_fold)}\n")

        best_params = model.get_params()

        print("Total scores:")

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, y_pred_probs)
        total_roc_auc = roc_auc_score(y_true, y_pred_probs)
        print(f"ROC AUC score: {total_roc_auc}\n")

    elif tune_params and param_function is not None:
        print("Tuning hyperparameters...\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=tuning_test_size, stratify=y, random_state=42
        )
        data_splits.append(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }
        )
        best_params = tune_hyperparameters(
            model,
            param_function,
            X_train,
            y_train,
            numerical_cols,
            categorical_cols,
            n_folds=3,
            scoring=tuning_scoring,
            direction=tuning_direction,
            n_jobs=n_jobs,
        )
        model.set_params(**best_params)
        print(f"Model hyperparameters after tuning: {model.get_params()}")

        model_pipeline = get_model_pipeline(
            model, numerical_cols, categorical_cols)
        model_pipeline.fit(X_train, y_train)
        y_true = y_test
        y_pred_probs = model_pipeline.predict_proba(X_test)[:, 1]
        y_pred = model_pipeline.predict(X_test)

        print("Total scores:")

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, y_pred_probs)
        total_roc_auc = roc_auc_score(y_test, y_pred_probs)
        print(f"ROC AUC score: {total_roc_auc}\n")

    else:
        raise ValueError(
            "If tune_params is True, param_function must be provided.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred_probs = np.asarray(y_pred_probs)

    return TrainingResult(
        model_name=model.__class__.__name__,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_probs=y_pred_probs,
        fp_rate=false_positive_rate,
        tp_rate=true_positive_rate,
        roc_auc=float(total_roc_auc),
        pipeline=model_pipeline,
        best_params=best_params,
        data_splits=data_splits,
    )


def tune_hyperparameters(
    model,
    param_space_function: callable,
    X: DataFrame,
    y: Series,
    numerical_cols: list[str],
    categorical_cols: list[str],
    n_trials: int = 100,
    n_folds: int = 5,
    scoring: str = "roc_auc",
    direction: str = "maximize",
    n_jobs: int = -1,
) -> dict:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)

    def objective(trial):
        params = param_space_function(trial)

        pipeline_params = {f"classifier__{k}": v for k, v in params.items()}

        model_pipeline = get_model_pipeline(
            model, numerical_cols, categorical_cols)
        model_pipeline.set_params(**pipeline_params)

        scores = cross_val_score(
            model_pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )
        return np.mean(scores)

    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    print(f"Best hyperparameters: {study.best_params}")
    print(f"Best {scoring} score: {study.best_value:.4f}\n")

    return study.best_params


def train_model(
    model,
    X: DataFrame,
    y: Series,
    numerical_cols: list[str],
    categorical_cols: list[str],
    n_folds: int = 5,
    tune_params: bool = False,
    param_function: callable | None = None,
    tuning_scoring: str = "roc_auc",
    tuning_direction: str = "maximize",
    tuning_test_size: float = 0.2,
    n_jobs: int = -1,
) -> TrainingResult:
    y_pred_probs = np.array([])
    y_pred = np.array([])
    y_true = np.array([])

    data_splits = []
    model_pipeline: Pipeline = get_model_pipeline(
        model, numerical_cols, categorical_cols)

    if not tune_params:
        # this is just a regular cross-validation

        print("Training model with default hyperparameters...\n")

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)

        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model_pipeline = get_model_pipeline(
                model, numerical_cols, categorical_cols)

            X_train_split, X_test_split = X.iloc[train_idx], X.iloc[test_idx]
            y_train_split, y_test_split = y.iloc[train_idx], y.iloc[test_idx]
            data_splits.append(
                {
                    "X_train": X_train_split,
                    "y_train": y_train_split,
                    "X_test": X_test_split,
                    "y_test": y_test_split,
                }
            )

            model_pipeline.fit(X_train_split, y_train_split)
            y_pred_prob_fold = model_pipeline.predict_proba(X_test_split)[:, 1]
            y_pred_fold = model_pipeline.predict(X_test_split)
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test_split, y_pred_prob_fold
            )
            y_pred_probs = np.concatenate([y_pred_probs, y_pred_prob_fold])
            y_pred = np.concatenate([y_pred, y_pred_fold])
            y_true = np.concatenate([y_true, y_test_split])
            print(f"Fold {i}:")
            print(
                f"ROC AUC score: {roc_auc_score(y_test_split, y_pred_prob_fold)}\n")

        # compute global metrics
        best_params = model.get_params()
        print("Total scores:")
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, y_pred_probs)
        total_roc_auc = roc_auc_score(y_true, y_pred_probs)
        print(f"ROC AUC score: {total_roc_auc}\n")

        # train model on all data points
        fully_trained_pipeline = get_model_pipeline(
            model, numerical_cols, categorical_cols)
        fully_trained_pipeline.fit(X, y)

    elif tune_params and param_function is not None:
        # first we perform hyperparameter tuning on the holdout training set
        # doing cross-validation on it
        # then we train the model on the whole training set with best hyperparams
        # and evaluate it on the validation set
        print("Tuning hyperparameters...\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=tuning_test_size, stratify=y, random_state=42
        )
        data_splits.append(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
            }
        )
        best_params = tune_hyperparameters(
            model,
            param_function,
            X_train,
            y_train,
            numerical_cols,
            categorical_cols,
            n_folds=3,
            scoring=tuning_scoring,
            direction=tuning_direction,
            n_jobs=n_jobs,
        )
        model.set_params(**best_params)
        print(f"Model hyperparameters after tuning: {model.get_params()}")

        model_pipeline = get_model_pipeline(
            model, numerical_cols, categorical_cols)
        model_pipeline.fit(X_train, y_train)
        y_true = y_test
        y_pred_probs = model_pipeline.predict_proba(X_test)[:, 1]
        y_pred = model_pipeline.predict(X_test)

        print("Total scores:")

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, y_pred_probs)
        total_roc_auc = roc_auc_score(y_test, y_pred_probs)
        print(f"ROC AUC score: {total_roc_auc}\n")

        # train the model on all data points
        fully_trained_pipeline = get_model_pipeline(
            model, numerical_cols, categorical_cols)
        fully_trained_pipeline.fit(X, y)

    else:
        raise ValueError(
            "If tune_params is True, param_function must be provided.")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred_probs = np.asarray(y_pred_probs)

    return TrainingResult(
        model_name=model.__class__.__name__,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_probs=y_pred_probs,
        fp_rate=false_positive_rate,
        tp_rate=true_positive_rate,
        roc_auc=float(total_roc_auc),
        pipeline=model_pipeline,
        best_params=best_params,
        data_splits=data_splits,
        fully_trained_pipeline=fully_trained_pipeline
    )


def evaluate_model(
    model_pipeline: Pipeline,
    X: DataFrame,
    y: Series
) -> EvaluationResult:
    y_pred_probs = model_pipeline.predict_proba(X)[:, 1]
    y_pred = model_pipeline.predict(X)
    y_true = y.values

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true, y_pred_probs)
    roc_auc = roc_auc_score(y_true, y_pred_probs)

    print("Evaluation results:")
    print(f"ROC AUC score: {roc_auc}")

    return EvaluationResult(
        model_name=model_pipeline.named_steps["classifier"].__class__.__name__,
        y_true=y_true,
        y_pred=y_pred,
        y_pred_probs=y_pred_probs,
        fp_rate=false_positive_rate,
        tp_rate=true_positive_rate,
        roc_auc=float(roc_auc)
    )
