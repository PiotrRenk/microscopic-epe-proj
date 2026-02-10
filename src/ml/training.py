import numpy as np
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve

from src.results.TrainingResult import TrainingResult


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
    param_grid: dict | None = None,
    tuning_scoring: str = "roc_auc",
    tuning_test_size: float = 0.2,
) -> TrainingResult:
    y_pred_probs = np.array([])
    y_pred = np.array([])
    y_true = np.array([])

    data_split = []
    model_pipeline: Pipeline | None = None

    if not tune_params:
        print("Training model with default hyperparameters...\n")

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)

        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model_pipeline = get_model_pipeline(model, numerical_cols, categorical_cols)

            X_train_split, X_test_split = X.iloc[train_idx], X.iloc[test_idx]
            y_train_split, y_test_split = y.iloc[train_idx], y.iloc[test_idx]
            data_split.append(
                (X_train_split, y_train_split, X_test_split, y_test_split)
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
            print(f"ROC AUC score: {roc_auc_score(y_test_split, y_pred_prob_fold)}\n")

        best_params = model.get_params()

        print("Total scores:")

        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_pred_probs)
        total_roc_auc = roc_auc_score(y_true, y_pred_probs)
        print(f"ROC AUC score: {total_roc_auc}\n")

    elif tune_params and param_grid is not None:
        print("Tuning hyperparameters...\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=tuning_test_size, stratify=y, random_state=42
        )
        data_split.append((X_train, y_train, X_test, y_test))
        best_params = tune_hyperparameters(
            model,
            param_grid,
            X_train,
            y_train,
            numerical_cols,
            categorical_cols,
            n_folds=3,
            scoring=tuning_scoring,
        )
        model.set_params(**best_params)
        print(f"Model hyperparameters after tuning: {model.get_params()}")

        model_pipeline = get_model_pipeline(model, numerical_cols, categorical_cols)
        model_pipeline.fit(X_train, y_train)
        y_true = y_test
        y_pred_probs = model_pipeline.predict_proba(X_test)[:, 1]
        y_pred = model_pipeline.predict(X_test)

        print("Total scores:")

        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred_probs)
        total_roc_auc = roc_auc_score(y_test, y_pred_probs)
        print(f"ROC AUC score: {total_roc_auc}\n")

    else:
        raise ValueError("If tune_params is True, param_grid must be provided.")

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
        roc_auc=total_roc_auc,
        pipeline=model_pipeline,
        best_params=best_params,
        data_splits={
            "X_train": data_split[0][0],
            "y_train": data_split[0][1],
            "X_test": data_split[0][2],
            "y_test": data_split[0][3],
        },
    )


def tune_hyperparameters(
    model,
    param_grid: dict,
    X: DataFrame,
    y: Series,
    numerical_cols: list[str],
    categorical_cols: list[str],
    n_folds: int = 5,
    scoring: str = "roc_auc",
):
    model_pipeline = get_model_pipeline(model, numerical_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)
    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
    )
    grid_search.fit(X, y)
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best {scoring} score: {grid_search.best_score_}\n")
    best_params = grid_search.best_params_
    best_params = {
        key.replace("classifier__", ""): value for key, value in best_params.items()
    }
    return best_params
