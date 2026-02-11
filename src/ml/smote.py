"""
Unverified functions!
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTENC

from ml.training import get_model_pipeline, tune_hyperparameters

# TODO: check if SMOTENC is used correctly
def train_and_evaluate_model_with_smote(
    model, X, y, numerical_cols, categorical_cols,
    n_folds=5, tune_params=False, param_grid=None,
    tuning_scoring='roc_auc', tuning_test_size=0.2
):
    y_pred_probs = np.array([])
    y_pred = np.array([])
    y_true = np.array([])

    if not tune_params:
        print("Training model with default hyperparameters...\n")

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)

        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            num_pipeline = Pipeline(
                steps=[('impute', SimpleImputer(strategy='median'))])
            cat_pipeline = Pipeline(
                steps=[('impute', SimpleImputer(strategy='most_frequent'))])
            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ])

            X_train_imputed = preprocessor.fit_transform(X_train)

            smote = SMOTENC(random_state=42, categorical_features=[
                            X.columns.get_loc(col) for col in categorical_cols])
            X_train_smote, y_train_smote = smote.fit_resample(
                X_train_imputed, y_train)

            X_train_smote = pd.DataFrame(
                X_train_smote, columns=numerical_cols + categorical_cols)

            model_pipeline = get_model_pipeline(
                model, numerical_cols, categorical_cols)
            model_pipeline.fit(X_train_smote, y_train_smote)

            y_pred_prob_fold = model_pipeline.predict_proba(X_test)[:, 1]
            y_pred_fold = model_pipeline.predict(X_test)

            y_pred_probs = np.concatenate([y_pred_probs, y_pred_prob_fold])
            y_pred = np.concatenate([y_pred, y_pred_fold])
            y_true = np.concatenate([y_true, y_test])

            print(
                f"Fold {i}: ROC AUC = {roc_auc_score(y_test, y_pred_prob_fold):.4f}")

        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        total_roc_auc = roc_auc_score(y_true, y_pred_probs)
        best_params = model.get_params()
        print(f"\nOverall ROC AUC across folds: {total_roc_auc:.4f}")

        return y_true, y_pred, y_pred_probs, fpr, tpr, total_roc_auc, model, best_params

    else:
        print("Tuning hyperparameters with SMOTE on training set only...\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=tuning_test_size, stratify=y, random_state=42
        )

        num_pipeline = Pipeline(
            steps=[('impute', SimpleImputer(strategy='median'))])
        cat_pipeline = Pipeline(
            steps=[('impute', SimpleImputer(strategy='most_frequent'))])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ])

        X_train_imputed = preprocessor.fit_transform(X_train)

        smote = SMOTENC(random_state=42, categorical_features=[
                        X.columns.get_loc(col) for col in categorical_cols])
        X_train_smote, y_train_smote = smote.fit_resample(
            X_train_imputed, y_train)

        X_train_smote = pd.DataFrame(
            X_train_smote, columns=numerical_cols + categorical_cols)
        best_params = tune_hyperparameters(
            model, param_grid, X_train_smote, y_train_smote,
            numerical_cols, categorical_cols, n_folds=3, scoring=tuning_scoring
        )
        model.set_params(**best_params)
        print(f"Best params after tuning:\n{best_params}")

        model_pipeline = get_model_pipeline(
            model, numerical_cols, categorical_cols)

        model_pipeline.fit(X_train_smote, y_train_smote)

        y_pred_probs = model_pipeline.predict_proba(X_test)[:, 1]
        y_pred = model_pipeline.predict(X_test)
        y_true = y_test

        fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
        total_roc_auc = roc_auc_score(y_test, y_pred_probs)
        print(f"\nROC AUC on test set: {total_roc_auc:.4f}")

        return y_true, y_pred, y_pred_probs, fpr, tpr, total_roc_auc, model, best_params