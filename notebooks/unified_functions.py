import dalex as dx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from yellowbrick.classifier import DiscriminationThreshold


def get_model_pipeline(model, numerical_cols, categorical_cols):
    num_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('normalization', MinMaxScaler())
    ])
    cat_pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return model_pipeline


def train_and_evaluate_model(model, X, y, numerical_cols, categorical_cols, n_folds, tune_params=False, param_grid=None, tuning_scoring='roc_auc', tuning_test_size=0.2):
    y_pred_probs = np.array([])
    y_pred = np.array([])
    y_true = np.array([])

    if not tune_params:
        print("Training model with default hyperparameters...\n")

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)

        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model_pipeline = get_model_pipeline(
                model, numerical_cols, categorical_cols)

            X_train_split, X_test_split = X.iloc[train_idx], X.iloc[test_idx]
            y_train_split, y_test_split = y.iloc[train_idx], y.iloc[test_idx]
            model_pipeline.fit(X_train_split, y_train_split)
            y_pred_prob_fold = model_pipeline.predict_proba(X_test_split)[:, 1]
            y_pred_fold = model_pipeline.predict(X_test_split)
            false_positive_rate, true_positive_rate, _ = roc_curve(
                y_test_split, y_pred_prob_fold)
            y_pred_probs = np.concatenate([y_pred_probs, y_pred_prob_fold])
            y_pred = np.concatenate([y_pred, y_pred_fold])
            y_true = np.concatenate([y_true, y_test_split])
            print(f"Fold {i}:")
            print(
                f"ROC AUC score: {roc_auc_score(y_test_split, y_pred_prob_fold)}\n")

        best_params = model.get_params()

        print(f"Total scores:")

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true, y_pred_probs)
        total_roc_auc = roc_auc_score(y_true, y_pred_probs)
        print(f"ROC AUC score: {total_roc_auc}\n")

    elif tune_params and param_grid is not None:
        print("Tuning hyperparameters...\n")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=tuning_test_size, stratify=y, random_state=42)

        best_params = tune_hyperparameters(
            model, param_grid, X_train, y_train, numerical_cols, categorical_cols, n_folds=3, scoring=tuning_scoring)
        model.set_params(**best_params)
        print(f"Model hyperparameters after tuning: {model.get_params()}")

        model_pipeline = get_model_pipeline(
            model, numerical_cols, categorical_cols)
        model_pipeline.fit(X_train, y_train)
        y_true = y_test
        y_pred_probs = model_pipeline.predict_proba(X_test)[:, 1]
        y_pred = model_pipeline.predict(X_test)

        print(f"Total scores:")

        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, y_pred_probs)
        total_roc_auc = roc_auc_score(y_test, y_pred_probs)
        print(f"ROC AUC score: {total_roc_auc}\n")

    return y_true, y_pred, y_pred_probs, false_positive_rate, true_positive_rate, total_roc_auc, model_pipeline, best_params


def tune_hyperparameters(model, param_grid, X, y, numerical_cols, categorical_cols, n_folds, scoring='roc_auc'):
    model_pipeline = get_model_pipeline(
        model, numerical_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)
    grid_search = GridSearchCV(
        estimator=model_pipeline, param_grid=param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid_search.fit(X, y)
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Best {scoring} score: {grid_search.best_score_}\n")
    best_params = grid_search.best_params_
    best_params = {key.replace('classifier__', ''): value for key, value in best_params.items()}
    return best_params


def plot_roc(false_positive_rate, true_positive_rate, total_roc_auc):
    plt.figure(figsize=(6, 5))
    plt.plot(false_positive_rate, true_positive_rate, color='blue',
             lw=2, label=f'ROC (AUC = {total_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_true, y_pred=None, y_pred_probs=None, threshold=0.5):
    if y_pred_probs is not None:
        y_pred = (np.array(y_pred_probs) >= threshold).astype(int)
    if y_pred is None:
        raise ValueError("Provide either y_pred or y_pred_probs")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion matrix (threshold={threshold})')
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print("=" * 30)
    print(f"Sensitivity: {tp / (tp + fn):.4f}")
    print(f"Specificity: {tn / (tn + fp):.4f}")
    print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
    print("=" * 30)


def plot_confusion_matrix_multiclass(y_true, y_pred=None, y_pred_probs=None, threshold=0.5, label_encoder=None):
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

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion matrix (threshold={threshold})')
    plt.show()


def plot_feature_importances(model, X, y):
    explainer = dx.Explainer(model, X, y)
    importances = explainer.model_parts()
    importances.plot()


def plot_discrimination_threshold(model, X, y):
    visualizer = DiscriminationThreshold(model, random_state=2)
    visualizer.fit(X, y)
    visualizer.show()
    visualizer.show()


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

def discrimination_threshold(y_true, y_pred_prob, method='f1'):
    if method == 'f1':
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        scores = 2 * (precision * recall) / (precision + recall)
    elif method == 'accuracy':
        thresholds = np.unique(y_pred_prob)
        scores = np.array([])
        for threshold in thresholds:
            y_pred = (y_pred_prob >= threshold).astype(int)
            accuracy = np.mean(y_pred == y_true)
            scores = np.append(scores, accuracy)
    best_threshold_idx = np.argmax(scores)
    best_threshold = thresholds[best_threshold_idx]
    return best_threshold