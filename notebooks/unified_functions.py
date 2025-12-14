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

    data_split = []

    if not tune_params:
        print("Training model with default hyperparameters...\n")

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2)

        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model_pipeline = get_model_pipeline(
                model, numerical_cols, categorical_cols)

            X_train_split, X_test_split = X.iloc[train_idx], X.iloc[test_idx]
            y_train_split, y_test_split = y.iloc[train_idx], y.iloc[test_idx]
            data_split.append((X_train_split, y_train_split, X_test_split, y_test_split))

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
        data_split.append((X_train, y_train, X_test, y_test))
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
    
    return y_true, y_pred, y_pred_probs, false_positive_rate, true_positive_rate, total_roc_auc, model_pipeline, best_params, data_split


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


def plot_distribution(df, target_col, title, xlabel, fig_width=1000, fig_height=600, save_path=None):
    plt.figure(figsize=(fig_width/100, fig_height/100), dpi=100)
    ax = sns.barplot(x=df[target_col].value_counts().index, y=df[target_col].value_counts().values, color='#284577')
    ax.bar_label(ax.containers[0], fontsize=16, padding=-1)
    plt.xticks(ticks=[0, 1], labels=['No', 'Yes'], fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel('Number of Patients', fontsize=16)
    plt.title(title, fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()


def plot_roc(false_positive_rate, true_positive_rate, total_roc_auc, fig_width=1000, fig_height=600, save_path=None):
    plt.figure(figsize=(fig_width/100, fig_height/100), dpi=100)
    plt.plot(false_positive_rate, true_positive_rate, color='#284577',
             lw=3, label=f'ROC (AUC = {total_roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('ROC curve', fontsize=18)
    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)

    plt.show()


def plot_confusion_matrix(y_true, y_pred=None, y_pred_probs=None, threshold=0.5, fig_width=800, fig_height=600, save_path=None):
    if y_pred_probs is not None:
        y_pred = (np.array(y_pred_probs) >= threshold).astype(int)
    if y_pred is None:
        raise ValueError("Provide either y_pred or y_pred_probs")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(fig_width/100, fig_height/100), dpi=100)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion matrix (threshold={threshold})')
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


def plot_feature_importances(model, X, y, title, max_vars=6, fig_width=1000, fig_height=600):
    explainer = dx.Explainer(model, X, y)
    importances = explainer.model_parts()
    fig = importances.plot(max_vars=max_vars, show=False)

    fig.update_layout(title_text=title, title_x=0.5, title_font_size=25, title_font_color='black')
    fig.update_layout(font_color='black', font_size=14)
    fig.update_layout(width=fig_width, height=fig_height)
    fig.update_annotations(text="", selector={'text': 'XGBClassifier'})
    fig.update_annotations(font_size=18)
    fig.update_traces(marker_color='#284577') # '#46bac2'
    fig.update_layout(yaxis_tickfont_size=18)
    fig.show()


def plot_discrimination_threshold(model, X, y, fig_width=1000, fig_height=600):
    visualizer = DiscriminationThreshold(model, random_state=2, size=(fig_width, fig_height))
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


def calculate_splines(var):
    PSAPreopKnot1 = 0.2
    PSAPreopKnot2 = 4.8
    PSAPreopKnot3 = 7.35
    PSAPreopKnot4 = 307.0

    psa_spline_1 = max(var - PSAPreopKnot1, 0) ** 3 - max(var - PSAPreopKnot3, 0) ** 3 * (PSAPreopKnot4 - PSAPreopKnot1) / (PSAPreopKnot4 - PSAPreopKnot3) + max(var - PSAPreopKnot4, 0) ** 3 * (PSAPreopKnot3 - PSAPreopKnot1) / (PSAPreopKnot4 - PSAPreopKnot3)
    psa_spline_2 = max(var - PSAPreopKnot2, 0) ** 3 - max(var - PSAPreopKnot3, 0) ** 3 * (PSAPreopKnot4 - PSAPreopKnot2) / (PSAPreopKnot4 - PSAPreopKnot3) + max(var - PSAPreopKnot4, 0) ** 3 * (PSAPreopKnot3 - PSAPreopKnot2) / (PSAPreopKnot4 - PSAPreopKnot3)
    return psa_spline_1, psa_spline_2

def stage(tnm, stage_weights):
    tnm = tnm.lower()
    if 't2a' in tnm: return stage_weights['2a']
    if 't2b' in tnm: return stage_weights['2b']
    if 't2c' in tnm: return stage_weights['2c']
    if 't3' in tnm or 't4' in tnm: return stage_weights['3plus']
    return 0.0

def gleason(isup, gleason_weights):
    if isup == 2: return gleason_weights[2]
    if isup == 3: return gleason_weights[3]
    if isup == 4: return gleason_weights[4]
    if isup == 5: return gleason_weights[5]
    return 0.0

def getCores(patient, split='na', column_P='ilość\xa0+ wycinków P', column_L='Ilość + wycinków L'):
    cores_P = patient[column_P]
    if cores_P == 0:
        pos_P = 0
        neg_P = 0
    else:
        cores_P = cores_P.split(split)
        pos_P = int(cores_P[0])
        neg_P = int(cores_P[1]) - pos_P

    cores_L = patient[column_L]
    if cores_L == 0:
        pos_L = 0
        neg_L = 0
    else:
        cores_L = cores_L.split(split)
        pos_L = int(cores_L[0])
        neg_L = int(cores_L[1]) - pos_L

    return pos_P + pos_L, neg_P + neg_L

def mskcc_predict(patient, target='EPE'):
    # MSKCC coefficients
    coefficients = {
        'EPE': {
            'intercept': -4.3619472,
            'age': 0.02789286,
            'psa': 0.25391759,
            'psa_spline_1': -0.00185119,
            'psa_spline_2': 0.00512636,
            'biopsy_gleason': {
                2: 0.9456123,
                3: 1.38600414,
                4: 1.47229518,
                5: 2.52672521
            },
            'clinical_stage': {
                '2a': 0.29223705,
                '2b': 0.8045909,
                '2c': 0.85209752,
                '3plus': 1.67265063
            }
        },

        'EPE cores': {
            'intercept': -4.14615912,
            'age': 0.03263727,
            'psa': 0.22419499,
            'psa_spline_1': -0.00151357,
            'psa_spline_2': 0.0041806,
            'biopsy_gleason': {
                2: 0.62975509,
                3: 1.04483516,
                4: 1.11988728,
                5: 2.04021401
            },
            'clinical_stage': {
                '2a': 0.21016394,
                '2b': 0.7631916,
                '2c': 0.56884638,
                '3plus': 1.46007476
            },
            'no_positive_cores': 0.08760181,
            'no_negative_cores': -0.06353104
        },

        'N+': {
            'intercept': -5.83057151,
            'age': 0.00521158,
            'psa': 0.18754729,
            'psa_spline_1': -0.00122617,
            'psa_spline_2': 0.0033653,
            'biopsy_gleason': {
                2: 1.52752948,
                3: 2.57873595,
                4: 2.75375893,
                5: 3.50034615
            },
            'clinical_stage': {
                '2a': 0.26172062,
                '2b': 0.55860494,
                '2c': 0.84874365,
                '3plus': 1.09527926
            }
        },

        'N+ cores': {
            'intercept': -5.37408605,
            'age': 0.01061319,
            'psa': 0.22266602,
            'psa_spline_1': -0.001599,
            'psa_spline_2': 0.00441175,
            'biopsy_gleason': {
                2: 0.998897,
                3: 2.03362879,
                4: 2.177025,
                5: 2.87515732
            },
            'clinical_stage': {
                '2a': 0.17084652,
                '2b': 0.49919005,
                '2c': 0.46102494,
                '3plus': 0.74403104
            },
            'no_positive_cores': 0.05972006,
            'no_negative_cores': -0.09466818
        },

        'SVI': {
            'intercept': -5.96514874,
            'age': 0.01303056,
            'psa': 0.16554996,
            'psa_spline_1': -0.00095842,
            'psa_spline_2': 0.00262538,
            'biopsy_gleason': {
                2: 1.24764853,
                3: 2.04455038,
                4: 2.19455366,
                5: 3.17089909
            },
            'clinical_stage': {
                '2a': 0.28661144,
                '2b': 0.58408599,
                '2c': 1.04372486,
                '3plus': 1.23384554
            }
        },

        'SVI cores': {
            'intercept': -6.11427809,
            'age': 0.02483733,
            'psa': 0.1475095,
            'psa_spline_1': -0.00085465,
            'psa_spline_2': 0.00234133,
            'biopsy_gleason': {
                2: 1.11283762,
                3: 1.89864418,
                4: 2.06573755,
                5: 2.97808178
            },
            'clinical_stage': {
                '2a': 0.13563953,
                '2b': 0.43790414,
                '2c': 0.67764059,
                '3plus': 0.80895957
            },
            'no_positive_cores': 0.07682603,
            'no_negative_cores': -0.10972019
        }
    }
    
    psa_spline_1, psa_spline_2 = calculate_splines(patient['PSA'])

    b = coefficients[target]['intercept'] + \
        coefficients[target]['age'] * patient['wiek'] + \
        coefficients[target]['psa'] * patient['PSA'] + \
        coefficients[target]['psa_spline_1'] * psa_spline_1 + \
        coefficients[target]['psa_spline_2'] * psa_spline_2 + \
        gleason(patient['Bx ISUP Grade'], coefficients[target]['biopsy_gleason']) + \
        stage(patient['TNM'], coefficients[target]['clinical_stage'])

    if 'cores' in target:
        pos_cores, neg_cores = getCores(patient)
        b += coefficients[target]['no_positive_cores'] * pos_cores + \
             coefficients[target]['no_negative_cores'] * neg_cores
    
    return np.exp(b) / (1 + np.exp(b))