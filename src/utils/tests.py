import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def delong_roc_test(y_true, y_pred1, y_pred2):
    """
    Compare two ROC curves using DeLong's test.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred1 : array-like
        Predicted probabilities from model 1
    y_pred2 : array-like
        Predicted probabilities from model 2

    Returns:
    --------
    z_score : float
        Z-statistic for the difference
    p_value : float
        Two-tailed p-value
    """

    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)

    def compute_ground_truth_statistics(ground_truth):
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        order = (-ground_truth).argsort()
        label_1_count = int(ground_truth.sum())
        return order, label_1_count

    def compute_midrank_weight(x, sample_weight):
        unique_values, unique_inverse, unique_counts = np.unique(
            x, return_inverse=True, return_counts=True
        )
        cumsum = np.cumsum(unique_counts)
        midrank = cumsum - unique_counts / 2.0
        return midrank[unique_inverse]

    def fast_delong(predictions_sorted_transposed, label_1_count, sample_weight=None):
        m = label_1_count
        n = predictions_sorted_transposed.shape[1] - m
        positive_examples = predictions_sorted_transposed[:, :m]
        negative_examples = predictions_sorted_transposed[:, m:]
        k = predictions_sorted_transposed.shape[0]

        tx = np.empty([k, m], dtype=float)
        ty = np.empty([k, n], dtype=float)
        tz = np.empty([k, m + n], dtype=float)

        for r in range(k):
            tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight)
            ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight)
            tz[r, :] = compute_midrank_weight(
                predictions_sorted_transposed[r, :], sample_weight
            )

        aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
        v01 = (tz[:, :m] - tx[:, :]) / n
        v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

        sx = np.cov(v01)
        sy = np.cov(v10)

        delongcov = sx / m + sy / n
        return aucs, delongcov

    order, label_1_count = compute_ground_truth_statistics(y_true)
    predictions_sorted_transposed = np.vstack([y_pred1, y_pred2])[:, order]

    aucs, delongcov = fast_delong(predictions_sorted_transposed, label_1_count)

    z_score = (aucs[0] - aucs[1]) / np.sqrt(
        delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    )
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return z_score, p_value, auc1, auc2


def bootstrap_auc_ci(y_true, y_pred_probs, n_bootstraps=2000, confidence_level=0.95):
    """
    Calculate bootstrap confidence interval for AUC.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_probs : array-like
        Predicted probabilities
    n_bootstraps : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)

    Returns:
    --------
    auc : float
        Original AUC
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.utils import resample

    auc = roc_auc_score(y_true, y_pred_probs)

    # Bootstrap
    bootstrapped_aucs = []
    for i in range(n_bootstraps):
        # Resample with replacement
        indices = resample(
            range(len(y_true)), replace=True, n_samples=len(y_true), random_state=i
        )
        y_true_boot = (
            y_true.iloc[indices] if hasattr(y_true, "iloc") else y_true[indices]
        )
        y_pred_boot = y_pred_probs[indices]

        try:
            boot_auc = roc_auc_score(y_true_boot, y_pred_boot)
            bootstrapped_aucs.append(boot_auc)
        except Exception:
            continue

    # Calculate percentile confidence interval
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(bootstrapped_aucs, alpha * 100)
    ci_upper = np.percentile(bootstrapped_aucs, (1 - alpha) * 100)

    return auc, ci_lower, ci_upper


def hanley_mcneil_test(auc1, n1, n_pos1, auc2, n2, n_pos2):
    """
    Compare two AUCs from different studies using Hanley-McNeil method.
    This is an APPROXIMATE test when you don't have the raw data.

    Parameters:
    -----------
    auc1, auc2 : float
        AUC values to compare
    n1, n2 : int
        Total sample sizes
    n_pos1, n_pos2 : int
        Number of positive cases in each sample

    Returns:
    --------
    z_score : float
        Z-statistic
    p_value : float
        Two-tailed p-value
    """
    from scipy import stats

    # Number of negative cases
    n_neg1 = n1 - n_pos1
    n_neg2 = n2 - n_pos2

    # Hanley-McNeil variance estimation
    Q1_1 = auc1 / (2 - auc1)
    Q2_1 = 2 * auc1**2 / (1 + auc1)
    se1_sq = (
        auc1 * (1 - auc1)
        + (n_pos1 - 1) * (Q1_1 - auc1**2)
        + (n_neg1 - 1) * (Q2_1 - auc1**2)
    ) / (n_pos1 * n_neg1)

    Q1_2 = auc2 / (2 - auc2)
    Q2_2 = 2 * auc2**2 / (1 + auc2)
    se2_sq = (
        auc2 * (1 - auc2)
        + (n_pos2 - 1) * (Q1_2 - auc2**2)
        + (n_neg2 - 1) * (Q2_2 - auc2**2)
    ) / (n_pos2 * n_neg2)

    # Z-score (assumes independence between samples)
    se_diff = np.sqrt(se1_sq + se2_sq)
    z_score = (auc1 - auc2) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    return z_score, p_value, se1_sq, se2_sq
