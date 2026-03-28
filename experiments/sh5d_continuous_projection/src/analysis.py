"""Statistical analysis for SH5d."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings


FEATURE_GROUPS = {
    'full_embedding': [
        'mean_cosine_dist', 'max_cosine_dist', 'embedding_path_length',
        'embedding_displacement', 'path_efficiency'
    ],
    'slod_axis': [
        'slod_axis_mean', 'slod_axis_variance', 'slod_axis_range',
        'slod_axis_drift_mean', 'slod_axis_drift_max',
        'slod_axis_direction', 'slod_axis_monotonicity'
    ],
    'orthogonal': [
        'orthogonal_drift_mean', 'orthogonal_variance', 'slod_ratio'
    ]
}

ALL_FEATURES = (FEATURE_GROUPS['full_embedding'] +
                FEATURE_GROUPS['slod_axis'] +
                FEATURE_GROUPS['orthogonal'])


def compute_correlations(df: pd.DataFrame,
                         features: list = None,
                         metrics: list = None,
                         n_bonferroni: int = 30) -> pd.DataFrame:
    """Compute Spearman correlations between features and quality metrics.

    Args:
        df: DataFrame with features and metrics
        features: list of feature column names
        metrics: list of metric column names
        n_bonferroni: number of tests for Bonferroni correction

    Returns:
        DataFrame with rho, p-value, corrected p-value, significance for each pair
    """
    if features is None:
        features = ALL_FEATURES
    if metrics is None:
        metrics = ['answer_token_f1', 'attribution_f1']

    results = []
    for feat in features:
        for metric in metrics:
            valid = df[[feat, metric]].dropna()
            if len(valid) < 10:
                results.append({
                    'feature': feat, 'metric': metric,
                    'rho': np.nan, 'p_value': np.nan,
                    'p_corrected': np.nan, 'significant': False, 'n': len(valid)
                })
                continue
            rho, p = spearmanr(valid[feat], valid[metric])
            p_corrected = min(p * n_bonferroni, 1.0)
            results.append({
                'feature': feat, 'metric': metric,
                'rho': float(rho), 'p_value': float(p),
                'p_corrected': float(p_corrected),
                'significant': p_corrected < 0.05,
                'n': len(valid)
            })
    return pd.DataFrame(results)


def logistic_regression_auroc(df: pd.DataFrame,
                               features: list,
                               target_metric: str = 'answer_token_f1',
                               cv_folds: int = 5,
                               random_seed: int = 42) -> float:
    """Train logistic regression to predict above-median quality, return AUROC.

    Args:
        df: DataFrame with features and target metric
        features: feature columns to use
        target_metric: quality metric to binarize
        cv_folds: number of CV folds
        random_seed: random seed

    Returns:
        mean AUROC across folds
    """
    valid = df[features + [target_metric]].dropna()
    if len(valid) < 20:
        return np.nan

    X = valid[features].values
    y = (valid[target_metric] >= valid[target_metric].median()).astype(int).values

    # Check if there's variance
    if y.sum() == 0 or y.sum() == len(y):
        return 0.5

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=random_seed))
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')

    return float(scores.mean())


def run_ablation(df: pd.DataFrame, target_metrics: list = None,
                 cv_folds: int = 5, random_seed: int = 42) -> dict:
    """Run ablation: AUROC for each feature group and combined.

    Returns:
        dict of {group_name: {metric: auroc}}
    """
    if target_metrics is None:
        target_metrics = ['answer_token_f1', 'attribution_f1']

    results = {}
    groups_to_test = dict(FEATURE_GROUPS)
    groups_to_test['all_sh5d'] = ALL_FEATURES

    for group_name, feats in groups_to_test.items():
        results[group_name] = {}
        for metric in target_metrics:
            auroc = logistic_regression_auroc(df, feats, metric, cv_folds, random_seed)
            results[group_name][metric] = auroc
            print(f"  {group_name} -> {metric}: AUROC = {auroc:.4f}")
    return results
