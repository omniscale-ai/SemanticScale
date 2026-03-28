"""Stage C: Statistical Analysis.

Spearman correlations, Wilcoxon signed-rank tests, logistic regression,
and partial correlations for alignment features vs answer quality.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

ALIGNMENT_FEATURES = [
    "mean_alignment_gap", "max_alignment_gap", "jsd",
    "dominant_level_match", "context_reasoning_correlation",
    "weighted_alignment_gap", "context_diversity", "reasoning_diversity",
    "diversity_ratio", "soft_mean_gap", "soft_jsd",
]

JUMP_FEATURES = [
    "normalized_jump_rate", "slod_variance", "mean_abs_delta",
    "max_jump", "direction_changes",
]

QUALITY_METRICS = ["answer_token_f1", "attribution_f1"]


def load_alignment_data(path: Path) -> pd.DataFrame:
    """Load alignment features JSONL into DataFrame."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def compute_correlations(df: pd.DataFrame, features: list[str],
                         quality_metrics: list[str],
                         bonferroni_n: int,
                         alpha: float = 0.05) -> dict:
    """Compute Spearman correlations between features and quality metrics.

    Returns dict with results per feature per metric.
    """
    results = {}
    for feat in features:
        results[feat] = {}
        for metric in quality_metrics:
            valid = df[[feat, metric]].dropna()
            if len(valid) < 10:
                results[feat][metric] = {
                    "rho": float("nan"), "p_raw": float("nan"),
                    "p_corrected": float("nan"), "n": len(valid),
                    "significant": False,
                }
                continue
            rho, p = stats.spearmanr(valid[feat], valid[metric])
            p_corrected = min(p * bonferroni_n, 1.0)
            results[feat][metric] = {
                "rho": float(rho),
                "p_raw": float(p),
                "p_corrected": float(p_corrected),
                "n": len(valid),
                "significant": bool(p_corrected < alpha),
            }
    return results


def compute_correlations_by_condition(df: pd.DataFrame, features: list[str],
                                      quality_metrics: list[str]) -> dict:
    """Compute Spearman correlations per condition (no Bonferroni)."""
    results = {}
    for cond in df["condition"].unique():
        sub = df[df["condition"] == cond]
        results[cond] = {}
        for feat in features:
            results[cond][feat] = {}
            for metric in quality_metrics:
                valid = sub[[feat, metric]].dropna()
                if len(valid) < 10:
                    results[cond][feat][metric] = {"rho": float("nan"), "p": float("nan"), "n": len(valid)}
                    continue
                rho, p = stats.spearmanr(valid[feat], valid[metric])
                results[cond][feat][metric] = {"rho": float(rho), "p": float(p), "n": len(valid)}
    return results


def compute_wilcoxon_tests(df: pd.DataFrame, condition_pairs: list,
                           features: list[str]) -> dict:
    """Compute Wilcoxon signed-rank tests for paired condition comparisons.

    Pairs traces by question_id, tests if routed condition has lower alignment gap.
    """
    results = {}
    for routed, baseline in condition_pairs:
        pair_key = f"{routed}_vs_{baseline}"
        results[pair_key] = {}

        df_routed = df[df["condition"] == routed].set_index("question_id")
        df_baseline = df[df["condition"] == baseline].set_index("question_id")
        common_qids = df_routed.index.intersection(df_baseline.index)

        for feat in features:
            vals_routed = df_routed.loc[common_qids, feat].dropna()
            vals_baseline = df_baseline.loc[common_qids, feat].dropna()
            common = vals_routed.index.intersection(vals_baseline.index)

            if len(common) < 10:
                results[pair_key][feat] = {
                    "stat": float("nan"), "p": float("nan"),
                    "n_pairs": len(common),
                    "mean_routed": float("nan"), "mean_baseline": float("nan"),
                    "significant": False,
                }
                continue

            x = vals_routed.loc[common].values
            y = vals_baseline.loc[common].values
            diff = x - y

            # Remove zeros (ties)
            nonzero = diff[diff != 0]
            if len(nonzero) < 5:
                results[pair_key][feat] = {
                    "stat": float("nan"), "p": float("nan"),
                    "n_pairs": len(common),
                    "mean_routed": float(np.mean(x)), "mean_baseline": float(np.mean(y)),
                    "significant": False,
                }
                continue

            try:
                stat, p = stats.wilcoxon(x, y, alternative="two-sided")
                results[pair_key][feat] = {
                    "stat": float(stat),
                    "p": float(p),
                    "n_pairs": len(common),
                    "mean_routed": float(np.mean(x)),
                    "mean_baseline": float(np.mean(y)),
                    "routed_lower": bool(np.mean(x) < np.mean(y)),
                    "significant": bool(p < 0.05),
                }
            except Exception as e:
                results[pair_key][feat] = {
                    "stat": float("nan"), "p": float("nan"),
                    "n_pairs": len(common),
                    "mean_routed": float(np.mean(x)), "mean_baseline": float(np.mean(y)),
                    "significant": False, "error": str(e),
                }

    return results


def compute_logistic_regression(df: pd.DataFrame, feature_sets: dict,
                                target_metric: str = "answer_token_f1",
                                n_folds: int = 5, random_state: int = 42) -> dict:
    """Compute logistic regression predicting above-median quality.

    Args:
        df: DataFrame with features and quality metrics
        feature_sets: dict of {model_name: list_of_feature_names}
        target_metric: quality metric to binarize at median
        n_folds: number of CV folds
        random_state: random seed

    Returns:
        dict with AUROC per model
    """
    # Create binary target
    median_val = df[target_metric].median()
    y = (df[target_metric] > median_val).astype(int)

    results = {}
    for model_name, features in feature_sets.items():
        # Select features and drop NaN rows
        available = [f for f in features if f in df.columns]
        if not available:
            results[model_name] = {"auroc_mean": float("nan"), "auroc_std": float("nan"),
                                   "n_features": 0, "n_samples": 0}
            continue

        sub = df[available + [target_metric]].dropna()
        X = sub[available].values
        y_sub = (sub[target_metric] > median_val).astype(int).values

        if len(np.unique(y_sub)) < 2 or len(y_sub) < 20:
            results[model_name] = {"auroc_mean": float("nan"), "auroc_std": float("nan"),
                                   "n_features": len(available), "n_samples": len(y_sub)}
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Stratified k-fold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        aurocs = []
        coefficients = []

        for train_idx, test_idx in skf.split(X_scaled, y_sub):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_sub[train_idx], y_sub[test_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LogisticRegression(max_iter=1000, random_state=random_state)
                model.fit(X_train, y_train)

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)

            try:
                auroc = roc_auc_score(y_test, y_prob)
                aurocs.append(auroc)
                coefficients.append(model.coef_[0].tolist())
            except ValueError:
                pass

        if aurocs:
            # Average coefficients for feature importance
            avg_coef = np.mean(coefficients, axis=0).tolist()
            feat_importance = {f: c for f, c in zip(available, avg_coef)}

            results[model_name] = {
                "auroc_mean": float(np.mean(aurocs)),
                "auroc_std": float(np.std(aurocs)),
                "n_features": len(available),
                "n_samples": len(y_sub),
                "feature_importance": feat_importance,
                "per_fold_auroc": [float(a) for a in aurocs],
            }
        else:
            results[model_name] = {"auroc_mean": float("nan"), "auroc_std": float("nan"),
                                   "n_features": len(available), "n_samples": len(y_sub)}

    return results


def compute_partial_correlations(df: pd.DataFrame, features: list[str],
                                 quality_metrics: list[str],
                                 control_vars: list[str]) -> dict:
    """Compute partial Spearman correlations controlling for confounders.

    Uses rank-based partial correlation (rank everything, then partial out).
    """
    results = {}

    # Prepare control variables (handle categorical answer_type)
    df_work = df.copy()
    if "answer_type" in control_vars:
        dummies = pd.get_dummies(df_work["answer_type"], prefix="atype", drop_first=True)
        df_work = pd.concat([df_work, dummies], axis=1)
        control_cols = [c for c in control_vars if c != "answer_type"] + list(dummies.columns)
    else:
        control_cols = control_vars

    for feat in features:
        results[feat] = {}
        for metric in quality_metrics:
            cols_needed = [feat, metric] + control_cols
            valid = df_work[cols_needed].dropna()
            if len(valid) < 20:
                results[feat][metric] = {"partial_rho": float("nan"), "n": len(valid)}
                continue

            # Rank-transform
            ranked = valid.rank()

            # Residualize feat and metric on controls
            try:
                from numpy.linalg import lstsq
                X_ctrl = ranked[control_cols].values
                X_ctrl = np.column_stack([X_ctrl, np.ones(len(X_ctrl))])

                y_feat = ranked[feat].values
                y_metric = ranked[metric].values

                coef_feat, _, _, _ = lstsq(X_ctrl, y_feat, rcond=None)
                coef_metric, _, _, _ = lstsq(X_ctrl, y_metric, rcond=None)

                resid_feat = y_feat - X_ctrl @ coef_feat
                resid_metric = y_metric - X_ctrl @ coef_metric

                rho, p = stats.spearmanr(resid_feat, resid_metric)
                results[feat][metric] = {
                    "partial_rho": float(rho),
                    "p": float(p),
                    "n": len(valid),
                }
            except Exception as e:
                results[feat][metric] = {"partial_rho": float("nan"), "n": len(valid),
                                         "error": str(e)}

    return results


def run_all_statistics(alignment_path: Path, output_dir: Path, config: dict):
    """Run all statistical analyses and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_alignment_data(alignment_path)
    print(f"Loaded {len(df)} traces for statistical analysis")

    alpha = config["statistics"]["alpha"]
    bonferroni_n = config["statistics"]["bonferroni_n_tests"]
    condition_pairs = [tuple(p) for p in config["statistics"]["condition_pairs"]]
    n_folds = config["statistics"]["logistic"]["cv_folds"]
    random_state = config["statistics"]["logistic"]["random_state"]

    # C1: Spearman correlations (pooled)
    print("\n=== C1: Spearman Correlations (pooled) ===")
    corr_results = compute_correlations(df, ALIGNMENT_FEATURES, QUALITY_METRICS,
                                        bonferroni_n, alpha)
    # Also per-condition
    corr_by_cond = compute_correlations_by_condition(df, ALIGNMENT_FEATURES, QUALITY_METRICS)

    # Print summary
    for feat in ALIGNMENT_FEATURES:
        for metric in QUALITY_METRICS:
            r = corr_results[feat][metric]
            sig = " ***" if r["significant"] else ""
            print(f"  {feat:35s} vs {metric:20s}: rho={r['rho']:+.4f}, "
                  f"p_corr={r['p_corrected']:.4f}, n={r['n']}{sig}")

    with open(output_dir / "correlations.json", "w") as f:
        json.dump({"pooled": corr_results, "by_condition": corr_by_cond}, f, indent=2)

    # C2: Wilcoxon tests
    print("\n=== C2: Wilcoxon Signed-Rank Tests ===")
    wilcoxon_features = ["mean_alignment_gap", "jsd", "weighted_alignment_gap", "soft_jsd"]
    wilcoxon_results = compute_wilcoxon_tests(df, condition_pairs, wilcoxon_features)

    for pair_key, feats in wilcoxon_results.items():
        print(f"\n  {pair_key}:")
        for feat, r in feats.items():
            sig = " ***" if r.get("significant") else ""
            direction = " (routed lower)" if r.get("routed_lower") else " (routed higher)"
            print(f"    {feat:35s}: p={r['p']:.4f}, "
                  f"mean_routed={r['mean_routed']:.4f}, "
                  f"mean_baseline={r['mean_baseline']:.4f}{direction}{sig}")

    with open(output_dir / "wilcoxon.json", "w") as f:
        json.dump(wilcoxon_results, f, indent=2)

    # C3: Logistic regression
    print("\n=== C3: Logistic Regression ===")
    feature_sets = {
        "alignment_only": ALIGNMENT_FEATURES,
        "jump_only": JUMP_FEATURES,
        "combined": ALIGNMENT_FEATURES + JUMP_FEATURES,
    }
    logistic_results = compute_logistic_regression(df, feature_sets,
                                                    n_folds=n_folds,
                                                    random_state=random_state)

    for model_name, r in logistic_results.items():
        print(f"  {model_name:20s}: AUROC = {r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f} "
              f"(n={r['n_samples']}, {r['n_features']} features)")

    with open(output_dir / "logistic.json", "w") as f:
        json.dump(logistic_results, f, indent=2)

    # C4: Partial correlations
    print("\n=== C4: Partial Correlations ===")
    control_vars = ["n_steps", "answer_type"]
    partial_results = compute_partial_correlations(df, ALIGNMENT_FEATURES,
                                                    QUALITY_METRICS, control_vars)

    for feat in ALIGNMENT_FEATURES:
        for metric in QUALITY_METRICS:
            r = partial_results[feat][metric]
            print(f"  {feat:35s} vs {metric:20s}: partial_rho={r['partial_rho']:+.4f}, n={r['n']}")

    with open(output_dir / "partial_correlations.json", "w") as f:
        json.dump(partial_results, f, indent=2)

    print(f"\nAll statistical results saved to {output_dir}")
    return {
        "correlations": corr_results,
        "wilcoxon": wilcoxon_results,
        "logistic": logistic_results,
        "partial_correlations": partial_results,
    }
