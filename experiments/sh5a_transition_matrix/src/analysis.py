"""Stage D: Statistical analysis — correlation, clustering, chi-squared, logistic."""

import json
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .features import get_numeric_feature_names


def load_features(path: str) -> list[dict]:
    """Load features from JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def correlation_analysis(
    features: list[dict],
    feature_names: list[str],
    target_names: list[str],
    bonferroni_n: int = 60,
    alpha: float = 0.05,
) -> list[dict]:
    """Compute Spearman correlations between features and targets.

    Returns list of result dicts sorted by |rho|.
    """
    results = []
    bonferroni_alpha = alpha / bonferroni_n

    for fname in feature_names:
        vals = [f[fname] for f in features]
        # Skip constant features
        if len(set(vals)) <= 1:
            continue

        for tname in target_names:
            targets = [f[tname] for f in features]
            rho, p_val = stats.spearmanr(vals, targets)

            results.append({
                "feature": fname,
                "target": tname,
                "rho": round(float(rho), 6),
                "p_value": float(p_val),
                "p_bonferroni": float(min(p_val * bonferroni_n, 1.0)),
                "significant_bonferroni": bool(p_val < bonferroni_alpha),
                "abs_rho": round(abs(float(rho)), 6),
            })

    results.sort(key=lambda r: -r["abs_rho"])
    return results


def clustering_analysis(
    features: list[dict],
    matrix_key: str = "soft",  # "hard" or "soft"
    k_range: list[int] = None,
    random_state: int = 42,
) -> dict:
    """K-means clustering on flattened transition vectors.

    Returns dict with cluster assignments and evaluation metrics.
    """
    if k_range is None:
        k_range = [2, 3, 4]

    # Build feature matrix from 9 transition cells
    cell_names = [f"{matrix_key}_{cn}" for cn in [
        "macro->macro", "macro->meso", "macro->micro",
        "meso->macro", "meso->meso", "meso->micro",
        "micro->macro", "micro->meso", "micro->micro",
    ]]

    X = np.array([[f[cn] for cn in cell_names] for f in features])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try different k values
    best_k = None
    best_silhouette = -1
    all_results = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)

        # Compute per-cluster quality
        cluster_quality = {}
        for ci in range(k):
            mask = labels == ci
            cluster_features = [f for f, m in zip(features, mask) if m]
            token_f1_key = "answer_token_f1" if "answer_token_f1" in features[0] else "mean_token_f1"
            attr_f1_key = "attribution_f1" if "attribution_f1" in features[0] else "mean_attribution_f1"

            token_f1s = [f[token_f1_key] for f in cluster_features]
            attr_f1s = [f[attr_f1_key] for f in cluster_features]

            cluster_quality[str(ci)] = {
                "size": int(mask.sum()),
                "mean_token_f1": round(float(np.mean(token_f1s)), 4),
                "std_token_f1": round(float(np.std(token_f1s)), 4),
                "mean_attribution_f1": round(float(np.mean(attr_f1s)), 4),
                "std_attribution_f1": round(float(np.std(attr_f1s)), 4),
                "center": [round(float(v), 4) for v in km.cluster_centers_[ci]],
            }

        # ANOVA on token-F1 across clusters
        groups_token = [
            [f[token_f1_key] for f, m in zip(features, labels == ci) if m]
            for ci in range(k)
        ]
        groups_attr = [
            [f[attr_f1_key] for f, m in zip(features, labels == ci) if m]
            for ci in range(k)
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_token, p_token = stats.f_oneway(*groups_token)
            f_attr, p_attr = stats.f_oneway(*groups_attr)

        all_results[str(k)] = {
            "silhouette": round(float(sil), 4),
            "cluster_quality": cluster_quality,
            "anova_token_f1": {"F": round(float(f_token), 4), "p": float(p_token)},
            "anova_attribution_f1": {"F": round(float(f_attr), 4), "p": float(p_attr)},
        }

        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    return {
        "best_k": best_k,
        "best_silhouette": round(best_silhouette, 4),
        "results_by_k": all_results,
        "matrix_type": matrix_key,
    }


def condition_comparison(features: list[dict], matrix_key: str = "soft") -> dict:
    """Chi-squared test comparing transition distributions across conditions.

    Uses actual transition counts (un-normalized) summed across traces per condition.
    Since we have normalized matrices, we multiply by n_transitions to recover counts.
    """
    cell_names = [f"{matrix_key}_{cn}" for cn in [
        "macro->macro", "macro->meso", "macro->micro",
        "meso->macro", "meso->meso", "meso->micro",
        "micro->macro", "micro->meso", "micro->micro",
    ]]

    conditions = ["chunks_only", "naive_hybrid", "slod_weighted", "slod_weighted_parent"]

    # Build contingency table: conditions x 9 cells
    # Use sum of (matrix * n_transitions) to approximate counts
    contingency = np.zeros((len(conditions), 9), dtype=np.float64)

    for f in features:
        if "condition" not in f:
            continue
        ci = conditions.index(f["condition"]) if f["condition"] in conditions else -1
        if ci < 0:
            continue
        n_trans = f.get("n_transitions", 1)
        for j, cn in enumerate(cell_names):
            contingency[ci, j] += f[cn] * n_trans

    # Ensure no negative values and add small epsilon to avoid zero rows/cols
    contingency = np.maximum(contingency, 0)

    # Chi-squared test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Per-condition aggregated matrices (normalized)
    per_condition = {}
    for ci, cond in enumerate(conditions):
        row = contingency[ci]
        total = row.sum()
        if total > 0:
            row_norm = row / total
        else:
            row_norm = row
        per_condition[cond] = {
            "matrix_flat": [round(float(v), 4) for v in row_norm],
            "total_transitions": round(float(total), 1),
        }

    # Routed vs unrouted comparison
    routed_rows = contingency[2:4].sum(axis=0)  # slod_weighted + slod_weighted_parent
    unrouted_rows = contingency[0:2].sum(axis=0)  # chunks_only + naive_hybrid
    cont_2x9 = np.array([routed_rows, unrouted_rows])
    chi2_2, p_2, dof_2, _ = stats.chi2_contingency(cont_2x9)

    return {
        "chi2_4conditions": {"chi2": round(float(chi2), 4), "p": float(p_value), "dof": int(dof)},
        "chi2_routed_vs_unrouted": {"chi2": round(float(chi2_2), 4), "p": float(p_2), "dof": int(dof_2)},
        "per_condition": per_condition,
        "matrix_type": matrix_key,
    }


def logistic_regression_analysis(
    features: list[dict],
    feature_names: list[str],
    target_name: str = "answer_token_f1",
    random_state: int = 42,
    cv_folds: int = 5,
) -> dict:
    """Logistic regression predicting above-median quality from transition features."""
    # Build arrays
    X = np.array([[f[fn] for fn in feature_names] for f in features])
    y_cont = np.array([f[target_name] for f in features])
    median_val = np.median(y_cont)
    y = (y_cont > median_val).astype(int)

    # Handle NaN/inf
    mask = np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 20:
        return {"error": "Too few valid samples", "n_samples": len(X)}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit model
    lr = LogisticRegression(C=1.0, penalty="l2", random_state=random_state, max_iter=1000)

    # Cross-validated accuracy
    cv_acc = cross_val_score(lr, X_scaled, y, cv=cv_folds, scoring="accuracy")
    cv_auc = cross_val_score(lr, X_scaled, y, cv=cv_folds, scoring="roc_auc")

    # Fit on full data for coefficients
    lr.fit(X_scaled, y)
    coefs = lr.coef_[0]

    # Top features by absolute coefficient
    top_idx = np.argsort(-np.abs(coefs))[:10]
    top_features = [
        {"feature": feature_names[i], "coefficient": round(float(coefs[i]), 4)}
        for i in top_idx
    ]

    return {
        "target": target_name,
        "median_threshold": round(float(median_val), 4),
        "n_samples": int(len(X)),
        "cv_accuracy_mean": round(float(np.mean(cv_acc)), 4),
        "cv_accuracy_std": round(float(np.std(cv_acc)), 4),
        "cv_auc_mean": round(float(np.mean(cv_auc)), 4),
        "cv_auc_std": round(float(np.std(cv_auc)), 4),
        "top_features": top_features,
    }


def run_full_analysis(
    features_path: str = "data/features.jsonl",
    features_agg_path: str = "data/features_agg.jsonl",
    results_dir: str = "data/results",
) -> dict:
    """Run all analyses and save results."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    features = load_features(features_path)
    features_agg = load_features(features_agg_path)

    # Feature names
    hard_names = get_numeric_feature_names("hard_")
    soft_names = get_numeric_feature_names("soft_")
    all_feature_names = hard_names + soft_names

    n_comparisons = len(all_feature_names) * 2  # 2 targets
    print(f"Total features: {len(all_feature_names)}, comparisons: {n_comparisons}")

    # D1: Correlation
    print("\n=== D1: Correlation Analysis (per-trace, N=2000) ===")
    corr_results = correlation_analysis(
        features, all_feature_names,
        ["answer_token_f1", "attribution_f1"],
        bonferroni_n=n_comparisons,
    )
    sig_count = sum(1 for r in corr_results if r["significant_bonferroni"])
    print(f"  Significant after Bonferroni: {sig_count}/{len(corr_results)}")
    top5 = corr_results[:5]
    for r in top5:
        sig = "*" if r["significant_bonferroni"] else ""
        print(f"  {r['feature']:40s} vs {r['target']:20s}: rho={r['rho']:+.4f} p={r['p_value']:.2e} {sig}")

    # Also on aggregated
    print("\n=== D1b: Correlation Analysis (per-question, N=500) ===")
    corr_agg = correlation_analysis(
        features_agg, all_feature_names,
        ["mean_token_f1", "mean_attribution_f1"],
        bonferroni_n=n_comparisons,
    )
    sig_agg = sum(1 for r in corr_agg if r["significant_bonferroni"])
    print(f"  Significant after Bonferroni: {sig_agg}/{len(corr_agg)}")

    with open(out / "correlation_results.json", "w") as f:
        json.dump({
            "per_trace": corr_results,
            "per_question": corr_agg,
            "n_comparisons": n_comparisons,
            "bonferroni_alpha": 0.05 / n_comparisons,
        }, f, indent=2)
    print(f"  Saved to {out / 'correlation_results.json'}")

    # D2: Clustering
    print("\n=== D2: Clustering Analysis ===")
    clust_soft = clustering_analysis(features, matrix_key="soft")
    print(f"  Best k (soft): {clust_soft['best_k']}, silhouette: {clust_soft['best_silhouette']}")
    best_k_str = str(clust_soft["best_k"])
    res = clust_soft["results_by_k"][best_k_str]
    print(f"  ANOVA token-F1: F={res['anova_token_f1']['F']:.4f}, p={res['anova_token_f1']['p']:.4e}")
    print(f"  ANOVA attr-F1:  F={res['anova_attribution_f1']['F']:.4f}, p={res['anova_attribution_f1']['p']:.4e}")

    clust_hard = clustering_analysis(features, matrix_key="hard")
    print(f"  Best k (hard): {clust_hard['best_k']}, silhouette: {clust_hard['best_silhouette']}")

    # Also on aggregated
    clust_agg = clustering_analysis(features_agg, matrix_key="soft")

    with open(out / "clustering_results.json", "w") as f:
        json.dump({
            "per_trace_soft": clust_soft,
            "per_trace_hard": clust_hard,
            "per_question_soft": clust_agg,
        }, f, indent=2)
    print(f"  Saved to {out / 'clustering_results.json'}")

    # D3: Condition comparison
    print("\n=== D3: Condition Comparison ===")
    cond_soft = condition_comparison(features, matrix_key="soft")
    cond_hard = condition_comparison(features, matrix_key="hard")
    print(f"  Chi2 (4 cond, soft): chi2={cond_soft['chi2_4conditions']['chi2']:.2f}, p={cond_soft['chi2_4conditions']['p']:.4e}")
    print(f"  Chi2 (routed vs unrouted, soft): chi2={cond_soft['chi2_routed_vs_unrouted']['chi2']:.2f}, p={cond_soft['chi2_routed_vs_unrouted']['p']:.4e}")

    with open(out / "condition_comparison.json", "w") as f:
        json.dump({"soft": cond_soft, "hard": cond_hard}, f, indent=2)
    print(f"  Saved to {out / 'condition_comparison.json'}")

    # D4: Logistic regression
    print("\n=== D4: Logistic Regression ===")
    log_token = logistic_regression_analysis(features, all_feature_names, "answer_token_f1")
    log_attr = logistic_regression_analysis(features, all_feature_names, "attribution_f1")
    print(f"  Token-F1: CV accuracy={log_token.get('cv_accuracy_mean', 'N/A')}, AUC={log_token.get('cv_auc_mean', 'N/A')}")
    print(f"  Attr-F1:  CV accuracy={log_attr.get('cv_accuracy_mean', 'N/A')}, AUC={log_attr.get('cv_auc_mean', 'N/A')}")

    with open(out / "logistic_results.json", "w") as f:
        json.dump({"token_f1": log_token, "attribution_f1": log_attr}, f, indent=2)

    # D5: Summary
    print("\n=== D5: Summary ===")
    h1_pass = sig_count > 0
    best_rho = corr_results[0] if corr_results else None

    best_k_soft = clust_soft["best_k"]
    h2_token_p = clust_soft["results_by_k"][str(best_k_soft)]["anova_token_f1"]["p"]
    h2_attr_p = clust_soft["results_by_k"][str(best_k_soft)]["anova_attribution_f1"]["p"]
    h2_pass = h2_token_p < 0.05 or h2_attr_p < 0.05

    h3_pass = cond_soft["chi2_4conditions"]["p"] < 0.05

    if h1_pass or h2_pass:
        overall = "CONFIRMED"
    elif h3_pass:
        overall = "PARTIAL"
    else:
        overall = "NOT CONFIRMED"

    summary = {
        "H1_transition_signature": {
            "pass": h1_pass,
            "n_significant": sig_count,
            "best_feature": best_rho["feature"] if best_rho else None,
            "best_rho": best_rho["rho"] if best_rho else None,
            "best_target": best_rho["target"] if best_rho else None,
            "best_p_bonferroni": best_rho["p_bonferroni"] if best_rho else None,
        },
        "H2_pattern_clusters": {
            "pass": h2_pass,
            "best_k": best_k_soft,
            "silhouette": clust_soft["best_silhouette"],
            "anova_token_f1_p": h2_token_p,
            "anova_attribution_f1_p": h2_attr_p,
        },
        "H3_condition_effect": {
            "pass": h3_pass,
            "chi2": cond_soft["chi2_4conditions"]["chi2"],
            "p": cond_soft["chi2_4conditions"]["p"],
            "routed_vs_unrouted_p": cond_soft["chi2_routed_vs_unrouted"]["p"],
        },
        "overall_verdict": overall,
        "comparison_with_sh5": {
            "sh5_jump_rate_token_f1_rho": 0.003,
            "sh5_jump_rate_attribution_f1_rho": 0.092,
            "sh5a_best_token_f1_rho": None,
            "sh5a_best_attribution_f1_rho": None,
        },
    }

    # Fill in best rho by target
    for r in corr_results:
        if r["target"] == "answer_token_f1" and summary["comparison_with_sh5"]["sh5a_best_token_f1_rho"] is None:
            summary["comparison_with_sh5"]["sh5a_best_token_f1_rho"] = r["rho"]
        if r["target"] == "attribution_f1" and summary["comparison_with_sh5"]["sh5a_best_attribution_f1_rho"] is None:
            summary["comparison_with_sh5"]["sh5a_best_attribution_f1_rho"] = r["rho"]

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  H1 (Transition Signature): {'PASS' if h1_pass else 'FAIL'}")
    print(f"  H2 (Pattern Clusters): {'PASS' if h2_pass else 'FAIL'}")
    print(f"  H3 (Condition Effect): {'PASS' if h3_pass else 'FAIL'}")
    print(f"  Overall: {overall}")

    return summary
