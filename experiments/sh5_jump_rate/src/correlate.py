"""Stage F: Correlation analysis, condition comparisons, regression."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import load_jsonl, resolve_path

logger = logging.getLogger(__name__)

JUMP_METRICS = [
    "jump_count",
    "mean_abs_delta",
    "max_jump",
    "slod_variance",
    "direction_changes",
    "normalized_jump_rate",
]

SCORE_METRICS = ["answer_token_f1", "attribution_f1"]


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_func,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a bivariate statistic."""
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            s = stat_func(x[idx], y[idx])
            if not np.isnan(s):
                boot_stats.append(s)
        except Exception:
            continue

    if not boot_stats:
        return (np.nan, np.nan)

    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (round(float(lower), 6), round(float(upper), 6))


def compute_correlations(
    jump_df: pd.DataFrame,
    score_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Compute Spearman and Pearson correlations between jump metrics and scores.

    Returns dict with correlation results.
    """
    n_boot = config["statistics"]["bootstrap_n"]
    boot_seed = config["statistics"]["bootstrap_seed"]
    alpha = config["statistics"]["alpha"]
    min_steps = config["cot"]["min_steps_filter"]

    # Merge dataframes
    merged = jump_df.merge(score_df, on=["question_id", "condition"], how="inner")
    logger.info(f"Merged dataset: {len(merged)} records")

    # Filter traces with too few steps
    merged_filtered = merged[merged["n_steps"] >= min_steps].copy()
    logger.info(
        f"After filtering (n_steps >= {min_steps}): {len(merged_filtered)} records"
    )

    results = {
        "overall": {},
        "by_condition": {},
        "by_answer_type": {},
        "n_total": len(merged_filtered),
    }

    # Overall correlations
    for jm in JUMP_METRICS:
        for sm in SCORE_METRICS:
            x = merged_filtered[jm].values
            y = merged_filtered[sm].values

            # Remove NaN pairs
            mask = ~(np.isnan(x) | np.isnan(y))
            x, y = x[mask], y[mask]

            if len(x) < 10:
                continue

            # Spearman
            rho_s, p_s = stats.spearmanr(x, y)
            # Pearson
            rho_p, p_p = stats.pearsonr(x, y)
            # Bootstrap CI for Spearman
            ci = _bootstrap_ci(
                x, y,
                lambda a, b: stats.spearmanr(a, b).statistic,
                n_boot=n_boot, alpha=alpha, seed=boot_seed,
            )

            key = f"{jm}_vs_{sm}"
            results["overall"][key] = {
                "spearman_rho": round(float(rho_s), 6),
                "spearman_p": round(float(p_s), 6),
                "pearson_r": round(float(rho_p), 6),
                "pearson_p": round(float(p_p), 6),
                "bootstrap_ci_95": ci,
                "n": int(len(x)),
            }

    # Per-condition correlations
    for condition in config["retrieval"]["conditions"]:
        cond_data = merged_filtered[merged_filtered["condition"] == condition]
        if len(cond_data) < 10:
            continue

        cond_results = {}
        for jm in JUMP_METRICS:
            for sm in SCORE_METRICS:
                x = cond_data[jm].values
                y = cond_data[sm].values
                mask = ~(np.isnan(x) | np.isnan(y))
                x, y = x[mask], y[mask]
                if len(x) < 10:
                    continue

                rho_s, p_s = stats.spearmanr(x, y)
                key = f"{jm}_vs_{sm}"
                cond_results[key] = {
                    "spearman_rho": round(float(rho_s), 6),
                    "spearman_p": round(float(p_s), 6),
                    "n": int(len(x)),
                }

        results["by_condition"][condition] = cond_results

    # Per answer_type correlations
    if "answer_type" in merged_filtered.columns:
        for atype in merged_filtered["answer_type"].unique():
            atype_data = merged_filtered[merged_filtered["answer_type"] == atype]
            if len(atype_data) < 10:
                continue

            atype_results = {}
            for jm in ["normalized_jump_rate"]:
                for sm in SCORE_METRICS:
                    x = atype_data[jm].values
                    y = atype_data[sm].values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    x, y = x[mask], y[mask]
                    if len(x) < 10:
                        continue

                    rho_s, p_s = stats.spearmanr(x, y)
                    ci = _bootstrap_ci(
                        x, y,
                        lambda a, b: stats.spearmanr(a, b).statistic,
                        n_boot=n_boot, alpha=alpha, seed=boot_seed,
                    )
                    key = f"{jm}_vs_{sm}"
                    atype_results[key] = {
                        "spearman_rho": round(float(rho_s), 6),
                        "spearman_p": round(float(p_s), 6),
                        "bootstrap_ci_95": ci,
                        "n": int(len(x)),
                    }

            results["by_answer_type"][atype] = atype_results

    return results


def compare_conditions(
    jump_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Compare jump metrics across routed vs unrouted conditions.

    Uses paired Wilcoxon signed-rank tests (paired by question_id).
    """
    routed = config["retrieval"]["routed_conditions"]
    unrouted = config["retrieval"]["unrouted_conditions"]
    min_steps = config["cot"]["min_steps_filter"]

    # Filter
    filtered = jump_df[jump_df["n_steps"] >= min_steps].copy()

    results = {
        "pairwise_tests": {},
        "condition_summaries": {},
    }

    # Summary stats per condition
    for cond in config["retrieval"]["conditions"]:
        cond_data = filtered[filtered["condition"] == cond]
        if len(cond_data) == 0:
            continue
        summary = {}
        for metric in JUMP_METRICS:
            vals = cond_data[metric].values
            summary[metric] = {
                "mean": round(float(np.mean(vals)), 6),
                "median": round(float(np.median(vals)), 6),
                "std": round(float(np.std(vals)), 6),
                "n": int(len(vals)),
            }
        results["condition_summaries"][cond] = summary

    # Pairwise Wilcoxon tests
    for r_cond in routed:
        for u_cond in unrouted:
            pair_key = f"{r_cond}_vs_{u_cond}"
            pair_results = {}

            r_data = filtered[filtered["condition"] == r_cond].set_index("question_id")
            u_data = filtered[filtered["condition"] == u_cond].set_index("question_id")

            # Find common question_ids
            common = sorted(set(r_data.index) & set(u_data.index))
            if len(common) < 10:
                logger.warning(f"Too few paired samples for {pair_key}: {len(common)}")
                continue

            for metric in JUMP_METRICS:
                r_vals = r_data.loc[common, metric].values
                u_vals = u_data.loc[common, metric].values

                # Wilcoxon signed-rank test
                diffs = r_vals - u_vals
                nonzero = diffs[diffs != 0]

                if len(nonzero) < 10:
                    pair_results[metric] = {
                        "note": "Too few non-zero differences",
                        "n_pairs": int(len(common)),
                    }
                    continue

                try:
                    stat, p_val = stats.wilcoxon(nonzero)
                except Exception as e:
                    logger.warning(f"Wilcoxon failed for {pair_key}/{metric}: {e}")
                    continue

                # Effect size: rank-biserial correlation
                # r = 1 - (2W / n(n+1))  where W is the test statistic
                n_nz = len(nonzero)
                rank_biserial = 1.0 - (2.0 * stat) / (n_nz * (n_nz + 1) / 2)

                pair_results[metric] = {
                    "wilcoxon_stat": round(float(stat), 4),
                    "wilcoxon_p": round(float(p_val), 6),
                    "rank_biserial_r": round(float(rank_biserial), 6),
                    "routed_mean": round(float(np.mean(r_vals)), 6),
                    "unrouted_mean": round(float(np.mean(u_vals)), 6),
                    "mean_diff": round(float(np.mean(diffs)), 6),
                    "n_pairs": int(len(common)),
                }

            results["pairwise_tests"][pair_key] = pair_results

    return results


def run_regression(
    jump_df: pd.DataFrame,
    score_df: pd.DataFrame,
    config: dict,
) -> dict:
    """Multiple regression: answer_token_f1 ~ normalized_jump_rate + condition + answer_type.

    Returns regression results dict.
    """
    from sklearn.linear_model import LinearRegression

    min_steps = config["cot"]["min_steps_filter"]

    merged = jump_df.merge(score_df, on=["question_id", "condition"], how="inner")
    merged = merged[merged["n_steps"] >= min_steps].copy()

    if len(merged) < 20:
        logger.warning("Not enough data for regression")
        return {"error": "insufficient data"}

    # Encode condition as dummy variables
    condition_dummies = pd.get_dummies(merged["condition"], prefix="cond", drop_first=True)
    # Encode answer_type as dummy variables
    answer_dummies = pd.get_dummies(merged["answer_type"], prefix="atype", drop_first=True)

    # Feature matrix
    X = pd.concat(
        [
            merged[["normalized_jump_rate", "n_steps"]],
            condition_dummies,
            answer_dummies,
        ],
        axis=1,
    )
    y = merged["answer_token_f1"].values

    # Fit
    reg = LinearRegression()
    reg.fit(X, y)

    # Results
    from sklearn.metrics import r2_score

    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)

    coefficients = {}
    for name, coef in zip(X.columns, reg.coef_):
        coefficients[name] = round(float(coef), 6)

    results = {
        "r_squared": round(float(r2), 6),
        "intercept": round(float(reg.intercept_), 6),
        "coefficients": coefficients,
        "n_samples": int(len(merged)),
        "features": list(X.columns),
    }

    logger.info(f"Regression R^2: {r2:.4f}")
    logger.info(f"Coefficients: {coefficients}")

    return results


def analyze_all(config: dict) -> tuple[Path, Path]:
    """Orchestrate Stage F: correlations, condition comparisons, regression.

    Returns paths to saved results.
    """
    data_dir = resolve_path(config, config["paths"]["data_dir"])
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    corr_path = results_dir / "correlation_results.json"
    cond_path = results_dir / "condition_comparison.json"

    if corr_path.exists() and cond_path.exists():
        logger.info("Correlation results already exist, skipping.")
        return corr_path, cond_path

    # Load data
    jump_df = pd.DataFrame(load_jsonl(data_dir / "jump_metrics.jsonl"))
    score_df = pd.DataFrame(load_jsonl(data_dir / "answer_scores.jsonl"))

    logger.info(f"Loaded {len(jump_df)} jump records, {len(score_df)} score records")

    # Correlations
    corr_results = compute_correlations(jump_df, score_df, config)

    # Condition comparisons
    cond_results = compare_conditions(jump_df, config)

    # Regression
    reg_results = run_regression(jump_df, score_df, config)
    corr_results["regression"] = reg_results

    # Save
    with open(corr_path, "w") as f:
        json.dump(corr_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved correlation results to {corr_path}")

    with open(cond_path, "w") as f:
        json.dump(cond_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved condition comparison to {cond_path}")

    # Report key findings
    _report_key_findings(corr_results, cond_results, config)

    return corr_path, cond_path


def _report_key_findings(corr_results: dict, cond_results: dict, config: dict) -> None:
    """Log key findings for quick inspection."""
    alpha = config["statistics"]["alpha"]
    min_rho = config["statistics"]["min_correlation_rho"]

    logger.info("\n" + "=" * 60)
    logger.info("KEY FINDINGS")
    logger.info("=" * 60)

    # Primary correlation test
    key = "normalized_jump_rate_vs_answer_token_f1"
    if key in corr_results.get("overall", {}):
        r = corr_results["overall"][key]
        rho = r["spearman_rho"]
        p = r["spearman_p"]
        ci = r.get("bootstrap_ci_95", (None, None))
        passed = rho < min_rho and p < alpha
        logger.info(
            f"\nH1 (correlation): rho={rho:.4f}, p={p:.4e}, CI={ci}"
        )
        logger.info(f"  Criterion: rho < {min_rho} and p < {alpha}")
        logger.info(f"  Result: {'PASSED' if passed else 'NOT PASSED'}")

    # Primary condition difference test
    pair_key = "slod_weighted_vs_chunks_only"
    metric = "normalized_jump_rate"
    if pair_key in cond_results.get("pairwise_tests", {}):
        pt = cond_results["pairwise_tests"][pair_key]
        if metric in pt:
            t = pt[metric]
            p = t["wilcoxon_p"]
            diff = t["mean_diff"]
            passed = diff < 0 and p < alpha
            logger.info(
                f"\nH2 (condition difference): "
                f"slod_weighted mean={t['routed_mean']:.4f}, "
                f"chunks_only mean={t['unrouted_mean']:.4f}, "
                f"diff={diff:.4f}, p={p:.4e}"
            )
            logger.info(f"  Criterion: routed < unrouted and p < {alpha}")
            logger.info(f"  Result: {'PASSED' if passed else 'NOT PASSED'}")

    logger.info("=" * 60)
