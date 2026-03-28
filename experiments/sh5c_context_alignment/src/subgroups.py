"""Stage D: Subgroup Analysis.

Analyzes alignment-quality relationships by answer type, context diversity,
and retrieval condition.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


ALIGNMENT_FEATURES = [
    "mean_alignment_gap", "jsd", "weighted_alignment_gap",
    "dominant_level_match", "soft_jsd",
]

QUALITY_METRICS = ["answer_token_f1", "attribution_f1"]


def analyze_by_answer_type(df: pd.DataFrame) -> dict:
    """Compute alignment-quality correlations per answer type."""
    results = {}
    for atype in df["answer_type"].unique():
        sub = df[df["answer_type"] == atype]
        results[atype] = {
            "n": len(sub),
            "correlations": {},
            "means": {},
        }
        for feat in ALIGNMENT_FEATURES:
            results[atype]["correlations"][feat] = {}
            for metric in QUALITY_METRICS:
                valid = sub[[feat, metric]].dropna()
                if len(valid) < 10:
                    results[atype]["correlations"][feat][metric] = {
                        "rho": float("nan"), "p": float("nan"), "n": len(valid)
                    }
                else:
                    rho, p = stats.spearmanr(valid[feat], valid[metric])
                    results[atype]["correlations"][feat][metric] = {
                        "rho": float(rho), "p": float(p), "n": len(valid)
                    }
            results[atype]["means"][feat] = float(sub[feat].dropna().mean()) if sub[feat].notna().any() else float("nan")

    return results


def analyze_by_context_diversity(df: pd.DataFrame) -> dict:
    """Compare alignment-quality correlation in high vs low diversity groups.

    Excludes chunks_only (always zero diversity) from the split.
    """
    # Exclude chunks_only for diversity split
    df_multi = df[df["condition"] != "chunks_only"].copy()

    if df_multi["context_diversity"].dropna().empty:
        return {"error": "No valid context_diversity values"}

    median_div = df_multi["context_diversity"].median()

    results = {"median_context_diversity": float(median_div), "groups": {}}

    for label, sub in [
        ("low_diversity", df_multi[df_multi["context_diversity"] <= median_div]),
        ("high_diversity", df_multi[df_multi["context_diversity"] > median_div]),
    ]:
        results["groups"][label] = {
            "n": len(sub),
            "mean_context_diversity": float(sub["context_diversity"].mean()),
            "correlations": {},
        }
        for feat in ALIGNMENT_FEATURES:
            results["groups"][label]["correlations"][feat] = {}
            for metric in QUALITY_METRICS:
                valid = sub[[feat, metric]].dropna()
                if len(valid) < 10:
                    results["groups"][label]["correlations"][feat][metric] = {
                        "rho": float("nan"), "p": float("nan"), "n": len(valid)
                    }
                else:
                    rho, p = stats.spearmanr(valid[feat], valid[metric])
                    results["groups"][label]["correlations"][feat][metric] = {
                        "rho": float(rho), "p": float(p), "n": len(valid)
                    }

    return results


def analyze_by_condition(df: pd.DataFrame) -> dict:
    """Compute mean alignment features per condition."""
    results = {}
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        results[cond] = {
            "n": len(sub),
            "means": {},
            "medians": {},
        }
        for feat in ALIGNMENT_FEATURES + ["context_diversity", "reasoning_diversity"]:
            valid = sub[feat].dropna()
            results[cond]["means"][feat] = float(valid.mean()) if len(valid) > 0 else float("nan")
            results[cond]["medians"][feat] = float(valid.median()) if len(valid) > 0 else float("nan")

        # Also quality metrics
        for metric in QUALITY_METRICS:
            results[cond]["means"][metric] = float(sub[metric].mean())
            results[cond]["medians"][metric] = float(sub[metric].median())

    return results


def run_subgroup_analysis(alignment_path: Path, output_dir: Path):
    """Run all subgroup analyses and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(alignment_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)

    print(f"Loaded {len(df)} traces for subgroup analysis")

    # D1: By answer type
    print("\n=== D1: By Answer Type ===")
    by_type = analyze_by_answer_type(df)
    for atype, r in by_type.items():
        print(f"\n  {atype} (n={r['n']}):")
        for feat in ALIGNMENT_FEATURES[:3]:
            for metric in QUALITY_METRICS:
                c = r["correlations"][feat][metric]
                print(f"    {feat} vs {metric}: rho={c['rho']:+.4f}, p={c['p']:.4f}")

    # D2: By context diversity
    print("\n=== D2: By Context Diversity ===")
    by_diversity = analyze_by_context_diversity(df)
    if "groups" in by_diversity:
        for label, r in by_diversity["groups"].items():
            print(f"\n  {label} (n={r['n']}):")
            for feat in ALIGNMENT_FEATURES[:3]:
                for metric in QUALITY_METRICS[:1]:
                    c = r["correlations"][feat][metric]
                    print(f"    {feat} vs {metric}: rho={c['rho']:+.4f}")

    # D3: By condition
    print("\n=== D3: By Condition ===")
    by_condition = analyze_by_condition(df)
    for cond, r in by_condition.items():
        print(f"  {cond:30s}: mean_alignment_gap={r['means'].get('mean_alignment_gap', float('nan')):.4f}, "
              f"mean_jsd={r['means'].get('jsd', float('nan')):.4f}, "
              f"mean_token_f1={r['means'].get('answer_token_f1', float('nan')):.4f}")

    # Save
    all_results = {
        "by_answer_type": by_type,
        "by_context_diversity": by_diversity,
        "by_condition": by_condition,
    }
    with open(output_dir / "subgroups.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else x)

    print(f"\nSubgroup results saved to {output_dir / 'subgroups.json'}")
    return all_results
