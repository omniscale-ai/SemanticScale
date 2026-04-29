#!/usr/bin/env python
"""SH6 Stage-5d: UMAP diagnostic on Stage-5 trajectory features.

Diagnostic tool for runs where both logreg and LightGBM hover near
chance (e.g. agenthallu, processbench-gsm8k/gsm8k). The question this answers:
"in the trajectory feature space, do correct and wrong items live in
different regions at all?"

A clean separation in 2D would mean classifiers should work and the
issue is in the model. Overlapping clouds mean the features genuinely
don't separate classes — no classifier or feature-engineering pass on
the *same axis* will help, and the gap is in the SLoD axis itself.

Usage:
    uv run python experiments/sh6_llm-pairwise-slod/scripts/05d_umap_diagnostic.py \\
        --features-csv experiments/sh6_llm-pairwise-slod/reports/agenthallu/framework-all/trajectory_features.csv

Outputs (next to the input CSV):
    diagnostics/umap_<feature_set>.png
    diagnostics/umap_<feature_set>.csv     # 2D coords + target + meta
    diagnostics/umap_summary.json          # neighbor-purity diagnostic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from semanticscale.sh6.failure_analysis import META_COLUMNS, choose_feature_sets
from semanticscale.utils import setup_logging

logger = logging.getLogger(__name__)


def _resolve_target(df: pd.DataFrame, target_label: str) -> pd.Series:
    if target_label == "auto":
        for cand in ("final_answer_correct", "is_correct"):
            if cand in df.columns and df[cand].notna().any():
                target_label = cand
                break
        else:
            msg = "No usable target column."
            raise ValueError(msg)
    return df[target_label].astype(float)


def _neighbor_purity(coords: np.ndarray, y: np.ndarray, k: int = 15) -> float:
    """Fraction of each item's k-nearest neighbors that share its class.

    A pure 50/50 mix gives ~base-rate purity (no information). Strong
    separation pushes purity toward 1.0. This is a non-classifier
    diagnostic of "are nearby points the same class?" and is a useful
    sanity check on UMAP plots.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nn.kneighbors(coords)
    # idx[:, 0] is the point itself; drop it.
    neighbor_labels = y[idx[:, 1:]]
    same = (neighbor_labels == y[:, None]).mean(axis=1)
    return float(same.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--target-label", default="auto")
    parser.add_argument("--feature-set", default="trajectory_full",
                        choices=("length_only", "trajectory_shape", "trajectory_full"))
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-subdir", default="diagnostics")
    args = parser.parse_args()

    setup_logging("INFO")

    features_csv = args.features_csv.resolve()
    if not features_csv.exists():
        logger.error("Features CSV not found: %s", features_csv)
        return 1

    df = pd.read_csv(features_csv)
    df["target"] = _resolve_target(df, args.target_label).astype(float)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)

    feature_sets = choose_feature_sets(df)
    feature_cols = feature_sets[args.feature_set]
    if not feature_cols:
        logger.error("Feature set '%s' is empty for this run", args.feature_set)
        return 1

    X = df[feature_cols].to_numpy(dtype=float)
    # Median impute + standardize so UMAP's distance is on a comparable scale
    # across features. Same preprocessing as the logreg pipeline so the 2D
    # picture reflects what the classifier actually sees.
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    y = df["target"].to_numpy()

    logger.info("Running UMAP on %d items × %d features (set=%s)",
                len(df), X.shape[1], args.feature_set)

    import umap
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        metric="euclidean",
    )
    coords = reducer.fit_transform(X)

    out_dir = features_csv.parent / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"umap_{args.feature_set}"

    # Save per-point coords + meta for downstream slicing.
    keep_meta = [c for c in ("id", "subject", "exit_status", "final_answer_correct", "is_correct")
                 if c in df.columns]
    coords_df = pd.DataFrame({"umap_x": coords[:, 0], "umap_y": coords[:, 1], "target": y})
    coords_df = pd.concat([coords_df, df[keep_meta].reset_index(drop=True)], axis=1)
    coords_csv = out_dir / f"{stem}.csv"
    coords_df.to_csv(coords_csv, index=False)

    purity = _neighbor_purity(coords, y, k=args.n_neighbors)
    base_rate_purity = float(max((y == 1).mean(), (y == 0).mean()))
    logger.info("k=%d neighbor purity: %.3f (base-rate purity %.3f → lift %.3f)",
                args.n_neighbors, purity, base_rate_purity, purity - base_rate_purity)

    # Plot.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    # (a) coloured by target
    ax = axes[0]
    for label, color, name in [(0, "#d6604d", "wrong"), (1, "#4393c3", "correct")]:
        mask = y == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.55,
                   c=color, label=f"{name} (n={int(mask.sum())})", edgecolor="none")
    ax.set_title(f"UMAP — coloured by target\nrun: {features_csv.parent.name}")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=9)

    # (b) optional second pane: by subject if it varies
    ax = axes[1]
    if "subject" in df.columns and df["subject"].nunique() > 1:
        subjects = df["subject"].fillna("?").astype(str)
        unique = sorted(subjects.unique())
        cmap = plt.colormaps["tab10"].resampled(len(unique))
        for i, subj in enumerate(unique):
            mask = subjects == subj
            ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.55,
                       c=[cmap(i)], label=f"{subj} (n={int(mask.sum())})", edgecolor="none")
        ax.legend(loc="best", fontsize=8)
        ax.set_title("UMAP — coloured by subject")
    else:
        # fallback: density histogram by target
        ax.hist([coords[y == 0, 0], coords[y == 1, 0]], bins=30,
                stacked=False, label=["wrong", "correct"], color=["#d6604d", "#4393c3"], alpha=0.6)
        ax.set_title("UMAP-1 marginal by target")
        ax.legend(fontsize=9)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")

    fig.suptitle(
        f"k-NN purity (k={args.n_neighbors}): {purity:.3f}   "
        f"base-rate ceiling: {base_rate_purity:.3f}   "
        f"lift: {purity - base_rate_purity:+.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    plot_path = out_dir / f"{stem}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary = {
        "features_csv": str(features_csv),
        "n_items": int(len(df)),
        "n_features_used": int(X.shape[1]),
        "feature_set": args.feature_set,
        "target_pos_rate": float((y == 1).mean()),
        "umap_neighbors": args.n_neighbors,
        "umap_min_dist": args.min_dist,
        "knn_purity": purity,
        "base_rate_purity": base_rate_purity,
        "knn_purity_lift": purity - base_rate_purity,
    }
    summary_path = out_dir / "umap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s, %s, %s", plot_path, coords_csv, summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
