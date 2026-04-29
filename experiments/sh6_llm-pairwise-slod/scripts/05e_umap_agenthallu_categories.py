#!/usr/bin/env python
"""SH6 Stage-5e: UMAP overlay with AgentHallu hallucination categories.

The general 05d_umap_diagnostic colours points by target / subject. This
script extends that view for AgentHallu only: the upstream repo carries
two hallucination labels per item that are not propagated into
trajectory_features.csv:

    hallucination_category     coarse type (e.g. "Tool-Use Hallucination")
    hallucination_subcategory  fine subtype

We re-read the source JSON files keyed by ``id = framework/json_stem``
and produce a 2x2 figure: target / category / subcategory / subject.

If the local AgentHallu clone is missing the script clones it lazily
into the path provided by --source-root (default /tmp/agenthallu-src).

Usage:
    uv run python experiments/sh6_llm-pairwise-slod/scripts/05e_umap_agenthallu_categories.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from semanticscale.sh6.failure_analysis import choose_feature_sets
from semanticscale.utils import setup_logging

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/liuxuannan/AgentHallu.git"
DEFAULT_FEATURES = (
    Path(__file__).resolve().parents[1]
    / "reports/agenthallu/framework-all/trajectory_features.csv"
)
DEFAULT_SOURCE_ROOT = Path("/tmp/agenthallu-src")


def _ensure_repo(source_root: Path) -> Path:
    """Clone the AgentHallu repo on demand and return the path to the dataset root."""
    repo = source_root / "AgentHallu"
    dataset_root = repo / "AgentHallu"
    if not dataset_root.is_dir():
        source_root.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning AgentHallu into %s ...", repo)
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(repo)],
            check=True,
        )
    return dataset_root


def _load_hallu_labels(dataset_root: Path) -> pd.DataFrame:
    """Walk the dataset directory and pull the per-item label fields we need."""
    rows = []
    for framework_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for jp in sorted(framework_dir.glob("*.json")):
            d = json.loads(jp.read_text(encoding="utf-8"))
            rows.append(
                {
                    "id": f"{framework_dir.name}/{jp.stem}",
                    "framework": framework_dir.name,
                    "is_hallucination": str(d.get("is_hallucination", "")).strip().lower() == "true",
                    "hallucination_category": d.get("hallucination_category"),
                    "hallucination_subcategory": d.get("hallucination_subcategory"),
                    "model_id": d.get("model_id"),
                    "agent_type": d.get("agent_type"),
                }
            )
    return pd.DataFrame(rows)


def _shorten(label: str | float | None, max_len: int = 28) -> str:
    """Trim long labels for legend readability."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return "N/A (correct)"
    s = str(label)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _scatter_by_category(
    ax, coords: np.ndarray, labels: pd.Series, title: str, top_k: int = 9
) -> None:
    """Plot UMAP coloured by a categorical label, capping the legend to top_k.

    AgentHallu has long-tailed subcategory taxonomies; without a cap the
    legend swallows half the figure. The ``top_k - 1`` most populous
    classes are coloured individually, the rest collapse to "other".
    """
    import matplotlib.pyplot as plt

    counts = labels.value_counts(dropna=False)
    keep = list(counts.head(top_k - 1).index)
    plot_label = labels.where(labels.isin(keep), other="other")

    unique = list(counts.head(top_k - 1).index) + (["other"] if (~labels.isin(keep)).any() else [])
    cmap = plt.colormaps["tab10"].resampled(max(len(unique), 1))

    for i, cat in enumerate(unique):
        mask = (plot_label == cat).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=14, alpha=0.6, c=[cmap(i)], edgecolor="none",
            label=f"{_shorten(cat)} (n={n})",
        )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=7, framealpha=0.85)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--feature-set", default="trajectory_full")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=9, help="Top-K categories shown individually; rest go to 'other'.")
    args = parser.parse_args()

    setup_logging("INFO")

    features_csv = args.features_csv.resolve()
    df = pd.read_csv(features_csv)
    if "id" not in df.columns:
        logger.error("trajectory_features.csv has no `id` column; cannot join labels.")
        return 1

    dataset_root = _ensure_repo(args.source_root)
    labels = _load_hallu_labels(dataset_root)
    logger.info("Loaded %d label rows from %s", len(labels), dataset_root)

    # Pick feature columns *before* the merge so the joined label fields
    # (`framework`, `model_id`, `agent_type`, …) don't leak into the
    # numeric matrix.
    df["target"] = df["final_answer_correct"].astype(int)
    feature_sets = choose_feature_sets(df)
    feature_cols = feature_sets[args.feature_set]
    if not feature_cols:
        logger.error("Feature set '%s' is empty", args.feature_set)
        return 1

    df = df.merge(labels, on="id", how="left", suffixes=("", "_repo"))
    n_unmatched = int(df["framework"].isna().sum())
    if n_unmatched:
        logger.warning("%d items in trajectory_features.csv had no match in repo", n_unmatched)

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X = df[feature_cols].to_numpy(dtype=float)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    y = df["target"].to_numpy()

    logger.info("Running UMAP on %d items × %d features", len(df), X.shape[1])
    import umap
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        metric="euclidean",
    )
    coords = reducer.fit_transform(X)

    out_dir = features_csv.parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_df = pd.DataFrame(
        {
            "id": df["id"],
            "umap_x": coords[:, 0],
            "umap_y": coords[:, 1],
            "target": y,
            "framework": df["framework"],
            "hallucination_category": df["hallucination_category"],
            "hallucination_subcategory": df["hallucination_subcategory"],
            "subject": df["subject"],
        }
    )
    coords_csv = out_dir / "umap_agenthallu_categories.csv"
    coords_df.to_csv(coords_csv, index=False)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) target
    ax = axes[0, 0]
    for label, color, name in [(0, "#d6604d", "wrong (hallucinated)"), (1, "#4393c3", "correct")]:
        mask = y == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, alpha=0.55,
                   c=color, edgecolor="none", label=f"{name} (n={int(mask.sum())})")
    ax.set_title("Target (hallucination vs correct)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.legend(loc="best", fontsize=9)

    # (b) hallucination_category — only on hallucinated items
    ax = axes[0, 1]
    cat = df["hallucination_category"].where(df["target"] == 0, other=np.nan)
    _scatter_by_category(ax, coords, cat, "Hallucination category (correct items hidden)", top_k=args.top_k)

    # (c) hallucination_subcategory — only on hallucinated items
    ax = axes[1, 0]
    sub = df["hallucination_subcategory"].where(df["target"] == 0, other=np.nan)
    _scatter_by_category(ax, coords, sub, "Hallucination subcategory (correct items hidden)", top_k=args.top_k)

    # (d) framework — sanity check, this is the cluster-driver from 05d
    ax = axes[1, 1]
    _scatter_by_category(ax, coords, df["framework"], "Framework (cluster sanity check)", top_k=args.top_k)

    n_total = len(df)
    n_hallu = int((y == 0).sum())
    cat_counts = df.loc[y == 0, "hallucination_category"].value_counts(dropna=False)
    fig.suptitle(
        f"AgentHallu — UMAP overlay  (n={n_total}, hallucinated={n_hallu})\n"
        f"top categories: {', '.join(f'{k} ({v})' for k, v in cat_counts.head(4).items())}",
        fontsize=11,
    )
    fig.tight_layout()
    plot_path = out_dir / "umap_agenthallu_categories.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary = {
        "n_items": int(n_total),
        "n_hallucinated": n_hallu,
        "n_unmatched_to_repo": n_unmatched,
        "category_counts": cat_counts.to_dict(),
    }
    (out_dir / "umap_agenthallu_categories.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote %s, %s", plot_path, coords_csv)
    logger.info("Category counts (hallucinated only):\n%s", cat_counts.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
