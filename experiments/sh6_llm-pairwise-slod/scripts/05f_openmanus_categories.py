#!/usr/bin/env python
"""SH6 Stage-5f: OpenManus-only diagnostic.

Why OpenManus: it is the only framework in AgentHallu that contains
non-trivial counts in three failure categories simultaneously
(Reasoning / Human-Interaction / Retrieval). Restricting to a single
framework controls for the framework-driven cluster structure that
dominated the global UMAP in 05e and produces an honest test of
"do trajectory features separate failure *types* once we hold the
agent platform fixed?".

Two diagnostics, same input subset:

1. UMAP on the 84 hallucinated OpenManus items, coloured by
   `hallucination_category` (3 classes).
2. Multiclass classification of `hallucination_category` with logreg
   and LightGBM under the same 5-fold StratifiedKFold protocol used in
   Stage-5b. Reports macro one-vs-rest AUC, per-class AUC, balanced
   accuracy, accuracy, and a confusion matrix; bootstraps macro-AUC
   for both models on identical resamples so the difference is paired.

Usage:
    uv run python experiments/sh6_llm-pairwise-slod/scripts/05f_openmanus_categories.py
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    repo = source_root / "AgentHallu"
    dataset_root = repo / "AgentHallu"
    if not dataset_root.is_dir():
        source_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)], check=True)
    return dataset_root


def _load_labels(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for fw_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for jp in sorted(fw_dir.glob("*.json")):
            d = json.loads(jp.read_text(encoding="utf-8"))
            rows.append(
                {
                    "id": f"{fw_dir.name}/{jp.stem}",
                    "framework": fw_dir.name,
                    "hallucination_category": d.get("hallucination_category"),
                    "hallucination_subcategory": d.get("hallucination_subcategory"),
                }
            )
    return pd.DataFrame(rows)


def _build_estimators(random_state: int) -> dict[str, Pipeline]:
    """Three estimators, same CV folds:

    - ``length_only`` — logreg on 3 chunk-count features, structural baseline.
    - ``logreg``      — logreg on trajectory_full (~63 cols).
    - ``lightgbm``    — gradient boosting on trajectory_full.

    Lift of trajectory_full models over length_only isolates the
    contribution of SLoD shape features beyond just counting chunks.
    """
    from lightgbm import LGBMClassifier

    def _logreg_pipe():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000, class_weight="balanced", random_state=random_state,
            )),
        ])

    return {
        "length_only": _logreg_pipe(),
        "logreg": _logreg_pipe(),
        "lightgbm": Pipeline([
            ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
            ("clf", LGBMClassifier(
                n_estimators=300, num_leaves=31, learning_rate=0.05, min_child_samples=10,
                feature_fraction=1.0, bagging_fraction=1.0,
                class_weight="balanced", random_state=random_state, n_jobs=1, verbose=-1,
            )),
        ]),
    }


def _eval_oof(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold) -> dict:
    """Cross-validated OOF probabilities + standard multiclass metrics."""
    probs = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
    preds = probs.argmax(axis=1)
    classes = np.unique(y)

    macro_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=classes))
    per_class_auc = {}
    for i, cls in enumerate(classes):
        per_class_auc[str(cls)] = float(roc_auc_score((y == cls).astype(int), probs[:, i]))

    bal_acc = float(balanced_accuracy_score(y, preds))
    acc = float((preds == y).mean())
    cm = confusion_matrix(y, preds, labels=classes).tolist()
    return {
        "probs": probs,
        "preds": preds,
        "classes": [str(c) for c in classes],
        "macro_auc_ovr": macro_auc,
        "per_class_auc": per_class_auc,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "confusion_matrix": cm,
    }


def _bootstrap_paired_macro_auc(
    y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray,
    n: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Δ macro-AUC with 95% percentile-bootstrap CI; resamples paired by row."""
    classes = np.unique(y)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        y_b = y[idx]
        if len(np.unique(y_b)) < len(classes):
            continue
        try:
            a = roc_auc_score(y_b, probs_a[idx], multi_class="ovr", average="macro", labels=classes)
            b = roc_auc_score(y_b, probs_b[idx], multi_class="ovr", average="macro", labels=classes)
        except ValueError:
            continue
        deltas.append(b - a)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(deltas)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _plot_umap(coords: np.ndarray, labels: pd.Series, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6.5))
    palette = {
        "Reasoning Hallucination": "#1b7837",
        "Human-Interaction Hallucination": "#c2a5cf",
        "Retrieval Hallucination": "#d6604d",
    }
    for cat, color in palette.items():
        m = (labels == cat).to_numpy()
        n = int(m.sum())
        if n == 0:
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=40, alpha=0.7, c=color,
                   edgecolor="white", linewidth=0.4, label=f"{cat} (n={n})")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("OpenManus failures — UMAP coloured by hallucination category")
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion(cm: list[list[int]], classes: list[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    arr = np.asarray(cm, dtype=float)
    norm = arr / arr.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    short = [c.replace(" Hallucination", "") for c in classes]
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(short, rotation=30, ha="right"); ax.set_yticklabels(short)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{int(arr[i, j])}\n({norm[i, j]:.2f})",
                    ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--feature-set", default="trajectory_full")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    args = parser.parse_args()

    setup_logging("INFO")

    df_full = pd.read_csv(args.features_csv.resolve())
    df_full["target"] = df_full["final_answer_correct"].astype(int)
    feature_cols = choose_feature_sets(df_full)[args.feature_set]
    if not feature_cols:
        logger.error("Feature set '%s' is empty", args.feature_set)
        return 1

    labels = _load_labels(_ensure_repo(args.source_root))
    df = df_full.merge(labels, on="id", how="left")

    fw_mask = df["framework"] == "OpenManus"
    fail_mask = df["target"] == 0
    sub = df[fw_mask & fail_mask].reset_index(drop=True)
    logger.info("OpenManus failures: %d items, classes: %s",
                len(sub), sub["hallucination_category"].value_counts().to_dict())

    X = sub[feature_cols].copy()
    y_str = sub["hallucination_category"].astype(str).to_numpy()
    classes = sorted(set(y_str.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in y_str], dtype=int)

    out_dir = args.features_csv.parent / "diagnostics" / "openmanus"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- UMAP ----------
    Xn = SimpleImputer(strategy="median").fit_transform(X.to_numpy(dtype=float))
    Xn = StandardScaler().fit_transform(Xn)

    import umap
    reducer = umap.UMAP(
        n_neighbors=min(args.n_neighbors, len(sub) - 1),
        min_dist=args.min_dist,
        random_state=args.random_state,
        metric="euclidean",
    )
    coords = reducer.fit_transform(Xn)
    pd.DataFrame({
        "id": sub["id"].to_numpy(),
        "umap_x": coords[:, 0], "umap_y": coords[:, 1],
        "category": y_str,
    }).to_csv(out_dir / "openmanus_umap.csv", index=False)
    _plot_umap(coords, sub["hallucination_category"], out_dir / "openmanus_umap.png")

    # ---------- Multiclass classification ----------
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    estimators = _build_estimators(args.random_state)

    LENGTH_COLS = [c for c in ("reasoning_n_chunks", "answer_n_chunks", "total_n_chunks") if c in feature_cols]

    results: dict[str, dict] = {}
    for name, est in estimators.items():
        X_in = sub[LENGTH_COLS] if name == "length_only" else X
        logger.info("Fitting %s (multiclass) on %d items × %d features", name, len(sub), X_in.shape[1])
        results[name] = _eval_oof(est, X_in, y, cv)
        _plot_confusion(
            results[name]["confusion_matrix"],
            classes,
            out_dir / f"openmanus_confusion_{name}.png",
            title=f"OpenManus — {name} confusion (macro-AUC {results[name]['macro_auc_ovr']:.3f})",
        )
        oof_df = pd.DataFrame(results[name]["probs"], columns=[f"p_{c}" for c in classes])
        oof_df.insert(0, "id", sub["id"].to_numpy())
        oof_df["true_class"] = y_str
        oof_df["pred_class"] = [classes[p] for p in results[name]["preds"]]
        oof_df.to_parquet(out_dir / f"openmanus_oof_{name}.parquet", index=False)

    rng = np.random.default_rng(args.random_state)
    delta_lgbm_vs_logreg = _bootstrap_paired_macro_auc(
        y, results["logreg"]["probs"], results["lightgbm"]["probs"],
        args.n_bootstrap, rng,
    )
    delta_logreg_vs_length = _bootstrap_paired_macro_auc(
        y, results["length_only"]["probs"], results["logreg"]["probs"],
        args.n_bootstrap, rng,
    )
    delta_lgbm_vs_length = _bootstrap_paired_macro_auc(
        y, results["length_only"]["probs"], results["lightgbm"]["probs"],
        args.n_bootstrap, rng,
    )

    # Chance baseline for context: macro-AUC = 0.5 by definition.
    chance_macro_auc = 0.5

    summary = {
        "n_items": int(len(sub)),
        "classes": classes,
        "class_counts": {c: int((y_str == c).sum()) for c in classes},
        "feature_set": args.feature_set,
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "chance_macro_auc": chance_macro_auc,
        "models": {
            name: {k: v for k, v in r.items() if k not in ("probs", "preds")}
            for name, r in results.items()
        },
        "delta_macro_auc": {
            "lightgbm_minus_logreg": dict(zip(("mean", "ci_low", "ci_high"), delta_lgbm_vs_logreg)),
            "logreg_minus_length_only": dict(zip(("mean", "ci_low", "ci_high"), delta_logreg_vs_length)),
            "lightgbm_minus_length_only": dict(zip(("mean", "ci_low", "ci_high"), delta_lgbm_vs_length)),
        },
    }
    (out_dir / "openmanus_classification.json").write_text(json.dumps(summary, indent=2))

    # ---------- Markdown ----------
    md = [
        "# SH6 Stage-5f — OpenManus Hallucination-Category Classification",
        "",
        f"**Subset:** OpenManus failures only ({len(sub)} items)",
        f"**Classes:** {', '.join(f'`{c}` (n={(y_str == c).sum()})' for c in classes)}",
        f"**Feature set:** `{args.feature_set}` ({X.shape[1]} cols)",
        f"**CV:** {args.cv_folds}-fold stratified, random_state={args.random_state}",
        f"**Chance baseline (macro one-vs-rest AUC):** {chance_macro_auc:.3f}",
        "",
        "## Summary",
        "",
        "| Model | Macro AUC (OvR) | Bal. Acc | Acc |",
        "|---|---:|---:|---:|",
    ]
    for name, r in results.items():
        md.append(f"| `{name}` | {r['macro_auc_ovr']:.3f} | {r['balanced_accuracy']:.3f} | {r['accuracy']:.3f} |")

    md += ["", "## Per-class AUC (one-vs-rest)", "",
           "| Class | length_only | logreg | lightgbm |", "|---|---:|---:|---:|"]
    for cls in classes:
        i = str(class_to_idx[cls])
        md.append(
            f"| `{cls}` | "
            f"{results['length_only']['per_class_auc'][i]:.3f} | "
            f"{results['logreg']['per_class_auc'][i]:.3f} | "
            f"{results['lightgbm']['per_class_auc'][i]:.3f} |"
        )

    md += ["", "## Δ macro-AUC (paired bootstrap, 95% CI)", "",
           "| Comparison | Δ mean | CI low | CI high | Verdict |",
           "|---|---:|---:|---:|:---|"]
    for label, (m, lo, hi) in [
        ("lightgbm − logreg", delta_lgbm_vs_logreg),
        ("logreg − length_only (shape lift)", delta_logreg_vs_length),
        ("lightgbm − length_only (shape lift)", delta_lgbm_vs_length),
    ]:
        verdict = (
            "significant lift" if (not np.isnan(lo) and lo > 0)
            else "significant regression" if (not np.isnan(hi) and hi < 0)
            else "inconclusive (CI straddles 0)"
        )
        md.append(f"| {label} | {m:+.3f} | {lo:+.3f} | {hi:+.3f} | {verdict} |")

    md += ["", "## Confusion matrices",
           "",
           "![length_only](openmanus_confusion_length_only.png)",
           "",
           "![logreg](openmanus_confusion_logreg.png)",
           "",
           "![lightgbm](openmanus_confusion_lightgbm.png)",
           "",
           "## UMAP",
           "",
           "![umap](openmanus_umap.png)",
           ""]
    (out_dir / "openmanus_classification.md").write_text("\n".join(md))

    logger.info(
        "Macro AUC: length_only=%.3f, logreg=%.3f, lightgbm=%.3f",
        results["length_only"]["macro_auc_ovr"],
        results["logreg"]["macro_auc_ovr"],
        results["lightgbm"]["macro_auc_ovr"],
    )
    logger.info("Δ lgbm − logreg = %+.3f [%+.3f, %+.3f]", *delta_lgbm_vs_logreg)
    logger.info("Δ lgbm − length_only = %+.3f [%+.3f, %+.3f]", *delta_lgbm_vs_length)
    return 0


if __name__ == "__main__":
    sys.exit(main())
