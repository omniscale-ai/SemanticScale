#!/usr/bin/env python
"""SH6 Stage-5g: Full-AgentHallu multiclass hallucination-category classification.

Generalisation of 05f to the full AgentHallu failure set (n=443 across 5
top-level categories), with two important additions:

1. **Framework-only baseline.** A logistic regression on a one-hot
   encoding of `framework` alone (no trajectory features). This tells
   us how much of any "category prediction" signal is just framework
   leakage. The intent is to make the comparison falsifiable: if
   trajectory-feature models barely beat framework-only, the answer
   to "do features encode failure type" is "no, they encode platform".
2. **Per-(framework, category) prevalence diagnostic** in the report,
   to make the leakage situation legible.

The script writes everything under
`reports/agenthallu/framework-all/diagnostics/all/`.

Usage:
    uv run python experiments/sh6_llm-pairwise-slod/scripts/05g_agenthallu_categories.py
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
                }
            )
    return pd.DataFrame(rows)


def _build_estimators(random_state: int, feature_cols: list[str]) -> dict[str, Pipeline]:
    """Five estimators on the same CV folds:

    - ``length_reasoning_only`` — just ``reasoning_n_chunks``. The honest
                                  length baseline: cannot leak framework
                                  identity through answer-channel presence.
    - ``length_only``           — 3 chunk-count features. Kept as a *leakage
                                  diagnostic*: ``answer_n_chunks`` is 0 for
                                  100% of BFCL items and >0 elsewhere, so
                                  it acts as a near-perfect framework
                                  fingerprint. Treat its lift over
                                  length_reasoning_only as a measure of how
                                  much "answer-channel presence" leaks into
                                  what looks like a structural baseline.
    - ``logreg``                — logreg on trajectory_full.
    - ``lightgbm``              — gradient boosting on trajectory_full.
    - ``framework_only``        — logreg on one-hot ``framework``.

    The cleanest "shape lift" is trajectory_full minus
    length_reasoning_only, *not* minus length_only.
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
        "length_reasoning_only": _logreg_pipe(),
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
        "framework_only": Pipeline([
            ("ohe", ColumnTransformer(
                transformers=[("fw", OneHotEncoder(handle_unknown="ignore"), ["framework"])],
                remainder="drop",
            )),
            ("clf", LogisticRegression(
                max_iter=5000, class_weight="balanced", random_state=random_state,
            )),
        ]),
    }


def _eval_oof(
    estimator: Pipeline, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold
) -> dict:
    probs = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
    preds = probs.argmax(axis=1)
    classes = np.unique(y)

    macro_auc = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=classes))
    per_class_auc = {
        str(cls): float(roc_auc_score((y == cls).astype(int), probs[:, i]))
        for i, cls in enumerate(classes)
    }
    return {
        "probs": probs,
        "preds": preds,
        "macro_auc_ovr": macro_auc,
        "per_class_auc": per_class_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "accuracy": float((preds == y).mean()),
        "confusion_matrix": confusion_matrix(y, preds, labels=classes).tolist(),
    }


def _bootstrap_paired_macro_auc(
    y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray,
    n: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    classes = np.unique(y)
    deltas = []
    for _ in range(n):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < len(classes):
            continue
        try:
            a = roc_auc_score(y[idx], probs_a[idx], multi_class="ovr", average="macro", labels=classes)
            b = roc_auc_score(y[idx], probs_b[idx], multi_class="ovr", average="macro", labels=classes)
        except ValueError:
            continue
        deltas.append(b - a)
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(deltas)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def _plot_confusion(cm, classes, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    arr = np.asarray(cm, dtype=float)
    norm = arr / arr.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
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
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_per_class_auc(results: dict, classes: list[str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    short = [c.replace(" Hallucination", "") for c in classes]
    models = list(results.keys())
    width = 0.8 / len(models)
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = {
        "length_reasoning_only": "#fddbc7",
        "length_only": "#f4a582",
        "logreg": "#2166ac",
        "lightgbm": "#1b7837",
        "framework_only": "#999999",
    }
    for i, m in enumerate(models):
        per = results[m]["per_class_auc"]
        # per_class_auc keys are stringified integer class indices.
        ys = [per[str(k)] for k in range(len(classes))]
        ax.bar(x + i * width - 0.4 + width / 2, ys, width=width,
               label=m, color=palette.get(m, "#aaaaaa"))
    ax.axhline(0.5, color="black", lw=0.8, ls="--", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right")
    ax.set_ylabel("One-vs-rest AUC")
    ax.set_title("AgentHallu — per-class AUC by model")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="best", fontsize=9)
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
    args = parser.parse_args()

    setup_logging("INFO")

    df = pd.read_csv(args.features_csv.resolve())
    df["target"] = df["final_answer_correct"].astype(int)
    feature_cols = choose_feature_sets(df)[args.feature_set]
    if not feature_cols:
        logger.error("Feature set '%s' is empty", args.feature_set)
        return 1

    labels = _load_labels(_ensure_repo(args.source_root))
    df = df.merge(labels, on="id", how="left")
    failures = df[df["target"] == 0].reset_index(drop=True)
    failures = failures.dropna(subset=["hallucination_category"]).reset_index(drop=True)

    classes = sorted(failures["hallucination_category"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = failures["hallucination_category"].map(class_to_idx).to_numpy()
    counts = failures["hallucination_category"].value_counts().to_dict()
    logger.info("AgentHallu failures: %d items, classes: %s", len(failures), counts)

    # Cross-tab for the leakage diagnostic.
    cross = pd.crosstab(failures["framework"], failures["hallucination_category"])

    out_dir = args.features_csv.parent / "diagnostics" / "all"
    out_dir.mkdir(parents=True, exist_ok=True)

    # X for trajectory-based models = numeric features. X for framework_only
    # is just the framework column. Pipelines pull the columns they need.
    X = failures[feature_cols + ["framework"]].copy()

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    estimators = _build_estimators(args.random_state, feature_cols)

    LENGTH_COLS = [c for c in ("reasoning_n_chunks", "answer_n_chunks", "total_n_chunks") if c in feature_cols]
    LENGTH_REASONING_COLS = [c for c in ("reasoning_n_chunks",) if c in feature_cols]

    results: dict[str, dict] = {}
    for name, est in estimators.items():
        if name == "framework_only":
            X_in = X[["framework"]]
        elif name == "length_only":
            X_in = X[LENGTH_COLS]
        elif name == "length_reasoning_only":
            X_in = X[LENGTH_REASONING_COLS]
        else:
            X_in = X[feature_cols]
        logger.info("Fitting %s on %d items × %s", name, len(failures), X_in.shape)
        results[name] = _eval_oof(est, X_in, y, cv)

        _plot_confusion(
            results[name]["confusion_matrix"], classes,
            out_dir / f"agenthallu_confusion_{name}.png",
            title=f"AgentHallu — {name} (macro-AUC {results[name]['macro_auc_ovr']:.3f})",
        )
        oof = pd.DataFrame(results[name]["probs"], columns=[f"p_{c}" for c in classes])
        oof.insert(0, "id", failures["id"].to_numpy())
        oof["framework"] = failures["framework"].to_numpy()
        oof["true_class"] = failures["hallucination_category"].to_numpy()
        oof["pred_class"] = [classes[p] for p in results[name]["preds"]]
        oof.to_parquet(out_dir / f"agenthallu_oof_{name}.parquet", index=False)

    _plot_per_class_auc(results, classes, out_dir / "agenthallu_per_class_auc.png")

    rng = np.random.default_rng(args.random_state)
    delta_lgbm_vs_logreg = _bootstrap_paired_macro_auc(
        y, results["logreg"]["probs"], results["lightgbm"]["probs"], args.n_bootstrap, rng)
    delta_lgbm_vs_fw = _bootstrap_paired_macro_auc(
        y, results["framework_only"]["probs"], results["lightgbm"]["probs"], args.n_bootstrap, rng)
    delta_logreg_vs_fw = _bootstrap_paired_macro_auc(
        y, results["framework_only"]["probs"], results["logreg"]["probs"], args.n_bootstrap, rng)
    # Lift of shape over chunk counts. We anchor on length_reasoning_only —
    # the honest length baseline — because length_only leaks framework
    # identity through answer_n_chunks (BFCL has 0 answer chunks → perfectly
    # tags Tool-Use Hallucination).
    delta_logreg_vs_length = _bootstrap_paired_macro_auc(
        y, results["length_reasoning_only"]["probs"], results["logreg"]["probs"], args.n_bootstrap, rng)
    delta_lgbm_vs_length = _bootstrap_paired_macro_auc(
        y, results["length_reasoning_only"]["probs"], results["lightgbm"]["probs"], args.n_bootstrap, rng)
    # Quantify the leakage: length_only over length_reasoning_only. The
    # bigger this is, the more the conventional 3-feature length baseline
    # is just a framework fingerprint.
    delta_length_leakage = _bootstrap_paired_macro_auc(
        y, results["length_reasoning_only"]["probs"], results["length_only"]["probs"], args.n_bootstrap, rng)

    summary = {
        "n_items": int(len(failures)),
        "classes": classes,
        "class_counts": {c: int(v) for c, v in counts.items()},
        "feature_set": args.feature_set,
        "cv_folds": args.cv_folds,
        "random_state": args.random_state,
        "chance_macro_auc": 0.5,
        "framework_x_category": cross.to_dict(),
        "models": {
            name: {k: v for k, v in r.items() if k not in ("probs", "preds")}
            for name, r in results.items()
        },
        "delta_macro_auc": {
            "lightgbm_minus_logreg": dict(zip(("mean", "ci_low", "ci_high"), delta_lgbm_vs_logreg)),
            "lightgbm_minus_framework_only": dict(zip(("mean", "ci_low", "ci_high"), delta_lgbm_vs_fw)),
            "logreg_minus_framework_only": dict(zip(("mean", "ci_low", "ci_high"), delta_logreg_vs_fw)),
            "logreg_minus_length_reasoning_only": dict(zip(("mean", "ci_low", "ci_high"), delta_logreg_vs_length)),
            "lightgbm_minus_length_reasoning_only": dict(zip(("mean", "ci_low", "ci_high"), delta_lgbm_vs_length)),
            "length_only_minus_length_reasoning_only": dict(zip(("mean", "ci_low", "ci_high"), delta_length_leakage)),
        },
    }
    (out_dir / "agenthallu_classification.json").write_text(json.dumps(summary, indent=2))

    # Markdown report.
    md = [
        "# SH6 Stage-5g — AgentHallu Hallucination-Category Classification (Full Set)",
        "",
        f"**Subset:** all hallucinated AgentHallu items ({len(failures)}; correct items excluded)",
        f"**Classes ({len(classes)}):** "
        + ", ".join(f"`{c}` (n={counts[c]})" for c in classes),
        f"**Feature set:** `{args.feature_set}` ({len(feature_cols)} cols)",
        f"**CV:** {args.cv_folds}-fold stratified, random_state={args.random_state}",
        f"**Chance baseline (macro one-vs-rest AUC):** 0.500",
        "",
        "## Class × framework cross-tab (leakage source)",
        "",
        cross.to_markdown(),
        "",
        "## Summary",
        "",
        "| Model | Macro AUC (OvR) | Bal. Acc | Acc |",
        "|---|---:|---:|---:|",
    ]
    for name, r in results.items():
        md.append(f"| `{name}` | {r['macro_auc_ovr']:.3f} | {r['balanced_accuracy']:.3f} | {r['accuracy']:.3f} |")

    md += ["", "## Per-class AUC (one-vs-rest)", "",
           "| Class | length_reasoning_only | length_only | logreg | lightgbm | framework_only |",
           "|---|---:|---:|---:|---:|---:|"]
    for cls in classes:
        i = class_to_idx[cls]
        md.append(
            f"| `{cls}` | "
            f"{results['length_reasoning_only']['per_class_auc'][str(i)]:.3f} | "
            f"{results['length_only']['per_class_auc'][str(i)]:.3f} | "
            f"{results['logreg']['per_class_auc'][str(i)]:.3f} | "
            f"{results['lightgbm']['per_class_auc'][str(i)]:.3f} | "
            f"{results['framework_only']['per_class_auc'][str(i)]:.3f} |"
        )

    md += ["", "## Δ macro-AUC (paired bootstrap, 95% CI)", "",
           "| Comparison | Δ mean | CI low | CI high | Verdict |",
           "|---|---:|---:|---:|:---|"]
    for label, (m, lo, hi) in [
        ("lightgbm − logreg", delta_lgbm_vs_logreg),
        ("logreg − length_reasoning_only (shape lift, clean)", delta_logreg_vs_length),
        ("lightgbm − length_reasoning_only (shape lift, clean)", delta_lgbm_vs_length),
        ("length_only − length_reasoning_only (framework leak in length baseline)", delta_length_leakage),
        ("lightgbm − framework_only", delta_lgbm_vs_fw),
        ("logreg − framework_only", delta_logreg_vs_fw),
    ]:
        verdict = (
            "significant lift" if (not np.isnan(lo) and lo > 0)
            else "significant regression" if (not np.isnan(hi) and hi < 0)
            else "inconclusive (CI straddles 0)"
        )
        md.append(f"| {label} | {m:+.3f} | {lo:+.3f} | {hi:+.3f} | {verdict} |")

    md += ["", "## Plots", "",
           "![per-class AUC](agenthallu_per_class_auc.png)",
           "",
           "![length_reasoning_only](agenthallu_confusion_length_reasoning_only.png)",
           "",
           "![length_only — leaks framework via answer_n_chunks](agenthallu_confusion_length_only.png)",
           "",
           "![logreg](agenthallu_confusion_logreg.png)",
           "",
           "![lightgbm](agenthallu_confusion_lightgbm.png)",
           "",
           "![framework_only](agenthallu_confusion_framework_only.png)",
           ""]

    (out_dir / "agenthallu_classification.md").write_text("\n".join(md))

    logger.info("Macro AUC: logreg=%.3f, lightgbm=%.3f, framework_only=%.3f",
                results["logreg"]["macro_auc_ovr"],
                results["lightgbm"]["macro_auc_ovr"],
                results["framework_only"]["macro_auc_ovr"])
    logger.info("Δ lgbm−logreg = %+.3f [%+.3f, %+.3f]", *delta_lgbm_vs_logreg)
    logger.info("Δ lgbm−fw_only = %+.3f [%+.3f, %+.3f]", *delta_lgbm_vs_fw)
    return 0


if __name__ == "__main__":
    sys.exit(main())
