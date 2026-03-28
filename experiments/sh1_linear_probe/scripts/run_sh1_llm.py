#!/usr/bin/env python3
"""SH1 rerun with LLM-derived labels (section-blind).

Trains the same linear probe on LLM-annotated spans instead of
section-derived SH0 labels. Tests on the 200-span LLM-annotated
test set from SH1c Exp C.

Usage:
    cd playground/SLoD-SH1
    python scripts/run_sh1_llm.py
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import LABEL_MAP, LABEL_NAMES, load_config, load_spans, setup_logging


def load_relabel_results(relabel_dir: Path) -> dict[int, str]:
    """Load all result_N.json files, return {span_index: llm_label}."""
    labels = {}
    for p in sorted(relabel_dir.glob("result_*.json")):
        with open(p) as f:
            batch = json.load(f)
        for item in batch:
            if isinstance(item, dict) and "id" in item and "label" in item:
                label = item["label"].lower().strip()
                if label in LABEL_MAP:
                    labels[int(item["id"])] = label
    return labels


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    config = load_config(PROJECT_ROOT / "config.yaml")
    data_dir = PROJECT_ROOT / config["data_dir"]
    sh0_dir = PROJECT_ROOT / config["sh0_data_dir"]
    reports_dir = PROJECT_ROOT / config["reports_dir"]
    relabel_dir = data_dir / "relabel"

    # Load spans and embeddings
    spans = load_spans(sh0_dir / config["primary_dataset"])

    emb_data = np.load(
        data_dir / "embeddings" / "scibert_length_matched.npz", allow_pickle=True
    )
    embeddings = emb_data["embeddings"]
    sh0_labels = emb_data["labels"]

    # Load LLM labels for train+val
    llm_labels = load_relabel_results(relabel_dir)
    print(f"LLM labels loaded: {len(llm_labels)} spans")

    if len(llm_labels) < 100:
        print("ERROR: Too few LLM labels found. Check data/relabel/result_*.json files.")
        sys.exit(1)

    # Load metadata to know which spans are train vs val
    with open(relabel_dir / "samples_metadata.json") as f:
        metadata = json.load(f)

    train_indices = []
    train_labels = []
    val_indices = []
    val_labels = []

    for sample in metadata:
        idx = sample["id"]
        if idx in llm_labels:
            if sample["split"] == "train":
                train_indices.append(idx)
                train_labels.append(LABEL_MAP[llm_labels[idx]])
            elif sample["split"] == "val":
                val_indices.append(idx)
                val_labels.append(LABEL_MAP[llm_labels[idx]])

    train_indices = np.array(train_indices)
    train_labels = np.array(train_labels)
    val_indices = np.array(val_indices)
    val_labels = np.array(val_labels)

    print(f"Train: {len(train_indices)} spans")
    print(f"  Label dist: {dict(Counter(train_labels.tolist()))}")
    print(f"Val: {len(val_indices)} spans")
    print(f"  Label dist: {dict(Counter(val_labels.tolist()))}")

    # Load test set (200 LLM-labeled spans from SH1c Exp C)
    with open(data_dir / "annotations_llm.json") as f:
        test_annotations = json.load(f)

    test_indices = np.array([a["span_index"] for a in test_annotations if "llm_label" in a])
    test_llm_labels = np.array([LABEL_MAP[a["llm_label"]] for a in test_annotations if "llm_label" in a])
    test_sh0_labels = np.array([LABEL_MAP[a["_sh0_label"]] for a in test_annotations if "llm_label" in a])

    print(f"Test: {len(test_indices)} spans (LLM-labeled from SH1c)")
    print(f"  LLM dist: {dict(Counter(test_llm_labels.tolist()))}")

    # Scale embeddings
    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_indices])
    X_val = scaler.transform(embeddings[val_indices])
    X_test = scaler.transform(embeddings[test_indices])

    # ---------------------------------------------------------------
    # Train probe on LLM labels
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SH1 RERUN: Probe trained on LLM labels")
    print("=" * 60)

    # Hyperparameter sweep
    best_c, best_val_f1 = None, -1
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_train, train_labels)
        vf1 = f1_score(val_labels, clf.predict(X_val), average="macro")
        print(f"  C={C}: val F1={vf1:.4f}")
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_c = C

    # Final model
    clf_llm = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs", random_state=42)
    clf_llm.fit(X_train, train_labels)
    y_pred_llm = clf_llm.predict(X_test)

    # Evaluate vs LLM test labels
    f1_vs_llm = f1_score(test_llm_labels, y_pred_llm, average="macro")
    print(f"\n  Best C={best_c}")
    print(f"  Probe (LLM-trained) F1 vs LLM test labels: {f1_vs_llm:.4f}")
    print(classification_report(test_llm_labels, y_pred_llm, target_names=LABEL_NAMES, digits=4))

    # Evaluate vs SH0 test labels (how much does it still agree with section-based?)
    f1_vs_sh0 = f1_score(test_sh0_labels, y_pred_llm, average="macro")
    print(f"  Probe (LLM-trained) F1 vs SH0 test labels: {f1_vs_sh0:.4f}")

    # ---------------------------------------------------------------
    # Compare: probe trained on SH0 labels
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON: Probe trained on SH0 labels (original)")
    print("=" * 60)

    # Use same train indices but with SH0 labels
    sh0_train_labels = sh0_labels[train_indices]

    best_c2, best_val_f1_2 = None, -1
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf2 = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
        clf2.fit(X_train, sh0_train_labels)
        vf1 = f1_score(val_labels, clf2.predict(X_val), average="macro")
        if vf1 > best_val_f1_2:
            best_val_f1_2 = vf1
            best_c2 = C

    clf_sh0 = LogisticRegression(C=best_c2, max_iter=2000, solver="lbfgs", random_state=42)
    clf_sh0.fit(X_train, sh0_train_labels)
    y_pred_sh0 = clf_sh0.predict(X_test)

    f1_sh0_vs_llm = f1_score(test_llm_labels, y_pred_sh0, average="macro")
    f1_sh0_vs_sh0 = f1_score(test_sh0_labels, y_pred_sh0, average="macro")
    print(f"  Probe (SH0-trained) F1 vs LLM test labels: {f1_sh0_vs_llm:.4f}")
    print(f"  Probe (SH0-trained) F1 vs SH0 test labels: {f1_sh0_vs_sh0:.4f}")

    # ---------------------------------------------------------------
    # Agreement between LLM train labels and SH0 labels
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LABEL AGREEMENT: LLM vs SH0 on training set")
    print("=" * 60)

    kappa = cohen_kappa_score(sh0_train_labels, train_labels)
    agree = np.mean(sh0_train_labels == train_labels)
    print(f"  Cohen's κ: {kappa:.4f}")
    print(f"  Agreement: {agree:.4f}")
    print(f"  Confusion (SH0 rows vs LLM cols):")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(sh0_train_labels, train_labels)
    print(f"           macro  meso  micro")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:>5}  {cm[i]}")

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Probe training':<25} {'Test vs LLM':>14} {'Test vs SH0':>14}")
    print("-" * 55)
    print(f"{'LLM labels':<25} {f1_vs_llm:>14.4f} {f1_vs_sh0:>14.4f}")
    print(f"{'SH0 labels':<25} {f1_sh0_vs_llm:>14.4f} {f1_sh0_vs_sh0:>14.4f}")
    print(f"{'SH1c residualized':<25} {'0.4039':>14} {'—':>14}")

    # ---------------------------------------------------------------
    # Save results and report
    # ---------------------------------------------------------------
    results = {
        "llm_trained": {
            "best_C": best_c,
            "f1_vs_llm": float(f1_vs_llm),
            "f1_vs_sh0": float(f1_vs_sh0),
            "train_n": len(train_indices),
            "val_n": len(val_indices),
            "test_n": len(test_indices),
        },
        "sh0_trained": {
            "best_C": best_c2,
            "f1_vs_llm": float(f1_sh0_vs_llm),
            "f1_vs_sh0": float(f1_sh0_vs_sh0),
        },
        "label_agreement": {
            "kappa": float(kappa),
            "agreement": float(agree),
        },
    }

    results_path = data_dir / "results" / "probe_results_llm_labels.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate report
    report_lines = [
        "# SH1 Rerun: LLM-Labeled Probe",
        "",
        "> Probe trained on 2000 LLM-annotated spans (section-blind), tested on 200 LLM-annotated spans.",
        "",
        "## Label Agreement (LLM vs SH0 on training set)",
        "",
        f"- Cohen's κ: {kappa:.4f}",
        f"- Raw agreement: {agree:.4f}",
        "",
        "## Probe Results",
        "",
        "| Probe training | F1 vs LLM labels | F1 vs SH0 labels |",
        "|---|---|---|",
        f"| **LLM labels** | **{f1_vs_llm:.4f}** | {f1_vs_sh0:.4f} |",
        f"| SH0 labels | {f1_sh0_vs_llm:.4f} | {f1_sh0_vs_sh0:.4f} |",
        f"| SH1c residualized | — | 0.4039 |",
        "",
        "## Per-Class F1 (LLM-trained probe vs LLM test labels)",
        "",
        classification_report(test_llm_labels, y_pred_llm, target_names=LABEL_NAMES, digits=4),
        "",
        "## Interpretation",
        "",
        f"The LLM-trained probe achieves F1={f1_vs_llm:.4f} on LLM test labels, ",
        f"compared to the SH0-trained probe's F1={f1_sh0_vs_llm:.4f} on the same labels. ",
        "This measures whether training on content-blind labels improves SLoD detection ",
        "beyond what section-derived labels provide.",
    ]

    report_path = reports_dir / "SH1_LLM_RERUN_REPORT.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Report saved to {report_path}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
