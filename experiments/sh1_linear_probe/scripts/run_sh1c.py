#!/usr/bin/env python3
"""SH1c: Section-Controlled SLoD Validation.

Addresses circularity concern: SH0 labels are derived from section names,
so the SH1 probe may decode section identity rather than SLoD.

Three experiments:
  A — Section-residualized probe (remove section signal from embeddings)
  B — Cross-section transfer (train on some section types, test on others)
  C — LLM-blind annotation (run separately via PAL consensus, results loaded here)

Usage:
    cd playground/SLoD-SH1
    python scripts/run_sh1c.py              # Exp A + B
    python scripts/run_sh1c.py --exp-c      # Include Exp C (needs annotations file)
    python scripts/run_sh1c.py --sample-c   # Generate sample file for LLM annotation
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import LABEL_MAP, LABEL_NAMES, load_config, load_spans, setup_logging

CHANCE_F1 = 0.333


def load_data(config):
    """Load spans, embeddings, and grouped splits."""
    sh0_dir = PROJECT_ROOT / config["sh0_data_dir"]
    data_dir = PROJECT_ROOT / config["data_dir"]

    spans = load_spans(sh0_dir / config["primary_dataset"])

    with open(data_dir / "splits_grouped.json") as f:
        splits = json.load(f)

    data = np.load(
        data_dir / "embeddings" / "scibert_length_matched.npz", allow_pickle=True
    )
    embeddings = data["embeddings"]
    labels = data["labels"]

    return spans, splits, embeddings, labels


# ===================================================================
# Exp A: Section-Residualized Probe
# ===================================================================


def exp_a_residualized_probe(spans, splits, embeddings, labels):
    """Remove section-name signal from embeddings, probe residuals."""
    logger = logging.getLogger(__name__)
    print("\n" + "=" * 60)
    print("EXP A: Section-Residualized Probe")
    print("=" * 60)

    train_idx = np.array(splits["train"])
    val_idx = np.array(splits["val"])
    test_idx = np.array(splits["test"])

    sec_names = [spans[i].get("section_name") or "UNK" for i in range(len(spans))]

    # Scale embeddings
    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_idx])
    X_val = scaler.transform(embeddings[val_idx])
    X_test = scaler.transform(embeddings[test_idx])

    y_train = labels[train_idx]
    y_val = labels[val_idx]
    y_test = labels[test_idx]

    # --- Baseline: section-name n-gram ---
    vec = CountVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    X_sec_train = vec.fit_transform([sec_names[i] for i in train_idx])
    X_sec_test = vec.transform([sec_names[i] for i in test_idx])

    clf_sec = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    clf_sec.fit(X_sec_train, y_train)
    sec_f1 = f1_score(y_test, clf_sec.predict(X_sec_test), average="macro")
    print(f"  Section-name baseline F1: {sec_f1:.4f}")

    # --- Full embedding probe (reference) ---
    clf_full = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", random_state=42)
    clf_full.fit(X_train, y_train)
    full_f1 = f1_score(y_test, clf_full.predict(X_test), average="macro")
    print(f"  Full embedding probe F1:  {full_f1:.4f}")

    # --- Residualization: remove section-predictable component ---
    # Strategy: learn section→embedding mapping (Ridge: sec_features → embedding),
    # then for each embedding, subtract its projection onto the section subspace.
    # This removes the component of embedding variance explained by section name.
    X_sec_train_dense = X_sec_train.toarray().astype(np.float32)
    X_sec_val_dense = vec.transform([sec_names[i] for i in val_idx]).toarray().astype(np.float32)
    X_sec_test_dense = X_sec_test.toarray().astype(np.float32)

    # Learn: section features → embedding (what embedding looks like given section)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_sec_train_dense, X_train)

    # Subtract section-predicted embedding component
    X_train_residual = X_train - ridge.predict(X_sec_train_dense)
    X_val_residual = X_val - ridge.predict(X_sec_val_dense)
    X_test_residual = X_test - ridge.predict(X_sec_test_dense)

    # Re-scale residuals
    scaler_r = StandardScaler()
    X_train_r = scaler_r.fit_transform(X_train_residual)
    X_val_r = scaler_r.transform(X_val_residual)
    X_test_r = scaler_r.transform(X_test_residual)

    # Hyperparameter sweep on residuals
    best_c, best_val_f1 = None, -1
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_train_r, y_train)
        vf1 = f1_score(y_val, clf.predict(X_val_r), average="macro")
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_c = C

    clf_residual = LogisticRegression(
        C=best_c, max_iter=2000, solver="lbfgs", random_state=42
    )
    clf_residual.fit(X_train_r, y_train)
    y_pred_r = clf_residual.predict(X_test_r)
    residual_f1 = f1_score(y_test, y_pred_r, average="macro")

    print(f"  Residualized probe F1:    {residual_f1:.4f} (C={best_c})")
    print(
        f"  Signal beyond section:    {residual_f1 - CHANCE_F1:+.4f} above chance ({CHANCE_F1:.3f})"
    )
    print(
        classification_report(
            y_test, y_pred_r, target_names=LABEL_NAMES, digits=4
        )
    )

    passed = residual_f1 > 0.40
    print(f"  Exit criterion (F1 > 0.40): {'PASS' if passed else 'FAIL'}")

    return {
        "section_name_f1": float(sec_f1),
        "full_probe_f1": float(full_f1),
        "residual_probe_f1": float(residual_f1),
        "residual_best_C": best_c,
        "above_chance": float(residual_f1 - CHANCE_F1),
        "passed": passed,
        "per_class_f1": {
            name: float(f)
            for name, f in zip(LABEL_NAMES, f1_score(y_test, y_pred_r, average=None))
        },
    }


# ===================================================================
# Exp B: Cross-Section Transfer
# ===================================================================


def exp_b_cross_section_transfer(spans, splits, embeddings, labels):
    """Train on one set of section types, test on different ones."""
    logger = logging.getLogger(__name__)
    print("\n" + "=" * 60)
    print("EXP B: Cross-Section Transfer")
    print("=" * 60)

    # Build section → dominant label mapping (from ALL data, not just train)
    sec_label = defaultdict(Counter)
    for s in spans:
        sec_label[s.get("section_name") or "UNK"][s["label"]] += 1

    # Classify sections by purity
    pure_macro_secs = set()
    pure_meso_secs = set()
    pure_micro_secs = set()
    mixed_secs = set()

    for sec, counts in sec_label.items():
        total = sum(counts.values())
        if total < 3:  # skip very rare sections
            continue
        dominant_label, dominant_count = counts.most_common(1)[0]
        purity = dominant_count / total
        if purity >= 0.95:  # near-pure sections
            if dominant_label == "macro":
                pure_macro_secs.add(sec)
            elif dominant_label == "meso":
                pure_meso_secs.add(sec)
            elif dominant_label == "micro":
                pure_micro_secs.add(sec)
        else:
            mixed_secs.add(sec)

    print(f"  Pure macro sections: {len(pure_macro_secs)}")
    print(f"  Pure meso sections:  {len(pure_meso_secs)}")
    print(f"  Pure micro sections: {len(pure_micro_secs)}")
    print(f"  Mixed sections:      {len(mixed_secs)}")

    train_idx = np.array(splits["train"])
    test_idx = np.array(splits["test"])

    scaler = StandardScaler()
    X_all = scaler.fit_transform(embeddings)

    results = {}

    # --- Transfer 1: Train on macro+micro sections, test on meso sections ---
    print("\n  --- Transfer: Train macro+micro → Test meso ---")
    train_mask = np.array(
        [
            i
            for i in train_idx
            if (spans[i].get("section_name") or "UNK") in (pure_macro_secs | pure_micro_secs)
        ]
    )
    test_mask = np.array(
        [
            i
            for i in test_idx
            if (spans[i].get("section_name") or "UNK") in pure_meso_secs
        ]
    )

    if len(train_mask) > 0 and len(test_mask) > 0:
        X_tr, y_tr = X_all[train_mask], labels[train_mask]
        X_te, y_te = X_all[test_mask], labels[test_mask]

        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        f1 = f1_score(y_te, y_pred, average="macro")
        acc = accuracy_score(y_te, y_pred)

        # Meso prediction rate
        meso_pred_rate = float(np.mean(y_pred == 1))
        meso_true_rate = float(np.mean(y_te == 1))

        print(f"  Train: {len(train_mask)} spans (macro+micro sections)")
        print(f"  Test:  {len(test_mask)} spans (meso sections)")
        print(f"  Test label distribution: {dict(Counter(y_te.tolist()))}")
        print(f"  Macro-F1: {f1:.4f}, Accuracy: {acc:.4f}")
        print(f"  Meso prediction rate: {meso_pred_rate:.3f} (true: {meso_true_rate:.3f})")
        print(classification_report(y_te, y_pred, target_names=LABEL_NAMES, digits=4))

        results["macro_micro_to_meso"] = {
            "train_n": len(train_mask),
            "test_n": len(test_mask),
            "macro_f1": float(f1),
            "accuracy": float(acc),
            "meso_prediction_rate": meso_pred_rate,
        }
    else:
        print("  Insufficient data for this transfer.")
        results["macro_micro_to_meso"] = None

    # --- Transfer 2: Train on macro sections only, test on micro sections ---
    print("\n  --- Transfer: Train macro-only → Test micro sections ---")
    train_mask2 = np.array(
        [i for i in train_idx if (spans[i].get("section_name") or "UNK") in pure_macro_secs]
    )
    test_mask2 = np.array(
        [i for i in test_idx if (spans[i].get("section_name") or "UNK") in pure_micro_secs]
    )

    if len(train_mask2) > 0 and len(test_mask2) > 0:
        X_tr2, y_tr2 = X_all[train_mask2], labels[train_mask2]
        X_te2, y_te2 = X_all[test_mask2], labels[test_mask2]

        # This is essentially a novelty detection — does probe say "not macro"?
        clf2 = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
        clf2.fit(X_tr2, y_tr2)
        y_pred2 = clf2.predict(X_te2)

        # What fraction does it correctly identify as NOT macro?
        not_macro_rate = float(np.mean(y_pred2 != 0))
        micro_pred_rate = float(np.mean(y_pred2 == 2))

        print(f"  Train: {len(train_mask2)} spans (macro sections only, label={dict(Counter(y_tr2.tolist()))})")
        print(f"  Test:  {len(test_mask2)} spans (micro sections, label={dict(Counter(y_te2.tolist()))})")
        print(f"  Not-macro prediction rate: {not_macro_rate:.3f}")
        print(f"  Micro prediction rate:     {micro_pred_rate:.3f}")

        results["macro_to_micro"] = {
            "train_n": len(train_mask2),
            "test_n": len(test_mask2),
            "not_macro_rate": not_macro_rate,
            "micro_pred_rate": micro_pred_rate,
        }
    else:
        print("  Insufficient data.")
        results["macro_to_micro"] = None

    # --- Transfer 3: Train on ALL section types, test on MIXED sections ---
    print("\n  --- Probe on mixed-label sections only ---")
    mixed_test = np.array(
        [i for i in test_idx if (spans[i].get("section_name") or "UNK") in mixed_secs]
    )
    if len(mixed_test) > 0:
        X_tr3, y_tr3 = X_all[train_idx], labels[train_idx]
        X_te3, y_te3 = X_all[mixed_test], labels[mixed_test]

        clf3 = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", random_state=42)
        clf3.fit(X_tr3, y_tr3)
        y_pred3 = clf3.predict(X_te3)
        f1_mixed = f1_score(y_te3, y_pred3, average="macro")

        print(f"  Test: {len(mixed_test)} spans from mixed sections")
        print(f"  Macro-F1: {f1_mixed:.4f}")
        print(classification_report(y_te3, y_pred3, target_names=LABEL_NAMES, digits=4))

        results["mixed_sections"] = {
            "test_n": len(mixed_test),
            "macro_f1": float(f1_mixed),
        }
    else:
        results["mixed_sections"] = None

    return results


# ===================================================================
# Exp C: LLM-Blind Annotation (load + analyze)
# ===================================================================


def sample_for_annotation(spans, splits, output_path: Path, n=200):
    """Generate stratified sample for LLM blind annotation."""
    print("\n" + "=" * 60)
    print(f"Generating annotation sample ({n} spans)")
    print("=" * 60)

    test_idx = splits["test"]
    rng = np.random.RandomState(42)

    # Categorize test spans by label_source
    by_source = defaultdict(list)
    for i in test_idx:
        s = spans[i]
        src = s.get("label_source", "unknown")
        by_source[src].append(i)

    # Also find mixed-section spans
    sec_label = defaultdict(Counter)
    for s in spans:
        sec_label[s.get("section_name") or "UNK"][s["label"]] += 1
    mixed_secs = {sec for sec, c in sec_label.items() if len(c) > 1}
    mixed_indices = [i for i in test_idx if (spans[i].get("section_name") or "UNK") in mixed_secs]

    # Target allocation
    targets = {
        "section_regex": 80,
        "position": 50,
        "combined_content": 50,  # combined + content merged
        "mixed_section": 20,
    }

    sample_indices = []
    used = set()

    # Mixed section first (subset of others)
    mixed_available = [i for i in mixed_indices if i not in used]
    if mixed_available:
        chosen = rng.choice(mixed_available, size=min(targets["mixed_section"], len(mixed_available)), replace=False)
        sample_indices.extend(chosen)
        used.update(chosen)

    # Combined + content
    combined_pool = by_source.get("combined", []) + by_source.get("content", [])
    combined_available = [i for i in combined_pool if i not in used]
    if combined_available:
        chosen = rng.choice(combined_available, size=min(targets["combined_content"], len(combined_available)), replace=False)
        sample_indices.extend(chosen)
        used.update(chosen)

    # Position
    pos_available = [i for i in by_source.get("position", []) if i not in used]
    if pos_available:
        chosen = rng.choice(pos_available, size=min(targets["position"], len(pos_available)), replace=False)
        sample_indices.extend(chosen)
        used.update(chosen)

    # Fill rest with section_regex
    remaining = n - len(sample_indices)
    regex_available = [i for i in by_source.get("section_regex", []) if i not in used]
    if regex_available and remaining > 0:
        chosen = rng.choice(regex_available, size=min(remaining, len(regex_available)), replace=False)
        sample_indices.extend(chosen)
        used.update(chosen)

    # Balance by label within sample
    rng.shuffle(sample_indices)

    # Build output
    samples = []
    for idx, i in enumerate(sample_indices):
        s = spans[i]
        samples.append({
            "id": idx,
            "span_index": int(i),
            "text": s["text"],
            # Hidden from LLM:
            "_section_name": s.get("section_name", ""),
            "_sh0_label": s["label"],
            "_label_source": s.get("label_source", ""),
            "_paper_id": s["paper_id"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    # Stats
    label_dist = Counter(s["_sh0_label"] for s in samples)
    source_dist = Counter(s["_label_source"] for s in samples)
    print(f"  Total: {len(samples)}")
    print(f"  By label: {dict(label_dist)}")
    print(f"  By source: {dict(source_dist)}")
    print(f"  Saved to: {output_path}")

    return samples


def exp_c_analyze_annotations(spans, splits, embeddings, labels, annotations_path: Path):
    """Analyze LLM annotations vs SH0 labels."""
    print("\n" + "=" * 60)
    print("EXP C: LLM-Blind Annotation Analysis")
    print("=" * 60)

    with open(annotations_path) as f:
        annotations = json.load(f)

    # Filter to annotated spans
    valid = [a for a in annotations if "llm_label" in a]
    print(f"  Annotated spans: {len(valid)}/{len(annotations)}")

    if len(valid) < 50:
        print("  Too few annotations, skipping analysis.")
        return {"error": "insufficient_annotations", "n": len(valid)}

    sh0_labels = [LABEL_MAP[a["_sh0_label"]] for a in valid]
    llm_labels = [LABEL_MAP[a["llm_label"]] for a in valid]

    # Cohen's kappa
    kappa = cohen_kappa_score(sh0_labels, llm_labels)
    agreement = accuracy_score(sh0_labels, llm_labels)

    print(f"  Cohen's κ: {kappa:.4f}")
    print(f"  Raw agreement: {agreement:.4f}")
    print(
        classification_report(
            sh0_labels, llm_labels, target_names=LABEL_NAMES, digits=4
        )
    )

    # Agreement by label_source
    print("\n  Agreement by label_source:")
    by_source = defaultdict(lambda: {"agree": 0, "total": 0})
    for a in valid:
        src = a.get("_label_source", "unknown")
        by_source[src]["total"] += 1
        if LABEL_MAP[a["_sh0_label"]] == LABEL_MAP[a["llm_label"]]:
            by_source[src]["agree"] += 1

    for src, counts in sorted(by_source.items()):
        rate = counts["agree"] / counts["total"] if counts["total"] > 0 else 0
        print(f"    {src}: {rate:.3f} ({counts['agree']}/{counts['total']})")

    # Probe on LLM labels
    print("\n  Probe evaluation on LLM labels:")
    span_indices = np.array([a["span_index"] for a in valid])
    llm_labels_arr = np.array(llm_labels)

    train_idx = np.array(splits["train"])
    scaler = StandardScaler()
    X_train = scaler.fit_transform(embeddings[train_idx])
    X_llm = scaler.transform(embeddings[span_indices])
    y_train = labels[train_idx]

    clf = LogisticRegression(C=10.0, max_iter=2000, solver="lbfgs", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_llm)

    probe_vs_llm_f1 = f1_score(llm_labels_arr, y_pred, average="macro")
    probe_vs_sh0_f1 = f1_score(
        np.array([LABEL_MAP[a["_sh0_label"]] for a in valid]), y_pred, average="macro"
    )
    print(f"  Probe F1 vs LLM labels: {probe_vs_llm_f1:.4f}")
    print(f"  Probe F1 vs SH0 labels: {probe_vs_sh0_f1:.4f}")

    passed = kappa > 0.50
    print(f"\n  Exit criterion (κ > 0.50): {'PASS' if passed else 'FAIL'}")

    return {
        "n_annotated": len(valid),
        "kappa": float(kappa),
        "agreement": float(agreement),
        "passed": passed,
        "probe_f1_vs_llm": float(probe_vs_llm_f1),
        "probe_f1_vs_sh0": float(probe_vs_sh0_f1),
        "by_source": {
            src: {"agreement": c["agree"] / c["total"], "n": c["total"]}
            for src, c in by_source.items()
        },
    }


# ===================================================================
# Report Generation
# ===================================================================


def generate_report(results: dict, report_path: Path):
    """Generate SH1c markdown report."""
    lines = []
    lines.append("# SH1c: Section-Controlled SLoD Validation")
    lines.append("")
    lines.append("> Generated by `run_sh1c.py`")
    lines.append("")

    lines.append("## The Confound")
    lines.append("")
    lines.append("SH0 labels are derived primarily from section names (75% `section_regex`, 18% `position`).")
    lines.append("A section-name n-gram baseline achieves F1 = 0.963, far above the embedding probe (F1 = 0.712).")
    lines.append("Question: does the probe capture SLoD signal *beyond* section identity?")
    lines.append("")

    # Exp A
    a = results.get("exp_a", {})
    lines.append("## Exp A: Section-Residualized Probe")
    lines.append("")
    lines.append("Remove section-predictable component from embeddings via Ridge regression, then probe residuals.")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Section-name baseline F1 | {a.get('section_name_f1', 0):.4f} |")
    lines.append(f"| Full embedding probe F1 | {a.get('full_probe_f1', 0):.4f} |")
    lines.append(f"| **Residualized probe F1** | **{a.get('residual_probe_f1', 0):.4f}** |")
    lines.append(f"| Above chance (+{CHANCE_F1:.3f}) | {a.get('above_chance', 0):+.4f} |")
    lines.append(f"| Exit criterion (>0.40) | {'PASS' if a.get('passed') else 'FAIL'} |")
    lines.append("")
    if a.get("per_class_f1"):
        lines.append("Per-class (residualized): " + ", ".join(
            f"{k}={v:.4f}" for k, v in a["per_class_f1"].items()
        ))
        lines.append("")

    # Exp B
    b = results.get("exp_b", {})
    lines.append("## Exp B: Cross-Section Transfer")
    lines.append("")

    b1 = b.get("macro_micro_to_meso")
    if b1:
        lines.append(f"### Train macro+micro sections → Test meso sections")
        lines.append(f"- Train: {b1['train_n']} spans, Test: {b1['test_n']} spans")
        lines.append(f"- Macro-F1: {b1['macro_f1']:.4f}")
        lines.append(f"- Meso prediction rate: {b1['meso_prediction_rate']:.3f}")
        lines.append("")

    b2 = b.get("macro_to_micro")
    if b2:
        lines.append(f"### Train macro sections → Test micro sections")
        lines.append(f"- Train: {b2['train_n']} spans, Test: {b2['test_n']} spans")
        lines.append(f"- Not-macro rate: {b2['not_macro_rate']:.3f}")
        lines.append(f"- Micro prediction rate: {b2['micro_pred_rate']:.3f}")
        lines.append("")

    b3 = b.get("mixed_sections")
    if b3:
        lines.append(f"### Probe on mixed-label sections")
        lines.append(f"- Test: {b3['test_n']} spans")
        lines.append(f"- Macro-F1: {b3['macro_f1']:.4f}")
        lines.append("")

    # Exp C
    c = results.get("exp_c")
    if c and "error" not in c:
        lines.append("## Exp C: LLM-Blind Annotation")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|---|---|")
        lines.append(f"| Annotated spans | {c['n_annotated']} |")
        lines.append(f"| **Cohen's κ** | **{c['kappa']:.4f}** |")
        lines.append(f"| Raw agreement | {c['agreement']:.4f} |")
        lines.append(f"| Probe F1 vs LLM labels | {c['probe_f1_vs_llm']:.4f} |")
        lines.append(f"| Probe F1 vs SH0 labels | {c['probe_f1_vs_sh0']:.4f} |")
        lines.append(f"| Exit criterion (κ>0.50) | {'PASS' if c['passed'] else 'FAIL'} |")
        lines.append("")
        if c.get("by_source"):
            lines.append("Agreement by label_source:")
            lines.append("")
            lines.append("| Source | Agreement | N |")
            lines.append("|---|---|---|")
            for src, info in c["by_source"].items():
                lines.append(f"| {src} | {info['agreement']:.3f} | {info['n']} |")
            lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")

    a_pass = a.get("passed", False)
    c_pass = c.get("passed", False) if c and "error" not in c else None

    if a_pass and c_pass:
        lines.append("**SH1 stands with caveat.** Residualized probe shows SLoD signal beyond section identity, and LLM annotations confirm labels are valid SLoD proxies. Report residualized F1 alongside full F1.")
    elif a_pass and c_pass is None:
        lines.append(f"**Exp A PASS (residual F1={a.get('residual_probe_f1', 0):.4f}).** Embedding signal exists beyond section identity. Exp C pending for label validation.")
    elif not a_pass and (c_pass is True):
        lines.append("**Labels valid but probe captures section, not SLoD.** LLM confirms labels are reasonable SLoD proxies, but the embedding probe fails to separate SLoD from section identity. Need a different approach (e.g., train on LLM labels directly).")
    elif not a_pass and c_pass is None:
        lines.append(f"**Exp A FAIL (residual F1={a.get('residual_probe_f1', 0):.4f}).** Embedding probe does not capture SLoD beyond section identity. Exp C needed to determine if this is a label problem or a probe problem.")
    else:
        lines.append("**SH1 must be revised.** Neither residualized probe nor LLM validation supports the claim that embeddings encode SLoD independently of section structure.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\nReport saved to {report_path}")


# ===================================================================
# Main
# ===================================================================


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="SH1c: Section-Controlled Validation")
    parser.add_argument("--exp-c", action="store_true", help="Include Exp C analysis")
    parser.add_argument("--sample-c", action="store_true", help="Generate sample for LLM annotation")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config.yaml")
    data_dir = PROJECT_ROOT / config["data_dir"]
    reports_dir = PROJECT_ROOT / config["reports_dir"]
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    spans, splits, embeddings, labels = load_data(config)

    annotations_path = data_dir / "annotations_llm.json"

    if args.sample_c:
        sample_for_annotation(spans, splits, annotations_path, n=200)
        return

    # Run Exp A + B
    results = {}
    results["exp_a"] = exp_a_residualized_probe(spans, splits, embeddings, labels)
    results["exp_b"] = exp_b_cross_section_transfer(spans, splits, embeddings, labels)

    # Exp C if requested and annotations exist
    if args.exp_c:
        if annotations_path.exists():
            results["exp_c"] = exp_c_analyze_annotations(
                spans, splits, embeddings, labels, annotations_path
            )
        else:
            print(f"\n  Annotations file not found: {annotations_path}")
            print(f"  Run with --sample-c first, annotate, then rerun with --exp-c")

    # Save results
    output_path = results_dir / "probe_results_section_controlled.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Generate report
    generate_report(results, reports_dir / "SH1c_SECTION_CONTROL_REPORT.md")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
