"""Linear probe training and evaluation for SH1."""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.utils import LABEL_NAMES, load_config, load_spans, load_splits, LABEL_MAP


def _make_classifier(clf_type: str, C: float, params: dict):
    """Instantiate a classifier from config."""
    p = dict(params)
    p["C"] = C
    if clf_type == "LogisticRegression":
        return LogisticRegression(**p)
    elif clf_type == "LinearSVC":
        return LinearSVC(**p)
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    clf_config: dict,
    C: float = 1.0,
) -> dict:
    """Train a classifier, evaluate on val and test, return metrics dict.

    Args:
        clf_config: Dict with 'type' and 'params'.
        C: Regularization parameter.

    Returns:
        Dict with train/val/test metrics.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Train
    clf = _make_classifier(clf_config["type"], C, clf_config["params"])
    clf.fit(X_train_s, y_train)

    # Evaluate
    results = {"C": C, "classifier": clf_config["type"]}

    for split_name, X_s, y_true in [
        ("val", X_val_s, y_val),
        ("test", X_test_s, y_test),
    ]:
        y_pred = clf.predict(X_s)
        results[split_name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro")),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro")),
            "per_class_f1": {
                name: float(f)
                for name, f in zip(
                    LABEL_NAMES,
                    f1_score(y_true, y_pred, average=None),
                )
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, target_names=LABEL_NAMES, output_dict=True
            ),
        }

    return results


def run_hyperparameter_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_values: list[float],
    clf_config: dict,
) -> tuple[float, float]:
    """Sweep over C values, return (best_C, best_val_macro_f1).

    Fits scaler on train, evaluates on val.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_c = C_values[0]
    best_f1 = -1.0

    for C in C_values:
        clf = _make_classifier(clf_config["type"], C, clf_config["params"])
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_val_s)
        f1 = f1_score(y_val, y_pred, average="macro")
        logging.info(f"  C={C}: val macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_c = C

    logging.info(f"  Best C={best_c} with val macro-F1={best_f1:.4f}")
    return best_c, best_f1


def run_baselines(
    X_test: np.ndarray,
    y_test: np.ndarray,
    word_counts_test: np.ndarray | None = None,
    feature_dim: int = 768,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    word_counts_train: np.ndarray | None = None,
) -> dict:
    """Run baseline classifiers on test set.

    Baselines:
    1. Random: predict uniformly at random.
    2. Majority: predict most common class.
    3. Word-count-only: LogReg on word_count feature alone.
    4. Random-embedding: LogReg on random vectors with same dimensionality.

    Returns:
        Dict of baseline_name -> metrics dict.
    """
    rng = np.random.RandomState(42)
    n_test = len(y_test)
    n_classes = len(LABEL_NAMES)
    baselines = {}

    # 1. Random baseline
    y_random = rng.randint(0, n_classes, size=n_test)
    baselines["random"] = {
        "macro_f1": float(f1_score(y_test, y_random, average="macro")),
        "accuracy": float(accuracy_score(y_test, y_random)),
    }
    logging.info(f"Random baseline: macro-F1 = {baselines['random']['macro_f1']:.4f}")

    # 2. Majority baseline
    from collections import Counter

    majority_label = Counter(y_test.tolist()).most_common(1)[0][0]
    y_majority = np.full(n_test, majority_label)
    baselines["majority"] = {
        "macro_f1": float(f1_score(y_test, y_majority, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, y_majority)),
    }
    logging.info(f"Majority baseline: macro-F1 = {baselines['majority']['macro_f1']:.4f}")

    # 3. Word-count-only baseline
    if word_counts_train is not None and word_counts_test is not None:
        wc_train = word_counts_train.reshape(-1, 1)
        wc_test = word_counts_test.reshape(-1, 1)
        scaler = StandardScaler()
        wc_train_s = scaler.fit_transform(wc_train)
        wc_test_s = scaler.transform(wc_test)
        clf_wc = LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=42
        )
        clf_wc.fit(wc_train_s, y_train)
        y_wc_pred = clf_wc.predict(wc_test_s)
        baselines["word_count_only"] = {
            "macro_f1": float(f1_score(y_test, y_wc_pred, average="macro")),
            "accuracy": float(accuracy_score(y_test, y_wc_pred)),
            "per_class_f1": {
                name: float(f)
                for name, f in zip(LABEL_NAMES, f1_score(y_test, y_wc_pred, average=None))
            },
            "confusion_matrix": confusion_matrix(y_test, y_wc_pred).tolist(),
        }
        logging.info(
            f"Word-count-only baseline: macro-F1 = {baselines['word_count_only']['macro_f1']:.4f}"
        )
    else:
        logging.warning("Word counts not available, skipping word-count-only baseline.")

    # 4. Random-embedding baseline
    if X_train is not None:
        n_train = len(y_train)
        X_rand_train = rng.randn(n_train, feature_dim).astype(np.float32)
        X_rand_test = rng.randn(n_test, feature_dim).astype(np.float32)
        scaler = StandardScaler()
        X_rand_train_s = scaler.fit_transform(X_rand_train)
        X_rand_test_s = scaler.transform(X_rand_test)
        clf_rand = LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", random_state=42
        )
        clf_rand.fit(X_rand_train_s, y_train)
        y_rand_pred = clf_rand.predict(X_rand_test_s)
        baselines["random_embedding"] = {
            "macro_f1": float(f1_score(y_test, y_rand_pred, average="macro")),
            "accuracy": float(accuracy_score(y_test, y_rand_pred)),
        }
        logging.info(
            f"Random-embedding baseline: macro-F1 = {baselines['random_embedding']['macro_f1']:.4f}"
        )

    return baselines


def run_binary_fallback(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Run binary classification keeping only macro (0) and micro (2).

    Filters out meso (1) samples, retrains LogReg, evaluates.

    Returns:
        Metrics dict for binary macro-vs-micro.
    """
    # Filter to macro and micro only
    train_mask = np.isin(y_train, [0, 2])
    test_mask = np.isin(y_test, [0, 2])

    X_train_bin = X_train[train_mask]
    y_train_bin = y_train[train_mask]
    X_test_bin = X_test[test_mask]
    y_test_bin = y_test[test_mask]

    logging.info(
        f"Binary fallback: train={len(y_train_bin)}, test={len(y_test_bin)} "
        f"(macro + micro only)"
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_bin)
    X_test_s = scaler.transform(X_test_bin)

    clf = LogisticRegression(
        C=1.0, max_iter=2000, solver="lbfgs", random_state=42
    )
    clf.fit(X_train_s, y_train_bin)
    y_pred = clf.predict(X_test_s)

    binary_labels = ["macro", "micro"]
    results = {
        "accuracy": float(accuracy_score(y_test_bin, y_pred)),
        "macro_f1": float(f1_score(y_test_bin, y_pred, average="macro")),
        "per_class_f1": {
            name: float(f)
            for name, f in zip(
                binary_labels,
                f1_score(y_test_bin, y_pred, average=None),
            )
        },
        "confusion_matrix": confusion_matrix(y_test_bin, y_pred).tolist(),
        "classification_report": classification_report(
            y_test_bin, y_pred, target_names=binary_labels, output_dict=True
        ),
        "n_train": len(y_train_bin),
        "n_test": len(y_test_bin),
    }

    logging.info(f"Binary fallback macro-F1 = {results['macro_f1']:.4f}")
    return results


def run_all_probes(
    config: dict,
    project_root: Path | None = None,
) -> dict:
    """Load embeddings and splits, run all probes, save results.

    Returns:
        Dict with all results.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / config["data_dir"]
    results_dir = data_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = data_dir / "embeddings"

    # Load splits
    splits = load_splits(data_dir / "splits.json")

    # Load spans for word counts
    sh0_dir = project_root / config["sh0_data_dir"]
    spans = load_spans(sh0_dir / config["primary_dataset"])
    word_counts = np.array([s.get("word_count", len(s["text"].split())) for s in spans])

    train_idx = np.array(splits["train"])
    val_idx = np.array(splits["val"])
    test_idx = np.array(splits["test"])

    word_counts_train = word_counts[train_idx]
    word_counts_test = word_counts[test_idx]

    all_results = {}

    # Run probes for each model
    for model_key in config["models"]:
        npz_path = embeddings_dir / f"{model_key}_length_matched.npz"
        if not npz_path.exists():
            logging.warning(f"Embeddings not found for {model_key}, skipping probes.")
            continue

        logging.info(f"\n{'='*60}")
        logging.info(f"Running probes for model: {model_key}")
        logging.info(f"{'='*60}")

        data = np.load(npz_path, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]

        X_train = embeddings[train_idx]
        y_train = labels[train_idx]
        X_val = embeddings[val_idx]
        y_val = labels[val_idx]
        X_test = embeddings[test_idx]
        y_test = labels[test_idx]

        model_results = {}

        for clf_key, clf_config in config["classifiers"].items():
            logging.info(f"\n--- {model_key} + {clf_key} ---")

            # Hyperparameter sweep
            C_values = clf_config.get("C_search", [1.0])
            best_c, best_val_f1 = run_hyperparameter_sweep(
                X_train, y_train, X_val, y_val, C_values, clf_config
            )

            # Full evaluation with best C
            metrics = train_and_evaluate(
                X_train, y_train, X_val, y_val, X_test, y_test, clf_config, C=best_c
            )

            logging.info(
                f"  Test macro-F1 = {metrics['test']['macro_f1']:.4f} "
                f"(C={best_c})"
            )

            model_results[clf_key] = metrics

        all_results[model_key] = model_results

    # Run baselines using the first available model's embeddings
    first_model = None
    for model_key in config["models"]:
        npz_path = embeddings_dir / f"{model_key}_length_matched.npz"
        if npz_path.exists():
            first_model = model_key
            break

    if first_model:
        logging.info(f"\n{'='*60}")
        logging.info("Running baselines")
        logging.info(f"{'='*60}")

        data = np.load(embeddings_dir / f"{first_model}_length_matched.npz", allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"]
        feature_dim = embeddings.shape[1]

        X_train = embeddings[train_idx]
        y_train = labels[train_idx]
        X_test = embeddings[test_idx]
        y_test = labels[test_idx]

        all_results["baselines"] = run_baselines(
            X_test=X_test,
            y_test=y_test,
            word_counts_test=word_counts_test,
            feature_dim=feature_dim,
            X_train=X_train,
            y_train=y_train,
            word_counts_train=word_counts_train,
        )

    # Save results
    results_path = results_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Results saved to {results_path}")

    return all_results
