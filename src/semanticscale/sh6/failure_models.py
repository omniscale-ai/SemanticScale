"""Stage-5 model registry for failure-prediction comparison.

A small registry that wraps the incumbent logistic-regression pipeline and
new models (LightGBM, MiniRocket, regression heads) behind a uniform interface:

    spec = MODEL_REGISTRY[name]
    result = spec.fit_predict_oof(X, y, cv, random_state)

Each ``OOFResult`` carries OOF predictions aligned to the input row order,
fold assignments, per-fold sklearn-style metric arrays, and an optional
feature-importance frame. That is everything callers need to:

- pair OOF predictions across models for paired bootstrap on Δ-AUC or Δ-ρ
  (random_state and CV identical → folds line up exactly),
- recompute any threshold-dependent metric without re-fitting,
- produce per-model report tables.

The protocol that fixes which classification models / hyperparameters /
decision rules are in scope lives in
``experiments/sh6_llm-pairwise-slod/DESIGN-stage5-models.md``. The regression
specs (``ridge``, ``lightgbm_reg``) are an additive extension used by
``08_score_regression.py`` to predict FrontierScience rubric scores; their
hyperparameters mirror the classification specs so any AUC-vs-ρ contrast is
not confounded by capacity differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TaskType = Literal["classification", "regression"]

CLASSIFICATION_SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1": "f1",
}


def _spearman_score(y_true, y_pred) -> float:
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if np.isfinite(rho) else 0.0


REGRESSION_SCORING = {
    "r2": "r2",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
    "neg_root_mean_squared_error": "neg_root_mean_squared_error",
    "spearman": make_scorer(_spearman_score, greater_is_better=True),
}


# Backwards-compatibility alias so callers that imported `SCORING` keep working.
SCORING = CLASSIFICATION_SCORING


@dataclass
class OOFResult:
    """Cross-validated output for one model on one feature set.

    All arrays are aligned to the row order of the input ``X``. For
    classification, ``predictions`` holds out-of-fold positive-class
    probabilities (length-N float array). For regression, ``predictions``
    holds out-of-fold continuous predictions on the original target scale.
    ``fold_assignments[i]`` is the fold index in which row ``i`` served as test.
    ``confusion_matrix`` is populated for classification only; regression sets
    it to an empty dict.
    """

    model_name: str
    task_type: TaskType
    n_features: int
    feature_cols: list[str]
    predictions: np.ndarray
    fold_assignments: np.ndarray
    fold_metrics: dict[str, dict[str, float]]
    confusion_matrix: dict[str, int]
    feature_importance: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)

    # Compatibility shim: old callers read .probabilities. Map to .predictions
    # so existing classification code paths keep working unchanged.
    @property
    def probabilities(self) -> np.ndarray:
        return self.predictions


class ModelSpec(Protocol):
    """Adapter contract for a Stage-5 model.

    Implementations build their own sklearn-compatible estimator. The shared
    driver below handles fold assignment, metric scoring, and per-task
    bookkeeping so individual specs only declare the estimator and (optionally)
    extract feature importance.
    """

    name: str
    task_type: TaskType

    def build_estimator(self, random_state: int):
        ...

    def feature_importance(self, fitted_estimator, feature_cols: list[str]) -> pd.DataFrame | None:
        ...


def _summarise_scores(scores: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric = key.removeprefix("test_")
        arr = np.asarray(values, dtype=float)
        summary[metric] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    return summary


def _fold_assignment(cv: BaseCrossValidator, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Materialise the fold index for every row, in the same order as ``X``."""
    folds = np.full(len(X), -1, dtype=int)
    for fold_idx, (_, test_idx) in enumerate(cv.split(X, y)):
        folds[test_idx] = fold_idx
    if (folds < 0).any():
        msg = "CV split did not cover every row; check CV settings."
        raise RuntimeError(msg)
    return folds


def run_oof(
    spec: ModelSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: BaseCrossValidator,
    random_state: int,
) -> OOFResult:
    """Common cross-validation driver shared by every registered model.

    Dispatches on ``spec.task_type``: classification produces positive-class
    OOF probabilities and a confusion matrix; regression produces continuous
    OOF predictions with R²/MAE/Spearman fold metrics.
    """
    estimator = spec.build_estimator(random_state)
    fold_assignments = _fold_assignment(cv, X, y)
    cv_iter = list(cv.split(X, y))
    task_type = getattr(spec, "task_type", "classification")

    if task_type == "classification":
        scoring = CLASSIFICATION_SCORING
        predict_method = "predict_proba"
    elif task_type == "regression":
        scoring = REGRESSION_SCORING
        predict_method = "predict"
    else:
        msg = f"Unknown task_type={task_type!r} on spec {spec.name}"
        raise ValueError(msg)

    scores = cross_validate(
        estimator,
        X,
        y,
        cv=cv_iter,
        scoring=scoring,
        n_jobs=None,
        return_estimator=True,
    )

    if task_type == "classification":
        preds = cross_val_predict(estimator, X, y, cv=cv_iter, method=predict_method)[:, 1]
        hard = (preds >= 0.5).astype(int)
        tp = int(np.sum((hard == 1) & (y == 1)))
        tn = int(np.sum((hard == 0) & (y == 0)))
        fp = int(np.sum((hard == 1) & (y == 0)))
        fn = int(np.sum((hard == 0) & (y == 1)))
        confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    else:
        preds = cross_val_predict(estimator, X, y, cv=cv_iter, method=predict_method)
        confusion = {}

    importance: pd.DataFrame | None = None
    fitted = scores.get("estimator")
    if fitted is not None and len(fitted) > 0:
        per_fold = [spec.feature_importance(est, list(X.columns)) for est in fitted]
        per_fold = [df for df in per_fold if df is not None]
        if per_fold:
            stacked = pd.concat(per_fold, ignore_index=True)
            importance = (
                stacked.groupby("feature", as_index=False)
                .agg(mean_importance=("importance", "mean"), std_importance=("importance", "std"))
                .sort_values("mean_importance", ascending=False, ignore_index=True)
            )

    return OOFResult(
        model_name=spec.name,
        task_type=task_type,
        n_features=X.shape[1],
        feature_cols=list(X.columns),
        predictions=preds,
        fold_assignments=fold_assignments,
        fold_metrics=_summarise_scores(scores),
        confusion_matrix=confusion,
        feature_importance=importance,
    )


# ----------------------------- model adapters ----------------------------- #


@dataclass
class LogRegSpec:
    """Median-imputed, z-scaled, class-balanced L2 logreg.

    Identical to the incumbent Stage-5 baseline in ``failure_analysis._make_estimator``;
    re-implemented here only so it lives behind the same registry interface as
    the new models. The unit test in Step 4 should confirm that running this
    spec on a fixed dataset reproduces the existing logreg metrics bit-for-bit.
    """

    name: str = "logreg"
    task_type: TaskType = "classification"

    def build_estimator(self, random_state: int):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def feature_importance(self, fitted_estimator, feature_cols: list[str]) -> pd.DataFrame | None:
        clf: LogisticRegression = fitted_estimator.named_steps["logreg"]
        coefs = clf.coef_[0]
        return pd.DataFrame(
            {"feature": feature_cols, "importance": np.abs(coefs), "signed": coefs}
        )


@dataclass
class LightGBMSpec:
    """Median-imputed gradient boosting.

    Hyperparameters are pinned by ``DESIGN-stage5-models.md`` and must not be
    tuned per-dataset in this protocol. ``class_weight="balanced"`` matches the
    logreg spec so AUC differences come from model capacity, not class
    weighting. ``feature_fraction=1.0`` and ``bagging_fraction=1.0`` are kept
    explicit to make any future tweak visible in code review.
    """

    name: str = "lightgbm"
    task_type: TaskType = "classification"
    n_estimators: int = 300
    num_leaves: int = 31
    learning_rate: float = 0.05
    min_child_samples: int = 10
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0

    def build_estimator(self, random_state: int):
        from lightgbm import LGBMClassifier

        # set_output("pandas") preserves column names through the imputer so
        # LightGBM's fit-time and predict-time feature names line up.
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
                (
                    "lgbm",
                    LGBMClassifier(
                        n_estimators=self.n_estimators,
                        num_leaves=self.num_leaves,
                        learning_rate=self.learning_rate,
                        min_child_samples=self.min_child_samples,
                        feature_fraction=self.feature_fraction,
                        bagging_fraction=self.bagging_fraction,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    def feature_importance(self, fitted_estimator, feature_cols: list[str]) -> pd.DataFrame | None:
        from lightgbm import LGBMClassifier

        clf: LGBMClassifier = fitted_estimator.named_steps["lgbm"]
        gain = clf.booster_.feature_importance(importance_type="gain")
        return pd.DataFrame(
            {"feature": feature_cols, "importance": gain.astype(float), "signed": gain.astype(float)}
        )


@dataclass
class RidgeSpec:
    """L2 ridge regressor — regression mirror of LogRegSpec.

    Same imputation + z-scaling stack so ridge and logreg differ only in the
    final estimator. Default ``alpha=1.0`` matches sklearn's default and is not
    tuned per run; if regularization needs lifting on small folds, that's a
    follow-up change with its own pre-registration.
    """

    name: str = "ridge"
    task_type: TaskType = "regression"
    alpha: float = 1.0

    def build_estimator(self, random_state: int):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "ridge",
                    Ridge(alpha=self.alpha, random_state=random_state),
                ),
            ]
        )

    def feature_importance(self, fitted_estimator, feature_cols: list[str]) -> pd.DataFrame | None:
        reg: Ridge = fitted_estimator.named_steps["ridge"]
        coefs = reg.coef_
        return pd.DataFrame(
            {"feature": feature_cols, "importance": np.abs(coefs), "signed": coefs}
        )


@dataclass
class LGBMRegSpec:
    """Median-imputed gradient boosting regressor — regression mirror of LightGBMSpec.

    Hyperparameters are intentionally identical to the classifier spec so any
    capacity difference between the two task types is visible without
    confounding from differing tree counts / leaves / learning rate.
    """

    name: str = "lightgbm_reg"
    task_type: TaskType = "regression"
    n_estimators: int = 300
    num_leaves: int = 31
    learning_rate: float = 0.05
    min_child_samples: int = 10
    feature_fraction: float = 1.0
    bagging_fraction: float = 1.0

    def build_estimator(self, random_state: int):
        from lightgbm import LGBMRegressor

        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
                (
                    "lgbm",
                    LGBMRegressor(
                        n_estimators=self.n_estimators,
                        num_leaves=self.num_leaves,
                        learning_rate=self.learning_rate,
                        min_child_samples=self.min_child_samples,
                        feature_fraction=self.feature_fraction,
                        bagging_fraction=self.bagging_fraction,
                        random_state=random_state,
                        n_jobs=1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    def feature_importance(self, fitted_estimator, feature_cols: list[str]) -> pd.DataFrame | None:
        from lightgbm import LGBMRegressor

        reg: LGBMRegressor = fitted_estimator.named_steps["lgbm"]
        gain = reg.booster_.feature_importance(importance_type="gain")
        return pd.DataFrame(
            {"feature": feature_cols, "importance": gain.astype(float), "signed": gain.astype(float)}
        )


MODEL_REGISTRY: dict[str, Callable[[], ModelSpec]] = {
    "logreg": LogRegSpec,
    "lightgbm": LightGBMSpec,
    "ridge": RidgeSpec,
    "lightgbm_reg": LGBMRegSpec,
}


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        msg = f"Unknown model spec '{name}'. Registered: {sorted(MODEL_REGISTRY)}"
        raise KeyError(msg)
    return MODEL_REGISTRY[name]()


def default_cv(task_type: TaskType, n_splits: int, random_state: int) -> BaseCrossValidator:
    """Pick the canonical CV splitter for a task type.

    Classification uses ``StratifiedKFold`` (matches the incumbent protocol).
    Regression uses plain ``KFold``; stratification is not meaningful for a
    continuous target.
    """
    if task_type == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if task_type == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    msg = f"Unknown task_type={task_type!r}"
    raise ValueError(msg)
