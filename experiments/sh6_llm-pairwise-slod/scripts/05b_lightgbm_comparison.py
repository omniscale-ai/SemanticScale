#!/usr/bin/env python
"""SH6 Stage-5b: LightGBM comparison runner.

Adds a second model on top of the existing logistic-regression baseline,
without touching the incumbent reports. For one run, this script:

1. Locates `trajectory_features.csv` produced by `05_analyze_failure_modes.py`.
2. Calls `run_models_on_run(...)` on both `logreg` and `lightgbm`.
3. Writes `oof_predictions_*.parquet`, `feature_importance_*.csv`, and
   `cv_metadata_*.json` to `reports/{dataset}/{run_slug}/artifacts/`.
4. Logs to `reports/{dataset}/{run_slug}/logs/05b_lightgbm_<ts>.log`.

The Δ-AUC vs incumbent and the paired-bootstrap CI are computed in Step 5
(`05z_aggregate_models.py`) where we have the full set of runs in scope.

Usage:
    # Resolve via config (preferred — same dataset/run_slug logic as 05).
    uv run python experiments/sh6_llm-pairwise-slod/scripts/05b_lightgbm_comparison.py \\
        --config experiments/sh6_llm-pairwise-slod/config/frontierscience-deepseek.yaml

    # Or point directly at a features CSV (bypasses config when run dir
    # was created by hand).
    uv run python experiments/sh6_llm-pairwise-slod/scripts/05b_lightgbm_comparison.py \\
        --features-csv experiments/sh6_llm-pairwise-slod/reports/.../trajectory_features.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

from semanticscale.sh6.failure_analysis_runner import run_models_on_run
from semanticscale.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_MODELS = ("logreg", "lightgbm")


def _resolve_run_dir_from_config(config_path: Path) -> Path:
    """Replicate the directory layout used by 05_analyze_failure_modes.py.

    Done via the same helpers (`ds.run_slug`, `paths.reports_dir`) so the
    new artifacts land alongside the existing report rather than in a
    parallel tree.
    """
    from semanticscale.sh6 import datasets as ds  # local import: shared with 05

    config = load_config(config_path)
    # `load_config` injects `_project_root` = directory of the YAML, which
    # `paths.reports_dir` (e.g. `../reports`) is relative to. Same logic as
    # the incumbent `05_analyze_failure_modes.py`.
    project_root = Path(config["_project_root"])
    dataset_name = config["dataset"]["name"]
    run_slug = ds.run_slug(config)
    reports_dir = (project_root / config["paths"]["reports_dir"]).resolve()
    return reports_dir / dataset_name / run_slug


def _attach_file_log(log_path: Path) -> logging.Handler:
    """Stream INFO logs into the run's logs/ directory for post-mortem audit."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="SH6 config YAML; used to derive run dir.")
    src.add_argument("--features-csv", type=Path, help="Direct path to trajectory_features.csv.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"Models to run (default: {' '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-label", default="auto")
    parser.add_argument("--feature-set", default="trajectory_full")
    args = parser.parse_args()

    setup_logging("INFO")

    if args.config:
        run_dir = _resolve_run_dir_from_config(args.config)
        features_csv = run_dir / "trajectory_features.csv"
    else:
        features_csv = args.features_csv.resolve()
        run_dir = features_csv.parent
    if not features_csv.exists():
        logger.error("Features CSV not found: %s", features_csv)
        return 1

    artifact_dir = run_dir / "artifacts"
    log_path = run_dir / "logs" / f"05b_lightgbm_{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}.log"
    file_handler = _attach_file_log(log_path)

    try:
        logger.info("Run dir: %s", run_dir)
        logger.info("Features CSV: %s", features_csv)
        logger.info("Artifacts dir: %s", artifact_dir)
        logger.info("Models: %s", args.models)

        summary = run_models_on_run(
            features_csv=features_csv,
            artifact_dir=artifact_dir,
            models=list(args.models),
            feature_set=args.feature_set,
            target_label=args.target_label,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            repo_root=Path.cwd(),
        )

        if "_skipped" in summary:
            logger.warning("Run skipped: %s", summary["_skipped"]["reason"])
            return 0

        for model_name, info in summary.items():
            logger.info(
                "%s — AUC %.4f ± %.4f (%.2fs)",
                model_name,
                info["auc_mean"],
                info["auc_std"],
                info["duration_s"],
            )
        return 0
    finally:
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    sys.exit(main())
