#!/usr/bin/env python
"""Step 3b: Train question-specific SLoD probe and classify test queries (v2).

This replaces the domain-mismatched SH1 document-span probe with a probe trained
directly on QASPER validation questions, using SLoD labels derived from gold
evidence matched to SH0-labeled spans.

Pipeline:
1. Derive question SLoD labels from gold evidence ↔ SH0 span matching
2. Train/validate probe on 80/20 split of labeled validation questions
3. Compare against SH1 probe on same probe-val set
4. Retrain on full validation set and classify test questions
5. Save predictions to data/query_slod_predictions_v2.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.query_classifier_v2 import train_question_probe, classify_test_queries


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")

    # Phase 1: Train and validate (80/20 split)
    print("\n" + "=" * 70)
    print("PHASE 1: Train/validate question-specific SLoD probe")
    print("=" * 70)
    result = train_question_probe(config, project_root)
    print(f"\nProbe validation macro-F1: {result['macro_f1']:.4f}")

    # Phase 2: Retrain on all validation, classify test
    print("\n" + "=" * 70)
    print("PHASE 2: Classify test queries with v2 probe")
    print("=" * 70)
    classify_test_queries(config, project_root)

    print("\nDone. Predictions saved to data/query_slod_predictions_v2.json")


if __name__ == "__main__":
    main()
