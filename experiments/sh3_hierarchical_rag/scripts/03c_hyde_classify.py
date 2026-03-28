#!/usr/bin/env python
"""Step 3c: HyDE-based query SLoD classification.

Pipeline:
1. Generate hypothetical answers for all QASPER questions using Claude Haiku
2. Embed hypothetical answers with SciBERT ([CLS] token)
3. Classify with SH1 LogReg probe (now in-domain: document-like text)
4. Save predictions to data/query_slod_predictions_hyde.json
5. Compare with v1 and v2 predictions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, setup_logging
from src.hyde_classifier import main as hyde_main


def main():
    setup_logging()
    project_root = Path(__file__).resolve().parent.parent
    config = load_config(project_root / "config.yaml")

    print("\n" + "=" * 70)
    print("HyDE-based Query SLoD Classification")
    print("=" * 70)
    print(f"Model: {config.get('hyde', {}).get('model', 'claude-haiku-4-5-20251001')}")
    print(f"Questions file: data/questions.jsonl")
    print(f"Output: data/{config.get('hyde', {}).get('predictions_file', 'query_slod_predictions_hyde.json')}")
    print()

    hyde_main(config, project_root)

    print("\nDone. HyDE predictions saved.")


if __name__ == "__main__":
    main()
