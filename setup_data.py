#!/usr/bin/env python3
"""Download SemanticScale experiment data from Hugging Face Hub.

Usage:
    python setup_data.py                          # Download all data (~3.4 GB)
    python setup_data.py --experiments sh0 sh1    # Download specific experiments
    python setup_data.py --list                   # List available experiments
"""

import argparse
import os
import sys

REPO_ID = "anaderi/semantic-scale-data"
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

EXPERIMENTS = {
    "sh0": {"path": "SH0", "size": "155 MB", "desc": "QASPER spans with SLoD labels"},
    "sh1": {"path": "SH1", "size": "315 MB", "desc": "Embeddings and annotations"},
    "sh2": {"path": "SH2", "size": "172 MB", "desc": "Steering vectors and answers"},
    "sh3": {"path": "SH3", "size": "2.1 GB", "desc": "Retrieval embeddings and results"},
    "sh4": {"path": "SH4", "size": "390 MB", "desc": "Extractions and features"},
    "sh5": {"path": "SH5", "size": "16 MB", "desc": "CoT traces and answer scores"},
    "sh5a": {"path": "SH5a", "size": "~5 MB", "desc": "Transition matrix results"},
    "sh5c": {"path": "SH5c", "size": "~5 MB", "desc": "Context alignment results"},
    "sh5d": {"path": "SH5d", "size": "~15 MB", "desc": "Continuous projection results"},
}


def list_experiments():
    print("Available experiments:\n")
    print(f"  {'Name':<8} {'Size':<10} Description")
    print(f"  {'----':<8} {'----':<10} -----------")
    for name, info in EXPERIMENTS.items():
        print(f"  {name:<8} {info['size']:<10} {info['desc']}")
    print(f"\n  Total: ~3.4 GB")


def download_experiment(name, data_dir):
    """Download a single experiment's data from HF Hub."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    info = EXPERIMENTS[name]
    hf_path = info["path"]
    local_dir = os.path.join(data_dir, name)

    print(f"  Downloading {name} ({info['size']})...")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=local_dir,
        allow_patterns=f"{hf_path}/**",
    )

    # Move files from subfolder to target directory if needed
    nested = os.path.join(local_dir, hf_path)
    if os.path.isdir(nested) and nested != local_dir:
        import shutil
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(local_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        os.rmdir(nested)

    print(f"  ✓ {name} -> {local_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Download SemanticScale experiment data from Hugging Face Hub."
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=list(EXPERIMENTS.keys()),
        help="Specific experiments to download (default: all)",
    )
    parser.add_argument(
        "--data-dir", default=os.environ.get("SLOD_DATA_ROOT", DEFAULT_DATA_DIR),
        help=f"Target directory for data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_exps",
        help="List available experiments and exit",
    )

    args = parser.parse_args()

    if args.list_exps:
        list_experiments()
        return

    experiments = args.experiments or list(EXPERIMENTS.keys())
    data_dir = args.data_dir

    os.makedirs(data_dir, exist_ok=True)

    print(f"Downloading {len(experiments)} experiment(s) to {data_dir}\n")

    for name in experiments:
        download_experiment(name, data_dir)

    print(f"\nDone. Data is in {data_dir}")
    print("Set SLOD_DATA_ROOT environment variable if using a non-default location.")


if __name__ == "__main__":
    main()
