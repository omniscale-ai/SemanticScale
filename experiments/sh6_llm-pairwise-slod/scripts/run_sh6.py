#!/usr/bin/env python
"""SH6 — End-to-end orchestrator: run all pipeline stages for one config.

Runs stages 01_traces → 05_analyze_failure_modes (and optionally 07_advanced)
sequentially, passing the same ``--config`` to each. Stops on the first
non-zero exit. Stage 06 (anchor validation) is a cross-dataset diagnostic and
is intentionally excluded.

Usage:
    uv run --env-file .env python experiments/sh6_llm-pairwise-slod/scripts/run_sh6.py \\
        --config experiments/sh6_llm-pairwise-slod/config/frontierscience-deepseek.yaml

    # Resume from a stage:
    ... run_sh6.py --config <cfg> --start-from 3

    # Also run advanced (stage 07) failure analysis:
    ... run_sh6.py --config <cfg> --include-advanced
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

STAGES = [
    (1, "01_traces.py"),
    (2, "02_slod.py"),
    (3, "03_analyze_accuracy.py"),
    (4, "04_plot_trajectories.py"),
    (5, "05_analyze_failure_modes.py"),
]

ADVANCED_STAGE = (7, "07_advanced_failure_analysis.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to SH6 config YAML")
    parser.add_argument("--start-from", type=int, default=1, dest="start_from",
                        help="First stage number to run (default: 1)")
    parser.add_argument("--stop-at", type=int, default=5, dest="stop_at",
                        help="Last stage number to run, inclusive (default: 5)")
    parser.add_argument("--include-advanced", action="store_true", dest="include_advanced",
                        help="Also run stage 07 (advanced failure analysis) after stage 5")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print commands without executing")
    return parser.parse_args()


def run_stage(stage_num: int, script_name: str, config_path: Path, dry_run: bool) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name), "--config", str(config_path)]
    print("\n" + "=" * 70)
    print(f"STAGE {stage_num:02d}: {script_name}")
    print("=" * 70)
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise SystemExit(f"\nStage {stage_num:02d} ({script_name}) failed "
                         f"with exit code {result.returncode} after {elapsed:.1f}s")
    print(f"\nStage {stage_num:02d} done in {elapsed:.1f}s")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    print(f"Config: {config_path}")
    print(f"Stages: {args.start_from}..{args.stop_at}"
          + (" + 7" if args.include_advanced else ""))

    overall_t0 = time.time()
    for num, script in STAGES:
        if num < args.start_from or num > args.stop_at:
            continue
        run_stage(num, script, config_path, args.dry_run)

    if args.include_advanced:
        num, script = ADVANCED_STAGE
        run_stage(num, script, config_path, args.dry_run)

    print("\n" + "=" * 70)
    print(f"All requested stages completed in {time.time() - overall_t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
