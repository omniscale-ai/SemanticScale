#!/usr/bin/env python
"""SH6 — End-to-end orchestrator for one config or all configs.

Runs stages 01_traces → 05_analyze_failure_modes (and optionally 06, 07)
sequentially. In single-config mode, stage scripts that accept ``--config`` are
invoked with the selected config. In all-configs mode, config-aware stages are
run for every YAML under ``config/``. Stage 06 is cross-dataset and runs once.

Usage:
    uv run --env-file .env python experiments/sh6_llm-pairwise-slod/scripts/run_sh6.py \\
        --config experiments/sh6_llm-pairwise-slod/config/frontierscience-deepseek.yaml

    # Resume from a stage:
    ... run_sh6.py --config <cfg> --start-from 3

    # Also run advanced (stage 07) failure analysis:
    ... run_sh6.py --config <cfg> --include-advanced

    # Run stages 3..6 for all configs, including model comparison artifacts:
    ... run_sh6.py --all-configs --start-from 3 --stop-at 6 --include-05b
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = SCRIPTS_DIR.parent / "config"

STAGES = [
    (1, "01_traces.py"),
    (2, "02_slod.py"),
    (3, "03_analyze_accuracy.py"),
    (4, "04_plot_trajectories.py"),
    (5, "05_analyze_failure_modes.py"),
]

ANCHOR_STAGE = (6, "06_anchor_validation.py")
ADVANCED_STAGE = (7, "07_advanced_failure_analysis.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Path to SH6 config YAML")
    parser.add_argument("--all-configs", action="store_true", dest="all_configs",
                        help="Run config-aware stages for every YAML in config/")
    parser.add_argument("--start-from", type=int, default=1, dest="start_from",
                        help="First stage number to run (default: 1)")
    parser.add_argument("--stop-at", type=int, default=5, dest="stop_at",
                        help="Last stage number to run, inclusive (default: 5; set 6 to include anchor validation)")
    parser.add_argument("--include-05b", action="store_true", dest="include_05b",
                        help="Also run Stage-05 model comparison artifacts (logreg + lightgbm)")
    parser.add_argument("--include-advanced", action="store_true", dest="include_advanced",
                        help="Also run stage 07 (advanced failure analysis) after stage 5")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Print commands without executing")
    args = parser.parse_args()

    if args.all_configs and args.config:
        parser.error("Use either --config or --all-configs, not both")
    if not args.all_configs and not args.config:
        parser.error("Provide --config for single-config mode, or use --all-configs")
    if args.start_from > args.stop_at:
        parser.error("--start-from must be <= --stop-at")
    if args.start_from < 1 or args.stop_at > 7:
        parser.error("Supported stage range is 1..7")
    return args


def run_stage(
    stage_num: int,
    script_name: str,
    config_path: Path | None,
    dry_run: bool,
    extra_args: list[str] | None = None,
) -> None:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])
    if extra_args:
        cmd.extend(extra_args)
    print("\n" + "=" * 70)
    print(f"STAGE {stage_num:02d}: {script_name}")
    print("=" * 70)
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        raise SystemExit(f"\nStage {stage_num:02d} ({script_name}) failed "
                         f"with exit code {exc.returncode} after {elapsed:.1f}s") from exc
    elapsed = time.time() - t0
    print(f"\nStage {stage_num:02d} done in {elapsed:.1f}s")


def discover_configs() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def run_for_single_config(
    args: argparse.Namespace,
    config_path: Path,
    run_anchor_stage: bool = True,
) -> None:
    print(f"Config: {config_path}")
    print(f"Stages: {args.start_from}..{args.stop_at}"
          + (" + 05b" if args.include_05b else "")
          + (" + 7" if args.include_advanced else ""))

    for num, script in STAGES:
        if num < args.start_from or num > args.stop_at:
            continue
        extra_args: list[str] | None = None
        if num == 5 and args.include_05b:
            extra_args = ["--models", "logreg", "lightgbm"]
        run_stage(num, script, config_path, args.dry_run, extra_args=extra_args)

    if run_anchor_stage and args.start_from <= ANCHOR_STAGE[0] <= args.stop_at:
        num, script = ANCHOR_STAGE
        run_stage(num, script, None, args.dry_run,
                  extra_args=["--llm-config", str(config_path)])

    if args.include_advanced and args.start_from <= ADVANCED_STAGE[0] <= args.stop_at:
        num, script = ADVANCED_STAGE
        run_stage(num, script, config_path, args.dry_run)


def run_for_all_configs(args: argparse.Namespace) -> None:
    configs = discover_configs()
    if not configs:
        raise SystemExit(f"No config YAML files found under {CONFIGS_DIR}")

    print(f"Configs: {len(configs)} found under {CONFIGS_DIR}")
    print(f"Stages: {args.start_from}..{args.stop_at}"
          + (" + 05b" if args.include_05b else "")
          + (" + 7" if args.include_advanced else ""))

    for idx, cfg in enumerate(configs, start=1):
        print("\n" + "#" * 70)
        print(f"CONFIG {idx:02d}/{len(configs)}: {cfg}")
        print("#" * 70)
        cfg = cfg.resolve()
        run_for_single_config(args, cfg, run_anchor_stage=False)

    # Stage 06 is cross-dataset; run it once after config-aware stages.
    if args.start_from <= ANCHOR_STAGE[0] <= args.stop_at:
        num, script = ANCHOR_STAGE
        run_stage(num, script, None, args.dry_run,
                  extra_args=["--llm-config", str(configs[0].resolve())])


def main() -> None:
    args = parse_args()

    overall_t0 = time.time()
    if args.all_configs:
        run_for_all_configs(args)
    else:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            raise SystemExit(f"Config not found: {config_path}")
        run_for_single_config(args, config_path)

    print("\n" + "=" * 70)
    print(f"All requested stages completed in {time.time() - overall_t0:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
