"""ProcessBench dataset loader.

Qwen/ProcessBench provides reasoning traces that are already chunked into
steps, together with a per-trace label marking the index of the first
erroneous step (-1 means the trace is fully correct). So Stage 1 here is
just "load and normalise" — no LLM inference is needed.

See https://huggingface.co/datasets/Qwen/ProcessBench.

Fields per row:
    id                       str        e.g. "gsm8k-0"
    generator                str        model that produced the trace
    problem                  str        problem statement
    steps                    list[str]  reasoning chunks
    final_answer_correct     bool
    label                    int        first erroneous step index; -1 if correct
"""

from __future__ import annotations

import datetime
from datetime import timezone
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_NAME = "processbench"
SLICE_NAME = "generator"

_ALL_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")


def slice_label(trace: dict) -> str | None:
    """Return the trace's generating model for per-model analysis grouping."""
    gen = trace.get("generator")
    return str(gen) if gen else None


def _subsets(config: dict, overrides: dict) -> list[str]:
    ds_cfg = config.get("dataset", {})
    subsets = overrides.get("subsets") or ds_cfg.get("subsets")
    if subsets in (None, "all"):
        return list(_ALL_SUBSETS)
    return list(subsets)


def _generators(config: dict, overrides: dict) -> list[str] | None:
    ds_cfg = config.get("dataset", {})
    gens = overrides.get("generators")
    if gens is None:
        gens = ds_cfg.get("generators")
    if gens in (None, "all"):
        return None
    return list(gens)


def run_slug(config: dict, overrides: dict) -> str:
    """Identifier for a ProcessBench run: subsets (+ optional generator filter)."""
    subsets = _subsets(config, overrides)
    slug = "+".join(sorted(subsets))
    gens = _generators(config, overrides)
    if gens:
        slug += "_gen-" + "+".join(sorted(gens))
    return slug


def produce_traces(
    config: dict,
    project_root: Path,
    overrides: dict,
) -> list[dict]:
    """Stage 1: load pre-existing ProcessBench traces and normalise them."""
    from datasets import load_dataset  # deferred: heavy dep

    hf_path = config.get("dataset", {}).get("hf_path", "Qwen/ProcessBench")
    subsets = _subsets(config, overrides)
    gen_filter = _generators(config, overrides)
    max_samples = overrides.get("max_samples")
    if max_samples is None:
        max_samples = config.get("dataset", {}).get("max_samples")

    slug = run_slug(config, overrides)
    ts = datetime.datetime.now(tz=timezone.utc).isoformat()

    name = config.get("dataset", {}).get("name", DATASET_NAME)
    traces: list[dict] = []
    for subset in subsets:
        logger.info("Loading %s split=%s", hf_path, subset)
        ds = load_dataset(hf_path, split=subset)
        for row in ds:
            if gen_filter and row["generator"] not in gen_filter:
                continue
            label = int(row["label"])
            is_correct = label == -1
            steps = [str(s) for s in row["steps"]]
            traces.append(
                {
                    "id": str(row["id"]),
                    "dataset": name,
                    "run_slug": slug,
                    "problem": str(row["problem"]),
                    "subject": subset,
                    "correct_answer": None,
                    "reasoning_text": None,
                    "answer_text": None,
                    "reasoning_chunks": steps,
                    "answer_chunks": None,
                    "is_correct": is_correct,
                    "error_step_index": None if is_correct else label,
                    "final_answer_correct": bool(row["final_answer_correct"]),
                    "generator": str(row["generator"]),
                    "model": None,
                    "usage": None,
                    "error": None,
                    "timestamp": ts,
                    # placeholder so downstream analysis that filters on these
                    # frontierscience-specific keys still sees something sane
                    "has_final_answer": None,
                    "grade": None,
                }
            )

    if max_samples is not None:
        traces = traces[:max_samples]

    logger.info(
        "Loaded %d ProcessBench traces (subsets=%s, generators=%s)",
        len(traces), subsets, gen_filter or "all",
    )
    return traces
