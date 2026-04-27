"""Dataset loaders for SH6.

Each dataset provides Stage 1 of the pipeline: producing a list of
**trace records**. A trace record is the unified structure consumed by
Stage 2 (SLoD ranking) and Stage 3 (analysis & plotting).

Trace record schema (all keys present; dataset-specific ones may be None):

    id: str
    dataset: str                    # "frontierscience" | "processbench"
    run_slug: str                   # identifies the run within a dataset
    problem: str
    subject: str
    correct_answer: str | None

    # free-form traces (frontierscience: produced by inference)
    reasoning_text: str | None
    answer_text: str | None

    # pre-chunked traces (processbench: provided by the dataset)
    reasoning_chunks: list[str] | None
    answer_chunks: list[str] | None

    is_correct: bool | None
    error_step_index: int | None    # processbench: index of first wrong step
                                    # (None means "correct")

    # provenance
    model: str | None               # None when the trace is not LLM-generated
                                    # by us (e.g. processbench pre-existing)
    usage: dict | None
    error: str | None
    timestamp: str
    # plus dataset-specific extras (grade, has_final_answer, generator, ...)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import agenthallu, frontierscience, gpqa_diamond, processbench, swe_agent_trajectories

_REGISTRY: dict[str, object] = {
    "frontierscience": frontierscience,
    "processbench": processbench,
    "swe-agent-trajectories": swe_agent_trajectories,
    "agenthallu": agenthallu,
    "gpqa-diamond": gpqa_diamond,
}


def _get_module(name: str):
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown dataset '{name}'. Known datasets: {known}"
        ) from exc


def dataset_name(config: dict) -> str:
    """Return the configured dataset name, defaulting to frontierscience."""
    return config.get("dataset", {}).get("name", "frontierscience")


def run_slug(config: dict, overrides: dict | None = None) -> str:
    """Return a filesystem-safe identifier for a run within a dataset."""
    module = _get_module(dataset_name(config))
    return module.run_slug(config, overrides or {})


def produce_traces(
    config: dict,
    project_root: Path,
    overrides: dict | None = None,
) -> list[dict]:
    """Stage 1: produce trace records for the configured dataset.

    For datasets that need LLM inference (frontierscience) this runs
    inference + grading. For datasets that already provide traces
    (processbench) this just loads and normalises them.
    """
    module = _get_module(dataset_name(config))
    return module.produce_traces(config, project_root, overrides or {})
