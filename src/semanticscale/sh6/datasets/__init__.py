"""Dataset loaders for SH6.

Each dataset provides Stage 1 of the pipeline: producing a list of
**trace records**. A trace record is the unified structure consumed by
Stage 2 (SLoD ranking) and Stage 3 (analysis & plotting).

Trace record schema (all keys present; dataset-specific ones may be None):

    id: str
    dataset: str                    # e.g. "frontierscience" | "processbench"
    run_slug: str                   # identifies the run within a dataset
    problem: str
    subject: str
    correct_answer: str | None

    # free-form traces (frontierscience: produced by inference)
    reasoning_text: str | None
    answer_text: str | None

    # pre-chunked traces (processbench / AgentErrorBench / AgentHallu: provided)
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

from . import (
    agenterrorbench,
    agenthallu,
    frontierscience,
    gpqa_diamond,
    gpqa_diamond_freeform,
    processbench,
    swe_agent_trajectories,
)
from .. import scoring as generic_scoring

_REGISTRY: dict[str, object] = {
    "frontierscience": frontierscience,
    "processbench": processbench,
    "swe-agent-trajectories": swe_agent_trajectories,
    "agenterrorbench": agenterrorbench,
    "agenthallu": agenthallu,
    "gpqa-diamond": gpqa_diamond,
    "gpqa-diamond-freeform": gpqa_diamond_freeform,
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


def resolve_run_dir(
    config: dict,
    data_dir: Path,
    run_slug: str,
    *,
    required_files: tuple[str, ...] = (),
) -> tuple[str, Path]:
    """Resolve a run slug to an on-disk directory.

    SH6 partial FrontierScience runs may live under a suffixed slug like
    ``..._types-research`` even when the config-derived base slug omits that
    suffix. When there is a single matching suffixed directory, prefer it.
    """
    dataset_dir = data_dir / dataset_name(config)
    run_dir = dataset_dir / run_slug
    if run_dir.exists() and all((run_dir / name).exists() for name in required_files):
        return run_slug, run_dir

    parent = run_dir.parent
    if not parent.exists():
        return run_slug, run_dir

    prefix = f"{run_dir.name}_"
    matches = [
        candidate
        for candidate in sorted(parent.iterdir())
        if candidate.is_dir()
        and candidate.name.startswith(prefix)
        and all((candidate / name).exists() for name in required_files)
    ]
    if not matches:
        return run_slug, run_dir
    if len(matches) > 1:
        options = ", ".join(str(path.relative_to(dataset_dir)) for path in matches)
        raise ValueError(
            f"Multiple SH6 run directories match '{run_slug}': {options}. "
            "Pass --run-slug explicitly."
        )

    resolved_dir = matches[0]
    resolved_slug = str(resolved_dir.relative_to(dataset_dir))
    return resolved_slug, resolved_dir


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


def score_results(config: dict, results: list[dict]) -> dict:
    """Score result rows, using a dataset-specific scorer when available."""
    module = _get_module(dataset_name(config))
    fn = getattr(module, "score_results", None)
    if fn is not None:
        return fn(results)
    return generic_scoring.score_results(results)


def slice_name(config: dict) -> str | None:
    """Return the analysis slice dimension for the configured dataset.

    A non-None value means Stages 3/4/5 should produce per-slice reports
    in addition to the global per-run reports. The value is also used as
    the subdirectory prefix (``by-{slice_name}/{value}/``).
    """
    module = _get_module(dataset_name(config))
    return getattr(module, "SLICE_NAME", None)


def slice_label(config: dict, trace: dict) -> str | None:
    """Return the slice label for one trace, or None if it can't be sliced."""
    module = _get_module(dataset_name(config))
    fn = getattr(module, "slice_label", None)
    return fn(trace) if fn else None
